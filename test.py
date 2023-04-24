import os
import json
import argparse
import itertools
import math
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
import scipy

from utils import commons
from utils import utils
from utils.data_utils import (
    TextAudioLoaderWithDuration,
    TextAudioCollateWithDuration,
    DistributedBucketSampler,
)
from models.models import (
    SynthesizerTrn,
    MultiPeriodDiscriminator,
)
from models.losses import (
    generator_loss,
    discriminator_loss,
    feature_loss,
    kl_loss_dtw,
    kl_loss,
)
from utils.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from text.symbols import symbols

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

global_step = 0


def main():
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."

    n_gpus = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "63331"
    # RuntimeError: Distributed package doesn't have NCCL built in
    # https://github.com/ray-project/ray_lightning/issues/13
    os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"

    hps = utils.get_hparams()
    
    if hps.warmup and not hps.train.use_gt_duration:
        print("'use_gt_duration' option is automatically set to true in warmup phase.")
        hps.train.use_gt_duration = True
    
    if hps.warmup:
        print("'c_kl_fwd' is set to 0 during warmup to learn a reasonable prior distribution.")  
        hps.train.c_kl_fwd = 0

    mp.spawn(
        run,
        nprocs=n_gpus,
        args=(
            n_gpus,
            hps,
        ),
    )


def run(rank, n_gpus, hps):
    global global_step
    
    logger = utils.get_logger(hps.model_dir)
    logger.info(hps)
    utils.check_git_hash(hps.model_dir)
    writer = SummaryWriter(log_dir=hps.model_dir)
    writer_test = SummaryWriter(log_dir=os.path.join(hps.model_dir, "test"))

    # RuntimeError: Distributed package doesn't have NCCL built in
    # https://stackoverflow.com/questions/73730819/how-to-set-backend-to-gloo-on-windows-in-pytorch
    dist.init_process_group(
        backend="gloo", init_method="env://", world_size=n_gpus, rank=rank
    )
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)

    collate_fn = TextAudioCollateWithDuration()
    test_dataset = TextAudioLoaderWithDuration(hps.data.test_files, hps.data)
    test_loader = DataLoader(
        test_dataset,
        num_workers=8,
        shuffle=False,
        # use the same batch size as train
        batch_size=hps.train.batch_size,
        pin_memory=False,
        drop_last=False,
        collate_fn=collate_fn,
    )

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        hps.models,
    ).cuda(rank)
    net_d = MultiPeriodDiscriminator().cuda(rank)
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )

    if not hps.warmup:
        net_g.attach_memory_bank(hps.models)
        optim_g.add_param_group({"params": list(net_g.memory_bank.parameters())})

    try:
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g
        )
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d
        )
        global_step = epoch_str * len(train_loader)
    except Exception as e:
        epoch_str = 0
        global_step = 0

    net_g = DDP(net_g, device_ids=[rank])
    net_d = DDP(net_d, device_ids=[rank])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 1
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 1
    )

    # Call test function
    test(hps, net_g, test_loader, writer_test, epoch=5)


def test(hps, generator, test_loader, writer_test, epoch=0):
    generator.eval()

    save_dir = os.path.join(writer_test.log_dir, f"{epoch}")
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (
            x,
            x_lengths,
            spec,
            spec_lengths,
            y,
            y_lengths,
            duration,
        ) in enumerate(test_loader):
            x, x_lengths = x.cuda(0), x_lengths.cuda(0)
            spec, spec_lengths = spec.cuda(0), spec_lengths.cuda(0)
            y, y_lengths = y.cuda(0), y_lengths.cuda(0)
            # remove else
            x = x[:1]
            x_lengths = x_lengths[:1]
            spec = spec[:1]
            spec_lengths = spec_lengths[:1]
            y = y[:1]
            y_lengths = y_lengths[:1]

            y_hat, mask, *_ = generator.module.infer(x, x_lengths, max_len=1000)
            y_hat_lengths = mask.sum([1, 2]).long() * hps.data.hop_length

            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1).float(),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )

            audio = y_hat[0, 0, : y_hat_lengths[0]].cpu().numpy()
            audio_gt = y[0, 0, : y_lengths[0]].cpu().numpy()
            scipy.io.wavfile.write(
                filename=os.path.join(save_dir, f"{batch_idx}.wav"),
                rate=hps.data.sampling_rate,
                data=audio,
            )
            scipy.io.wavfile.write(
                filename=os.path.join(save_dir, f"{batch_idx}_gt.wav"),
                rate=hps.data.sampling_rate,
                data=audio_gt,
            )

            if batch_idx >= 8:
                break

    image_dict = {
        "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy())
    }
    audio_dict = {"gen/audio": y_hat[0, :, : y_hat_lengths[0]]}
    if global_step == 0:
        image_dict.update(
            {"gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())}
        )
        audio_dict.update({"gt/audio": y[0, :, : y_lengths[0]]})

    utils.summarize(
        writer=writer_test,
        global_step=10,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate,
    )
    print("====> Epoch Evaluate: {}".format(epoch))


if __name__ == "__main__":
    main()
