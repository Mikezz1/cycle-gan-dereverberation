import torch
import torch.nn as nn
from typing import List
import torch.nn
import torch.nn.functional as F
import argparse
from cyclegan import *
from discriminator import *
from dataset import LibriSpeechDataset
import torchaudio
import wandb


def istft_transform(x):
    pad = nn.ConstantPad2d(padding=(0, 1, 0, 0), value=0)
    transform = torchaudio.transforms.InverseSpectrogram(n_fft=398, normalized=True)
    x = pad(x)
    return transform(
        torch.view_as_complex(x.transpose(1, 3).transpose(1, 2).contiguous())
    ).unsqueeze(1)


def train(clip_value):
    generator_r2d.train()
    generator_d2r.train()
    msd_r.train()
    mpd_d.train()

    wav_pad = nn.ConstantPad1d(padding=(0, 80), value=0)

    for epoch in range(EPOCHS):
        for batch_idx, (wav_d, spec_d, wav_r, spec_r) in enumerate(dataloader):
            wav_d.to(device)
            spec_d.to(device)

            wav_r.to(device)
            spec_r.to(device)

            optimizer_msd.zero_grad()
            optimizer_mpd.zero_grad()
            optimizer_g.zero_grad()

            # ------------- Train Discriminator ------------- #

            fake_spec_d = generator_r2d(spec_r)
            fake_spec_r = generator_d2r(spec_d)

            fake_wav_d, fake_wav_r = wav_pad(istft_transform(fake_spec_d)), wav_pad(
                istft_transform(fake_spec_r)
            )

            # disc r and disc fake r
            real_msd_d, _ = msd_d(wav_d)

            real_mpd_d, _ = mpd_d(wav_d)

            fake_msd_d, _ = msd_d(fake_wav_d)

            fake_mpd_d, _ = mpd_d(fake_wav_d)

            # disc d and disc fake d
            real_msd_r, _ = msd_d(wav_r)

            real_mpd_r, _ = mpd_d(wav_r)

            fake_msd_r, _ = msd_d(fake_wav_r)

            fake_mpd_r, _ = mpd_d(fake_wav_r)

            # adversarial loss of discriminator

            disc_adv_loss_d = discriminator_adv_loss(
                fake_msd_d, fake_mpd_d, real_msd_d, real_mpd_d
            )
            disc_adv_loss_r = discriminator_adv_loss(
                fake_msd_r, fake_mpd_r, real_msd_r, real_mpd_r
            )

            disc_loss = disc_adv_loss_d + disc_adv_loss_r

            disc_loss.backward()
            optimizer_msd.step()
            optimizer_mpd.step()

            nn.utils.clip_grad_norm_(msd_d.parameters(), clip_value)
            nn.utils.clip_grad_norm_(msd_r.parameters(), clip_value)
            nn.utils.clip_grad_norm_(mpd_d.parameters(), clip_value)
            nn.utils.clip_grad_norm_(mpd_r.parameters(), clip_value)

            # ------------- Train Generator ------------- #
            fake_spec_d = generator_r2d(spec_r)
            cycle_spec_r = generator_d2r(fake_spec_d)

            fake_spec_r = generator_d2r(spec_d)
            cycle_spec_d = generator_r2d(fake_spec_r)

            identity_d = generator_r2d(spec_d)

            fake_wav_d, fake_wav_r = wav_pad(istft_transform(fake_spec_d)), wav_pad(
                istft_transform(fake_spec_r)
            )

            # disc r and disc fake r
            real_msd_d, real_msd_d_fmap = msd_d(wav_d)

            real_mpd_d, real_mpd_d_fmap = mpd_d(wav_d)

            fake_msd_d, fake_msd_d_fmap = msd_d(fake_wav_d)

            fake_mpd_d, fake_mpd_d_fmap = mpd_d(fake_wav_d)

            # disc d and disc fake d
            real_msd_r, real_msd_r_fmap = msd_d(wav_r)

            real_mpd_r, real_mpd_r_fmap = mpd_d(wav_r)

            fake_msd_r, fake_msd_r_fmap = msd_d(fake_wav_r)

            fake_mpd_r, fake_mpd_r_fmap = mpd_d(fake_wav_r)

            # cycle d

            cycle_loss_d = F.l1_loss(cycle_spec_d, spec_d)

            # cycle r

            cycle_loss_r = F.l1_loss(cycle_spec_r, spec_r)

            # identity loss

            identity_loss_d = F.l1_loss(identity_d, spec_d)

            # adversarial loss of generator
            generator_adv_loss_d = generator_adv_loss(
                fake_msd_d,
                fake_mpd_d,
                fake_mpd_d_fmap,
                real_mpd_d_fmap,
                fake_msd_d_fmap,
                real_msd_d_fmap,
            )
            generator_adv_loss_r = generator_adv_loss(
                fake_msd_r,
                fake_mpd_r,
                fake_mpd_r_fmap,
                real_mpd_r_fmap,
                fake_msd_r_fmap,
                real_msd_r_fmap,
            )

            gen_loss = (
                sum(generator_adv_loss_d)
                + sum(generator_adv_loss_r)
                + identity_loss_d
                + cycle_loss_r
                + cycle_loss_d
            )

            gen_loss.backward()
            optimizer_g.step()

            nn.utils.clip_grad_norm_(generator_r2d.parameters(), clip_value)
            nn.utils.clip_grad_norm_(generator_d2r.parameters(), clip_value)

            wandb.log(
                {
                    "disc_loss": disc_loss.cpu().detach(),
                    "gen_loss": gen_loss.cpu().detach(),
                    "identity_loss_d": identity_loss_d.cpu().detach(),
                    "cycle_loss_d": cycle_loss_d.cpu().detach(),
                    "cycle_loss_r": cycle_loss_d.cpu().detach(),
                    "generator_adv_loss_d": sum(generator_adv_loss_d).cpu().detach(),
                    "generator_adv_loss_r": sum(generator_adv_loss_r).cpu().detach(),
                    "disc_adv_loss_d": disc_adv_loss_d.cpu().detach(),
                    "disc_adv_loss_d": disc_adv_loss_r.cpu().detach(),
                    "clean": wandb.Audio(
                        wav_d[0].cpu().detach().numpy().T, sample_rate=16000
                    ),
                    "rev": wandb.Audio(
                        wav_r[0].cpu().detach().numpy().T, sample_rate=16000
                    ),
                    "fake_wav_d": wandb.Audio(
                        fake_wav_d[0].cpu().detach().numpy().T, sample_rate=16000
                    ),
                    "fake_wav_r": wandb.Audio(
                        fake_wav_r[0].cpu().detach().numpy().T, sample_rate=16000
                    )
                    # 'fake_spec_d': wandb.Image(fake_spec_d[0]),
                    # 'fake_spec_r': wandb.Image(fake_spec_r[0]),
                    # 'spec_d': wandb.Image(spec_d[0]),
                    # 'spec_r': wandb.Image(spec_r[0]),
                    # 'cycle_spec_r': wandb.Image(cycle_spec_r[0]),
                    # 'cycle_spec_d': wandb.Image(cycle_spec_d[0]),
                }
            )


def log_everything():
    pass


if __name__ == "__main__":
    EPOCHS = 200
    LR = 1e-3
    BS = 2
    CLIP_VALUE = 10
    LIMIT = 2

    wandb.init(project="CycleGAN_dereverberation")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # _d2r means reverberated -> dry, _r2d means dry -> reverberated
    msd_r = MSD().to(device)
    mpd_r = MPD(periods=[2, 5, 7]).to(device)

    msd_d = MSD().to(device)
    mpd_d = MPD(periods=[2, 5, 7]).to(device)

    generator_r2d = Generator().to(device)
    generator_d2r = Generator().to(device)

    optimizer_g = torch.optim.Adam(
        list(generator_r2d.parameters()) + list(generator_d2r.parameters()), lr=LR
    )

    optimizer_msd = torch.optim.Adam(
        list(msd_d.parameters()) + list(msd_r.parameters()), lr=LR / 2
    )
    optimizer_mpd = torch.optim.Adam(
        list(mpd_r.parameters()) + list(mpd_d.parameters()), lr=LR / 2
    )

    dataset = LibriSpeechDataset(data_path="./data", split="train-clean-100")
    dataset = torch.utils.data.Subset(dataset, list(range(LIMIT)))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=BS, shuffle=True, num_workers=0
    )
    print(
        f"Generator: {sum(p.numel() for p in generator_r2d.parameters())}, MSD: {sum(p.numel() for p in msd_r.parameters())}, MPD: {sum(p.numel() for p in mpd_r.parameters())}"
    )
    train(clip_value=CLIP_VALUE)
