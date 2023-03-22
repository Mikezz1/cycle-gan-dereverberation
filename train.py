import torch
import torch.nn as nn
from typing import List
import torch.nn
import torch.nn.functional as F
import argparse
from cyclegan import *
from discriminator import *
from dataset import LibriSpeechDataset, LJSpeechDataset
import torchaudio
import wandb


def get_grad_norm(self, model, norm_type=2):
    parameters = model.parameters()
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]),
        norm_type,
    )
    return total_norm.item()


def istft_transform(x):
    pad = nn.ConstantPad2d(padding=(0, 1, 0, 0), value=0).to(device)
    transform = torchaudio.transforms.InverseSpectrogram(n_fft=798, normalized=True).to(
        device
    )
    x = pad(x)
    return transform(
        torch.view_as_complex(x.transpose(1, 3).transpose(1, 2).contiguous())
    ).unsqueeze(1)


def train(clip_value, use_mpd):
    generator_r2d.train()
    generator_d2r.train()
    msd_r.train()
    mpd_d.train() if use_mpd else None

    wav_pad = nn.ConstantPad1d(padding=(0, 40), value=0)

    for epoch in range(EPOCHS):
        for batch_idx, (wav_d, spec_d, wav_r, spec_r) in enumerate(dataloader):
            wav_d = wav_d.to(device)
            spec_d = spec_d.to(device)

            wav_r = wav_r.to(device)
            spec_r = spec_r.to(device)

            optimizer_msd.zero_grad()
            optimizer_mpd.zero_grad() if use_mpd else None
            optimizer_g.zero_grad()

            # ------------- Train Discriminator ------------- #

            with torch.no_grad():

                fake_spec_d = generator_r2d(spec_r)
                fake_spec_r = generator_d2r(spec_d)

            fake_wav_d, fake_wav_r = wav_pad(istft_transform(fake_spec_d)), wav_pad(
                istft_transform(fake_spec_r)
            )

            # disc r and disc fake r
            real_msd_d, _ = msd_d(wav_d)

            real_mpd_d, _ = (
                mpd_d(wav_d)
                if use_mpd
                else (
                    float("nan"),
                    float("nan"),
                )
            )

            fake_msd_d, _ = msd_d(fake_wav_d)

            fake_mpd_d, _ = (
                mpd_d(fake_wav_d)
                if use_mpd
                else (
                    float("nan"),
                    float("nan"),
                )
            )

            # disc d and disc fake d
            real_msd_r, _ = msd_d(wav_r)

            real_mpd_r, _ = (
                mpd_d(wav_r)
                if use_mpd
                else (
                    float("nan"),
                    float("nan"),
                )
            )

            fake_msd_r, _ = msd_d(fake_wav_r)

            fake_mpd_r, _ = (
                mpd_d(fake_wav_r)
                if use_mpd
                else (
                    float("nan"),
                    float("nan"),
                )
            )

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
            optimizer_mpd.step() if USE_MPD else None

            nn.utils.clip_grad_norm_(msd_d.parameters(), clip_value)
            nn.utils.clip_grad_norm_(msd_r.parameters(), clip_value)

            nn.utils.clip_grad_norm_(
                mpd_d.parameters(), clip_value
            ) if use_mpd else None

            nn.utils.clip_grad_norm_(
                mpd_r.parameters(), clip_value
            ) if use_mpd else None

            # ------------- Train Generator ------------- #
            fake_spec_d = generator_r2d(spec_r)
            cycle_spec_r = generator_d2r(fake_spec_d)

            fake_spec_r = generator_d2r(spec_d)
            cycle_spec_d = generator_r2d(fake_spec_r)

            identity_d = generator_r2d(spec_d)

            fake_wav_d, fake_wav_r = wav_pad(istft_transform(fake_spec_d)), wav_pad(
                istft_transform(fake_spec_r)
            )

            fake_wav_d_cycle, fake_wav_r_cycle = wav_pad(
                istft_transform(cycle_spec_d)
            ), wav_pad(istft_transform(cycle_spec_r))

            # disc r and disc fake r
            real_msd_d, real_msd_d_fmap = msd_d(wav_d)

            real_mpd_d, real_mpd_d_fmap = (
                mpd_d(wav_d) if use_mpd else float("nan"),
                float("nan"),
            )

            # disc d and disc fake d
            real_msd_r, real_msd_r_fmap = msd_d(wav_r)

            real_mpd_r, real_mpd_r_fmap = (
                mpd_d(wav_r) if use_mpd else float("nan"),
                float("nan"),
            )

            # cycle disc
            fake_msd_r_cycle, fake_msd_r_fmap_cycle = msd_r(fake_wav_r_cycle)
            fake_msd_d_cycle, fake_msd_d_fmap_cycle = msd_d(fake_wav_d_cycle)

            fake_mpd_r_cycle, fake_mpd_r_fmap_cycle = mpd_r(
                fake_wav_r_cycle
            ) if use_mpd else float("nan"), float("nan")
            fake_mpd_d_cycle, fake_mpd_d_fmap_cycle = mpd_d(
                fake_wav_d_cycle
            ) if use_mpd else float("nan"), float("nan")

            generator_adv_loss_d_cycle = generator_adv_loss(
                fake_msd_d_cycle,
                fake_mpd_d_cycle,
                fake_mpd_d_fmap_cycle,
                real_mpd_d_fmap,
                fake_msd_d_fmap_cycle,
                real_msd_d_fmap,
            )
            generator_adv_loss_r_cycle = generator_adv_loss(
                fake_msd_r_cycle,
                fake_mpd_r_cycle,
                fake_mpd_r_fmap_cycle,
                real_mpd_r_fmap,
                fake_msd_r_fmap_cycle,
                real_msd_r_fmap,
            )

            # cycle d

            cycle_loss_d = 0.1 * F.l1_loss(cycle_spec_d, spec_d)

            # cycle r

            cycle_loss_r = 0.1 * F.l1_loss(cycle_spec_r, spec_r)

            # identity loss

            identity_loss_d = F.l1_loss(identity_d, spec_d)

            gen_loss = (
                identity_loss_d
                + cycle_loss_r
                + cycle_loss_d
                + sum(generator_adv_loss_d_cycle)
                + sum(generator_adv_loss_r_cycle)
            )

            gen_loss.backward()
            optimizer_g.step()

            nn.utils.clip_grad_norm_(generator_r2d.parameters(), clip_value)
            nn.utils.clip_grad_norm_(generator_d2r.parameters(), clip_value)

            # wandb.log(
            #     {
            #         "disc_loss": disc_loss.cpu().detach(),
            #         "gen_loss": gen_loss.cpu().detach(),
            #         "identity_loss_d": identity_loss_d.cpu().detach(),
            #         "cycle_loss_d": cycle_loss_d.cpu().detach(),
            #         "cycle_loss_r": cycle_loss_d.cpu().detach(),
            #         "generator_adv_loss_d": sum(generator_adv_loss_d_cycle)
            #         .cpu()
            #         .detach(),
            #         "generator_adv_loss_r": sum(generator_adv_loss_r_cycle)
            #         .cpu()
            #         .detach(),
            #         "disc_adv_loss_d": disc_adv_loss_d.cpu().detach(),
            #         "disc_adv_loss_d": disc_adv_loss_r.cpu().detach(),
            #         "clean": wandb.Audio(
            #             wav_d[0].cpu().detach().numpy().T, sample_rate=22050
            #         ),
            #         "rev": wandb.Audio(
            #             wav_r[0].cpu().detach().numpy().T, sample_rate=22050
            #         ),
            #         "fake_wav_d": wandb.Audio(
            #             fake_wav_d[0].cpu().detach().numpy().T, sample_rate=22050
            #         ),
            #         "fake_wav_r": wandb.Audio(
            #             fake_wav_r[0].cpu().detach().numpy().T, sample_rate=22050
            #         ),
            #     }
            # )


def log_everything():
    pass


if __name__ == "__main__":
    EPOCHS = 40
    LR = 3e-4
    BS = 1
    CLIP_VALUE = 100
    LIMIT = 2
    USE_MPD = False
    SAMPLE_LENGTH = 8000

    # wandb.init(project="CycleGAN_dereverberation")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # _d2r means reverberated -> dry, _r2d means dry -> reverberated
    msd_r = MSD().to(device)
    mpd_r = MPD(periods=[2, 5, 7]).to(device) if USE_MPD else None

    msd_d = MSD().to(device)
    mpd_d = MPD(periods=[2, 5, 7]).to(device) if USE_MPD else None
    generator_r2d = Generator().to(device)
    generator_d2r = Generator().to(device)

    optimizer_g = torch.optim.Adam(
        list(generator_r2d.parameters()) + list(generator_d2r.parameters()), lr=LR
    )

    optimizer_msd = torch.optim.Adam(
        list(msd_d.parameters()) + list(msd_r.parameters()), lr=LR / 2
    )
    optimizer_mpd = (
        torch.optim.Adam(list(mpd_r.parameters()) + list(mpd_d.parameters()), lr=LR / 2)
        if USE_MPD
        else None
    )

    dataset = LJSpeechDataset(
        root_dir="data/train", sample_length=SAMPLE_LENGTH
    )  # LibriSpeechDataset(data_path="./data", split="train-clean-100")
    print(len(dataset))
    # dataset = torch.utils.data.Subset(dataset, list(range(LIMIT)))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=BS, shuffle=True, num_workers=0
    )
    print(
        f"Generator: {sum(p.numel() for p in generator_r2d.parameters())}, MSD: {sum(p.numel() for p in msd_r.parameters())}, MPD: {sum(p.numel() for p in mpd_r.parameters()) if USE_MPD else 0}"
    )
    train(clip_value=CLIP_VALUE, use_mpd=USE_MPD)
