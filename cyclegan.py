import torch
import torch.nn as nn
from typing import List
import torch.nn
import numpy as np
import torch.nn.functional as F


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, s):
        super(EncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=k,
            stride=s,
            padding=(
                int((k[0] - 1) / 2),
                int((k[1] - 1) / 2),
            ),
        )
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, s):
        super(DecoderBlock, self).__init__()
        self.act = nn.LeakyReLU()
        self.deconv1 = nn.ConvTranspose2d(
            in_ch,
            in_ch,
            k,
            stride=s,
            padding=(int((k[0] - 1) / 2), int((k[1] - 1) / 2)),
        )
        self.conv1 = nn.Conv2d(
            in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1
        )

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0),
            nn.Upsample(scale_factor=s),
        )

    def forward(self, x):
        resid = x
        x = self.act(self.deconv1(x))
        x = self.act(self.conv1(x))

        return x + self.proj(resid)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.act = nn.LeakyReLU()
        self.in_layer = nn.Conv2d(2, 32, kernel_size=7, padding=(7 - 1) // 2)
        self.out_layer = nn.Conv2d(32, 2, kernel_size=7, padding=(7 - 1) // 2)
        self.down = nn.ModuleList(
            [
                EncoderBlock(32, 64, k=(4, 4), s=(2, 2)),
                EncoderBlock(64, 64, k=(3, 4), s=(1, 2)),
                EncoderBlock(64, 128, k=(4, 4), s=(2, 2)),
                EncoderBlock(128, 128, k=(3, 4), s=(1, 2)),
                EncoderBlock(128, 256, k=(4, 4), s=(2, 2)),
                # EncoderBlock(256, 512, k=(3, 3), s=(1, 1)),
            ]
        )

        self.mid = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            self.act,
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            self.act,
        )

        self.up = nn.ModuleList(
            [
                # DecoderBlock(512 + 512, 256, k=(3, 3), s=(1, 1)),
                DecoderBlock(256 + 256, 128, k=(4, 4), s=(2, 2)),
                DecoderBlock(128 + 128, 128, k=(3, 4), s=(1, 2)),
                DecoderBlock(128 + 128, 64, k=(4, 4), s=(2, 2)),
                DecoderBlock(64 + 64, 64, k=(3, 4), s=(1, 2)),
                DecoderBlock(64 + 64, 32, k=(4, 4), s=(2, 2)),
            ]
        )

    def forward(self, x):
        shape = x.shape
        x = self.in_layer(x)

        skips = []
        for layer in self.down:
            x = layer(x)
            skips.append(x)

        x = self.mid(x)

        for layer in self.up:
            x = torch.cat([x, skips.pop()], dim=1)
            x = layer(x)

        x = self.out_layer(x)

        assert x.shape == shape
        return x


def generator_adv_loss(
    msd_out_fake: List[List[torch.Tensor]],
    mpd_out_fake: List[torch.Tensor],
    mpd_fmap_fake: List[torch.Tensor],
    mpd_fmap_real: List[torch.Tensor],
    msd_fmap_fake: List[torch.Tensor],
    msd_fmap_real: List[torch.Tensor],
):

    """
    Takes MSD and MPD outputs for both real and fakes and computes
    the adversarial and feature losses

    """

    mpd_loss = 0
    msd_loss = 0
    feature_loss_mpd = 0
    feature_loss_msd = 0

    if not np.isnan(mpd_out_fake):
        # mpd
        for output in mpd_out_fake:
            mpd_loss += torch.mean((output - 1) ** 2)

        for fmap_fake, fmap_real in zip(mpd_fmap_fake, mpd_fmap_real):
            for fmap_layer_f, fmap_layer_r in zip(fmap_fake, fmap_real):
                feature_loss_mpd += F.l1_loss(fmap_layer_f, fmap_layer_r)

    for fmap_fake, fmap_real in zip(msd_fmap_fake, msd_fmap_real):
        for fmap_layer_f, fmap_layer_r in zip(fmap_fake, fmap_real):
            feature_loss_msd += F.l1_loss(fmap_layer_f, fmap_layer_r)

    # msd
    for output in msd_out_fake:
        msd_loss += torch.mean((output - 1) ** 2)

    return (
        feature_loss_mpd,
        feature_loss_msd,
        msd_loss,
        mpd_loss,
    )


def discriminator_adv_loss(
    msd_out_fake: List[List[torch.Tensor]],
    mpd_out_fake: List[torch.Tensor],
    msd_out_real: List[List[torch.Tensor]],
    mpd_out_real: List[torch.Tensor],
):
    """
    Takes MSD and MPD outputs for both real and fakes and computes
    the adversarial loss for discriminator

    """

    msd_loss = 0
    mpd_loss = 0

    if not np.isnan(mpd_out_fake):
        for output_fake, output_real in zip(mpd_out_fake, mpd_out_real):
            mpd_loss = mpd_loss + torch.mean((output_real - 1) ** 2 + output_fake**2)

    for output_fake, output_real in zip(msd_out_fake, msd_out_real):
        msd_loss = msd_loss + torch.mean((output_real - 1) ** 2 + output_fake**2)

    return msd_loss + mpd_loss


def cyclic_loss_stft(real_stft, fake_stft):
    """Calculates cyclic loss

    Args:
        real_stft (torch.Tensor): real tensor of domain A
        fake_stft (torch.Tensor): G(G(A)) - inverted tensor of domain A
    """
    return F.l1_loss(real_stft, fake_stft)


def identity_loss(real_dereverb, gen_dereverg):
    """Calculates identity loss to assure that dereverberated input does not change
    if we pass it through the generator G()

    Args:
        real_dereverb (torch.Tensor): real dereverberated (dry) tensor
        gen_dereverg (torch.Tensor): output of  G_dereverb(real_dereverb)
    """
    return F.l1_loss(real_dereverb, gen_dereverg)
