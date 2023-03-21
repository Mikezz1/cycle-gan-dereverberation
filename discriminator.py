import torch
import torch.nn as nn
from typing import List
import torch.nn
from torch.nn.utils import spectral_norm, weight_norm
import torch.nn.functional as F


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class MSD(nn.Module):
    def __init__(self):
        super(MSD, self).__init__()
        # self.downsample = F.AvgPool1d(4, stride=2, padding=2)
        self.discs = nn.ModuleList(
            [
                ScaleDiscriminator(use_spectral_norm=True),
                ScaleDiscriminator(use_spectral_norm=False),
                ScaleDiscriminator(use_spectral_norm=False),
            ]
        )
        self.pools = nn.ModuleList([nn.AvgPool1d(4, 2, 2), nn.AvgPool1d(4, 2, 2)])

    def forward(self, x):
        fmaps = []
        outputs = []
        for i, discriminator in enumerate(self.discs):
            if i > 0:
                x = self.pools[i - 1](x)
            output, fmap = discriminator(x)
            outputs.append(output)
            fmaps.append(fmap)
        return outputs, fmaps


class ScaleDiscriminator(nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(ScaleDiscriminator, self).__init__()
        self.norm = spectral_norm if use_spectral_norm else weight_norm
        self.strides = [
            1,
            2,
            2,
            4,
            4,
        ]
        self.padding = [
            7,
            20,
            20,
            20,
            20,
        ]
        self.groups = [
            1,
            4,
            16,
            16,
            16,
        ]
        self.kernel_size = [
            15,
            41,
            41,
            41,
            41,
        ]
        self.in_channels = [
            1,
            128,
            128,
            256,
            512,
        ]
        self.out_channels = [
            128,
            128,
            256,
            512,
            1024,
        ]
        self.activation = nn.LeakyReLU()

        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    self.norm(
                        nn.Conv1d(
                            in_ch, out_ch, kernel, stride=s, groups=g, padding=pad
                        )
                    ),
                    self.activation,
                )
                for in_ch, out_ch, kernel, s, g, pad in zip(
                    self.in_channels,
                    self.out_channels,
                    self.kernel_size,
                    self.strides,
                    self.groups,
                    self.padding,
                )
            ]
        )
        self.convs.apply(weights_init)
        self.out = self.norm(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmaps = []
        # print("-----")
        for layer in self.convs:
            x = layer(x)
            fmaps.append(x)
        x = self.out(x)
        # fmaps.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmaps


class MPD(nn.Module):
    """Multi-period discriminator predicts labels for wavs
    and returns both predictions and intermediate states
    """

    def __init__(
        self,
        periods,
    ):
        super(MPD, self).__init__()
        self.discriminators = nn.ModuleList(
            [SubMPD(period) for i, period in enumerate(periods)]
        )

    def forward(self, y):
        """
        Args:
            y_gt (torch.Tensor): Ground truth wav
            y (torch.Tensor): Predicted wav
        Returns:
            preds_y (List(torch.Tensor)):  discriminators' output
            preds_y_gt (List(torch.Tensor)): intermediate feature maps
        """
        preds_y = []
        features_y = []
        for layer in self.discriminators:
            prediction_y, f_y = layer(y)
            preds_y.append(prediction_y)
            features_y.append(f_y)

        return preds_y, features_y


class SubMPD(nn.Module):
    """Building block od multi-period discriminator
    Args:
        p (int): period
    """

    def __init__(self, period):
        super(SubMPD, self).__init__()
        self.period = period

        self.convs = nn.ModuleList(
            (
                weight_norm(
                    nn.Conv2d(
                        in_channels=1,
                        out_channels=32,
                        kernel_size=(5, 1),
                        stride=(3, 1),
                        padding=(2, 0),
                    )
                ),
                weight_norm(
                    nn.Conv2d(
                        in_channels=32,
                        out_channels=128,
                        kernel_size=(5, 1),
                        stride=(3, 1),
                        padding=(2, 0),
                    )
                ),
                weight_norm(
                    nn.Conv2d(
                        in_channels=128,
                        out_channels=512,
                        kernel_size=(5, 1),
                        stride=(3, 1),
                        padding=(2, 0),
                    )
                ),
                weight_norm(
                    nn.Conv2d(
                        in_channels=512,
                        out_channels=512,
                        kernel_size=(5, 1),
                        stride=(3, 1),
                        padding=(2, 0),
                    )
                ),
                # weight_norm(
                #     nn.Conv2d(
                #         in_channels=1024,
                #         out_channels=1024,
                #         kernel_size=(5, 1),
                #         stride=(3, 1),
                #         padding=(2, 0),
                #     )
                # ),
            )
        )

        self.out = weight_norm(
            nn.Conv2d(
                in_channels=512, out_channels=1, kernel_size=(3, 1), padding=(1, 0)
            )
        )

        self.convs.apply(weights_init)

    def forward(self, x):
        """May also need to return convs from out module
        Args:
            x (_type_): _description_
        Returns:
            x (torch.Tensor): discriminator output
            feature_map (List(torch.Tensor)): intermidiate features
        """
        # input size: (B, 1, T)
        features = []
        (B, _, T) = x.size()
        # Pad wavs (refactor)
        if T % self.period != 0:
            pad_len = self.period - (T % self.period)
            x = F.pad(x, (0, pad_len), "reflect")
            T = T + pad_len

        x = x.view(B, 1, int(T / self.period), self.period)
        # After reshape: (B, 1, int(T/p), p)
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x)
            features.append(x)

        x = self.out(x)
        return x, features
