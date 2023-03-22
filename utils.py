from audiomentations import RoomSimulator, Compose
import torch


def add_rir(wav, sample_rate):
    aug = Compose(
        [
            RoomSimulator(
                p=1,
                min_absorption_value=0.075,
                max_absorption_value=0.2,
                calculation_mode="absorption",
            )
        ]
    )
    return torch.Tensor(aug(wav, sample_rate=sample_rate))
