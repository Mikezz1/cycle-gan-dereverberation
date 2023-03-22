import os
import tarfile
import torch
from torch.utils.data import Dataset
import torchaudio
from utils import add_rir
import random

# from torchaudio.transforms import Compose


class LibriSpeechDataset(Dataset):
    def __init__(self, data_path, split):
        self.data_path = data_path
        self.split = split
        self.segment_size = 16000
        audio_files = []

        self.transform = torchaudio.transforms.Spectrogram(
            n_fft=398, normalized=True, power=None
        )

        # Collect paths to all audio files
        for root, dirs, files in os.walk(os.path.join(data_path, split)):
            for file in files:
                if file.endswith(".flac"):
                    audio_files.append(os.path.join(root, file))

        # shuffle files to randomize split
        random.shuffle(audio_files)
        self.audio_files_dry = audio_files[: len(audio_files) // 2]
        self.audio_files_reverb = audio_files[len(audio_files) // 2 :]

    def __len__(self):
        return min(len(self.audio_files_dry), len(self.audio_files_reverb))

    def crop_audio(self, audio):
        ss = self.segment_size
        if audio.size(1) >= ss:
            max_audio_start = audio.size(1) - ss
            audio_start = random.randint(0, max_audio_start)
            audio = audio[:, audio_start : audio_start + ss]
        else:
            audio = torch.nn.functional.pad(audio, (0, ss - audio.size(1)), "constant")
        return audio

    def __getitem__(self, idx):
        dry_audio_path = self.audio_files_dry[idx]
        reverb_audio_path = self.audio_files_reverb[idx]

        dry_waveform, _ = torchaudio.load(dry_audio_path, normalize=True)

        dry_waveform = self.crop_audio(dry_waveform)
        reverb_waveform, sample_rate = torchaudio.load(
            reverb_audio_path, normalize=True
        )
        reverb_waveform = reverb_waveform
        reverb_waveform = self.crop_audio(add_rir(reverb_waveform.numpy(), sample_rate))

        # Apply any transformations to the audio
        if self.transform is not None:
            # Returns shape of (smth, h, w)
            dry_spec = self.transform(dry_waveform)

            # Returns shape of (smth, h, w, 2)
            dry_spec = torch.view_as_real(dry_spec)

            dry_spec = dry_spec.squeeze(0)
            # print(dry_spec.size())

            reverb_spec = self.transform(reverb_waveform)
            reverb_spec = torch.view_as_real(reverb_spec)
            reverb_spec = reverb_spec.squeeze(0)

            # print(reverb_spec.size())

        return (
            dry_waveform,
            dry_spec.transpose(0, 2).transpose(1, 2)[:, :, :-1],
            reverb_waveform,
            reverb_spec.transpose(0, 2).transpose(1, 2)[:, :, :-1],
        )


class LJSpeechDataset(Dataset):
    def __init__(self, root_dir, sample_length):
        self.root_dir = root_dir
        self.audio_files = sorted(os.listdir(root_dir))
        self.segment_size = 16000

        random.shuffle(self.audio_files)
        self.audio_files_dry = self.audio_files[: len(self.audio_files) // 2]
        self.audio_files_reverb = self.audio_files[len(self.audio_files) // 2 :]
        self.transform = torchaudio.transforms.Spectrogram(
            n_fft=798, normalized=True, power=None
        )

    def __len__(self):
        return min(len(self.audio_files_dry), len(self.audio_files_reverb))

    def crop_audio(self, audio):
        ss = self.segment_size
        if audio.size(1) >= ss:
            max_audio_start = audio.size(1) - ss
            audio_start = random.randint(0, max_audio_start)
            audio = audio[:, audio_start : audio_start + ss]
        else:
            audio = torch.nn.functional.pad(audio, (0, ss - audio.size(1)), "constant")
        return audio

    def __getitem__(self, idx):
        dry_audio_path = self.root_dir + "/" + self.audio_files_dry[idx]
        reverb_audio_path = self.root_dir + "/" + self.audio_files_reverb[idx]

        dry_waveform, _ = torchaudio.load(dry_audio_path)

        dry_waveform = self.crop_audio(dry_waveform)
        reverb_waveform, sample_rate = torchaudio.load(reverb_audio_path)
        reverb_waveform = self.crop_audio(reverb_waveform)
        reverb_waveform = self.crop_audio(add_rir(reverb_waveform.numpy(), sample_rate))

        # Apply any transformations to the audio
        if self.transform is not None:
            # Returns shape of (smth, h, w)
            dry_spec = self.transform(dry_waveform)

            # Returns shape of (smth, h, w, 2)
            dry_spec = torch.view_as_real(dry_spec)

            dry_spec = dry_spec.squeeze(0)
            # print(dry_spec.size())

            reverb_spec = self.transform(reverb_waveform)
            reverb_spec = torch.view_as_real(reverb_spec)
            reverb_spec = reverb_spec.squeeze(0)

            # print(reverb_spec.size())

        return (
            dry_waveform,
            dry_spec.transpose(0, 2).transpose(1, 2)[:, :, :-1],
            reverb_waveform,
            reverb_spec.transpose(0, 2).transpose(1, 2)[:, :, :-1],
        )
