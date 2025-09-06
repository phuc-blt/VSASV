import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import wavfile

from .augment import WavAugment  # cần file augment.py nếu dùng


def load_audio(filename, second=3):
    sample_rate, waveform = wavfile.read(filename)
    waveform = waveform.astype(np.float32)
    audio_length = waveform.shape[0]

    if second <= 0:
        return waveform.copy()

    length = int(sample_rate * second)
    if audio_length <= length:
        shortage = length - audio_length
        waveform = np.pad(waveform, (0, shortage), mode='wrap')
    else:
        start = random.randint(0, audio_length - length)
        waveform = waveform[start:start + length]
    return waveform.copy()


class Train_Dataset(Dataset):
    def __init__(self, train_csv_path, second=3, pairs=True, aug=False):
        self.metadata = []
        with open(train_csv_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 3:
                    continue
                spk, path, label_str = parts
                label = 0 if label_str.lower() == "bonafide" else 1
                self.metadata.append((path, label))

        self.second = second
        self.pairs = pairs
        self.aug = aug
        if self.aug:
            self.wav_aug = WavAugment()

        self.paths = [p for p, _ in self.metadata]
        self.labels = [l for _, l in self.metadata]

        print(f"[Train] Samples: {len(self.metadata)}, Bonafide: {self.labels.count(0)}, Spoof: {self.labels.count(1)}")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        path, label = self.metadata[index]
        wav1 = load_audio(path, self.second)
        if self.aug:
            wav1 = self.wav_aug(wav1)

        if not self.pairs:
            return torch.FloatTensor(wav1), torch.tensor(label).long()

        wav2 = load_audio(path, self.second)
        if self.aug:
            wav2 = self.wav_aug(wav2)

        return torch.FloatTensor(wav1), torch.FloatTensor(wav2), torch.tensor(label).long()


class Semi_Dataset(Dataset):
    def __init__(self, label_csv_path, unlabel_csv_path, second=3, pairs=True, aug=False):
        self.labeled = []
        with open(label_csv_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 3:
                    continue
                _, path, label_str = parts
                label = 0 if label_str.lower() == "bonafide" else 1
                self.labeled.append((path, label))

        self.unlabeled = []
        with open(unlabel_csv_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    self.unlabeled.append(parts[1])

        self.second = second
        self.pairs = pairs
        self.aug = aug
        if self.aug:
            self.wav_aug = WavAugment()

        print(f"[Semi] Labeled: {len(self.labeled)}, Unlabeled: {len(self.unlabeled)}")

    def __len__(self):
        return len(self.labeled)

    def __getitem__(self, index):
        path_l, label = self.labeled[index]
        wav_l = load_audio(path_l, self.second)

        idx = random.randint(0, len(self.unlabeled) - 1)
        path_u = self.unlabeled[idx]
        wav_u1 = load_audio(path_u, self.second)
        if self.aug:
            wav_u1 = self.wav_aug(wav_u1)

        if not self.pairs:
            return torch.FloatTensor(wav_l), torch.tensor(label).long(), torch.FloatTensor(wav_u1)

        wav_u2 = load_audio(path_u, self.second)
        if self.aug:
            wav_u2 = self.wav_aug(wav_u2)

        return (
            torch.FloatTensor(wav_l),
            torch.tensor(label).long(),
            torch.FloatTensor(wav_u1),
            torch.FloatTensor(wav_u2),
        )


class Evaluation_Dataset(Dataset):
    def __init__(self, paths, second=-1):
        self.paths = paths
        self.second = second
        print(f"[Eval] Total utterances: {len(self.paths)}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        wav_path = self.paths[index]
        waveform = load_audio(wav_path, self.second)
        return torch.FloatTensor(waveform), wav_path
