import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from .dataset import Evaluation_Dataset, Train_Dataset, Semi_Dataset


class SPK_datamodule(LightningDataModule):
    def __init__(
        self,
        train_csv_path,
        trial_path=None,
        unlabel_csv_path=None,
        second: int = 2,
        num_workers: int = 4,
        batch_size: int = 32,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = True,
        pairs: bool = True,
        aug: bool = False,
        semi: bool = False,
    ):
        super().__init__()
        self.train_csv_path = train_csv_path
        self.trial_path = trial_path
        self.unlabel_csv_path = unlabel_csv_path

        self.second = second
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.pairs = pairs
        self.aug = aug
        self.semi = semi

        print(f"[SPK_data] second = {self.second:.2f}, batch_size = {self.batch_size}, semi = {self.semi}")

    def train_dataloader(self):
        if self.semi and self.unlabel_csv_path is not None:
            dataset = Semi_Dataset(
                self.train_csv_path, self.unlabel_csv_path,
                second=self.second, pairs=self.pairs, aug=self.aug
            )
        else:
            dataset = Train_Dataset(
                self.train_csv_path, second=self.second, pairs=self.pairs, aug=self.aug
            )

        return DataLoader(
            dataset,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last
        )

    def val_dataloader(self):
        if self.trial_path is None:
            print("[SPK_data] No trial_path provided → val_dataloader() disabled.")
            return None

        try:
            trials = np.loadtxt(self.trial_path, str)
            self.trials = trials
            eval_paths = np.unique(np.concatenate((trials[:, 1], trials[:, 2])))

            print(f"[Eval] Enroll: {len(set(trials[:, 1]))}, Test: {len(set(trials[:, 2]))}, Total eval files: {len(eval_paths)}")

            dataset = Evaluation_Dataset(eval_paths, second=-1)

            return DataLoader(
                dataset,
                num_workers=self.num_workers,
                batch_size=1,
                shuffle=False,
                pin_memory=self.pin_memory
            )
        except Exception as e:
            print(f"[SPK_data] Error loading val_dataloader from trial_path: {e}")
            return None

    def test_dataloader(self):
        return self.val_dataloader()
