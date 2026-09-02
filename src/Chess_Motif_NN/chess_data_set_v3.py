import numpy as np
import torch
from torch.utils.data import Dataset


class ChessDataset(Dataset):
    def __init__(self,input_path="data/inputs_packed.npy", target_path="data/targets_packed.npy"):
        self.input_path = input_path
        self.target_path = target_path
        self.inputs = None
        self.targets = None

    def _load(self):
        if self.inputs is None:
            self.inputs = np.load(
                self.input_path,
                mmap_mode="r"
            )
            self.targets = np.load(
                self.target_path,
                mmap_mode="r"
            )

    def __len__(self):
        return 2568096 #Gotta test for pickling

    def __getitem__(self, idx):
        self._load()

        x = torch.tensor(
            self.inputs[idx],
            dtype=torch.float32
        )

        x[18] /= 100.0

        y = torch.tensor(
            self.targets[idx],
            dtype=torch.float32
        )

        return x, y

    def get_targets(self):
        self._load()
        return torch.from_numpy(self.targets)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["inputs"] = None
        state["targets"] = None
        return state