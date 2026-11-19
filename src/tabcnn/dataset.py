"""Frame-level dataset for TabCNN.

PyTorch port of the original Keras ``DataGenerator`` (a ``keras.utils.Sequence``).
Where the original subclassed ``Sequence`` and produced whole batches, this is a
``torch.utils.data.Dataset`` that returns one sample at a time; batching and
shuffling are left to ``torch.utils.data.DataLoader``.

Each ID in ``list_IDs`` has the form ``<name>_<frame_idx>`` and names one frame
of one precomputed spectrogram file ``<data_path>/<spec_repr>/<name>.npz`` with:
    repr:   (frames, bins)      spectral representation
    labels: (frames, 6, 21)     per-string one-hot fret labels

For a frame, a context window of ``con_win_size`` frames centered on it is cut
(zero-padded at the clip edges) and returned as a float32 tensor shaped
(1, bins, con_win_size) -- channels-first for PyTorch Conv2d, versus the
original's channels-last (bins, con_win_size, 1).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class TabFrameDataset(Dataset):
    def __init__(
        self,
        list_IDs: list[str],
        data_path: str | Path = "../data/spec_repr/",
        spec_repr: str = "c",
        con_win_size: int = 9,
    ) -> None:
        self.list_IDs = list(list_IDs)
        self.data_dir = Path(data_path) / spec_repr
        self.spec_repr = spec_repr
        self.con_win_size = con_win_size
        self.halfwin = con_win_size // 2

    def __len__(self) -> int:
        return len(self.list_IDs)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        ID = self.list_IDs[index]
        filename = "_".join(ID.split("_")[:-1]) + ".npz"
        frame_idx = int(ID.split("_")[-1])

        loaded = np.load(self.data_dir / filename)
        # Context window centered on frame_idx, zero-padded at the clip edges.
        full_x = np.pad(loaded["repr"], [(self.halfwin, self.halfwin), (0, 0)],
                        mode="constant")
        sample_x = full_x[frame_idx:frame_idx + self.con_win_size]  # (W, bins)

        # (1, bins, con_win_size), channels-first.
        x = np.expand_dims(sample_x.T, 0).astype(np.float32)
        y = loaded["labels"][frame_idx].astype(np.float32)  # (6, 21)
        return torch.from_numpy(x), torch.from_numpy(y)


def load_ids(data_path: str | Path, id_file: str = "id.csv") -> list[str]:
    """Read the newline/CSV list of frame IDs (one column, no header)."""
    import pandas as pd

    csv_file = Path(data_path) / id_file
    return list(pd.read_csv(csv_file, header=None)[0])


def partition_ids(list_IDs: list[str], test_split: int) -> dict[str, list[str]]:
    """Split IDs into train/test by guitarist, held out on ``test_split``.

    A guitarist is the integer prefix of the ID (``<guitarist>_...``), matching
    the original 6-fold leave-one-guitarist-out protocol: 5 guitarists train and
    the held-out one is the test set (there is no separate validation split).
    """
    partition: dict[str, list[str]] = {"training": [], "test": []}
    for ID in list_IDs:
        guitarist = int(ID.split("_")[0])
        key = "test" if guitarist == test_split else "training"
        partition[key].append(ID)
    return partition
