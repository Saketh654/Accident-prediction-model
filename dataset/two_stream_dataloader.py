"""
two_stream_dataloader.py
"""

from torch.utils.data import DataLoader
from dataset.two_stream_dataset import TwoStreamDataset


def get_two_stream_dataloader(
    labels_csv,
    batch_size=4,
    shuffle=True,
    num_workers=0   # keep 0 on Windows to avoid multiprocessing issues
):
    dataset = TwoStreamDataset(labels_csv)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader
