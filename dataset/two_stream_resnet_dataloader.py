"""
two_stream_dataloader.py

Windows-optimised DataLoader settings:
    num_workers=4       — sweet spot on Windows (beyond 4 gives diminishing returns)
    pin_memory=True     — copies batches into pinned RAM so GPU transfer is faster
    persistent_workers  — keeps worker processes alive between epochs (avoids
                          the ~10s worker spawn overhead at the start of each epoch)
    prefetch_factor=2   — each worker pre-loads 2 batches ahead of time so the
                          GPU never has to wait for the next batch
"""

from torch.utils.data import DataLoader
from dataset.two_stream_resnet_dataset import TwoStreamDataset

NUM_WORKERS = 6   # optimised for i7-12650H (10 cores, 16 logical processors)


def get_two_stream_dataloader(
    labels_csv,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    pin_memory=False,
    prefetch_factor=2,
    persistent_workers=False,
):
    dataset = TwoStreamDataset(labels_csv)  # whatever your dataset class is

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )

    return dataloader