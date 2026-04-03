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
from dataset.two_stream_dataset import TwoStreamDataset

NUM_WORKERS = 6   # optimised for i7-12650H (10 cores, 16 logical processors)


def get_two_stream_dataloader(
    labels_csv,
    batch_size=4,
    shuffle=True,
    num_workers=NUM_WORKERS
):
    dataset = TwoStreamDataset(labels_csv)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )

    return dataloader