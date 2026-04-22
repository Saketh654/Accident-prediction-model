"""
two_stream_resnet_dataloader.py

Optimized for: Ryzen 5 5600H (6C/12T) + RTX 3050 4 GB VRAM + 16 GB RAM

Worker tuning rationale:
    - 5600H has 6 physical / 12 logical cores
    - PyTorch main process + GPU transfer threads consume ~2 logical cores
    - NUM_WORKERS = 4 leaves 6 logical cores free and avoids RAM pressure
      (each worker holds its own copy of the dataset index + frame cache)
    - prefetch_factor = 2 keeps the GPU fed without pre-loading too many
      (B, C, T, H, W) tensors into pinned RAM simultaneously
"""

from torch.utils.data import DataLoader
from dataset.two_stream_resnet_dataset import TwoStreamDataset

# ── Tuned for Ryzen 5 5600H ────────────────────────────────────────────────────
# 4 workers is the sweet spot: enough parallelism, low memory overhead
NUM_WORKERS     = 4
PREFETCH_FACTOR = 2   # 2 batches queued per worker — safe for 16 GB RAM


def get_two_stream_dataloader(
    labels_csv: str,
    batch_size: int  = 8,
    shuffle: bool    = True,
    num_workers: int = NUM_WORKERS,
    pin_memory: bool = True,          # default True — copies to pinned RAM for faster GPU DMA
    prefetch_factor: int = PREFETCH_FACTOR,
    persistent_workers: bool = True,  # default True — avoids respawn overhead each epoch
) -> DataLoader:

    dataset = TwoStreamDataset(labels_csv)

    return DataLoader(
        dataset,
        batch_size          = batch_size,
        shuffle             = shuffle,
        num_workers         = num_workers,
        pin_memory          = pin_memory,
        prefetch_factor     = prefetch_factor if num_workers > 0 else None,
        persistent_workers  = persistent_workers if num_workers > 0 else False,
        # drop_last=True avoids a smaller final batch that can cause
        # uneven gradient norms with BCEWithLogitsLoss + pos_weight
        drop_last           = True,
    )