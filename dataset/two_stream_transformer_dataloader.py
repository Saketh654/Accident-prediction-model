"""
two_stream_transformer_dataloader.py

DataLoader for the Two-Stream Transformer.

Reuses TwoStreamDataset from two_stream_resnet_dataset.py — the dataset
already applies ImageNet normalization and returns (rgb, flow, label)
tensors, which is exactly what TwoStreamTransformer expects.

RTX 4060 Laptop (8GB VRAM) + i7-12650H (10 cores / 16 logical) tuning:
    num_workers  = 8   — i7-12650H has 16 logical cores; 8 workers keeps
                         CPU fully utilized without starving the main process
    pin_memory   = True  — copies batches into pinned (page-locked) RAM so
                           PCIe transfer to GPU is ~2x faster
    persistent_workers — avoids ~8s worker re-spawn penalty each epoch
    prefetch_factor = 3  — each worker queues 3 batches ahead; at batch_size
                           16 this pre-stages ~48 clips, hiding disk latency
"""

from torch.utils.data import DataLoader
from dataset.two_stream_resnet_dataset import TwoStreamDataset

# Tuned for i7-12650H (10P+2E cores, 16 logical processors)
NUM_WORKERS     = 8
PREFETCH_FACTOR = 3


def get_two_stream_transformer_dataloader(
    labels_csv: str,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = NUM_WORKERS,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = PREFETCH_FACTOR,
) -> DataLoader:
    """
    Returns a DataLoader optimized for RTX 4060 Laptop + i7-12650H.

    Args:
        labels_csv          : path to labels CSV with columns
                              rgb_dir, flow_dir, start, end, label
        batch_size          : clips per batch (default 16 with AMP;
                              drop to 8 if OOM without AMP)
        shuffle             : True for train, False for val/test
        num_workers         : CPU worker processes for data loading
        pin_memory          : pin host memory for faster GPU transfer
        persistent_workers  : keep workers alive between epochs
        prefetch_factor     : batches to pre-fetch per worker

    Returns:
        torch.utils.data.DataLoader
    """
    dataset = TwoStreamDataset(labels_csv)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=True,    # Avoids last-batch size mismatch with BatchNorm / AMP
    )
