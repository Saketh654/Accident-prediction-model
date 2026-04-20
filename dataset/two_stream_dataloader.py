from torch.utils.data import DataLoader
from dataset.two_stream_dataset import TwoStreamDataset

NUM_WORKERS = 6


def get_two_stream_dataloader(
    labels_csv,
    batch_size=4,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
):
    dataset = TwoStreamDataset(labels_csv)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    return dataloader