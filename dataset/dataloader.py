from torch.utils.data import DataLoader
from dataset.video_clip_dataset import VideoClipDataset

def get_dataloader(
    labels_csv,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    pin_memory=True
):
    """
    Creates PyTorch DataLoader
    """

    dataset = VideoClipDataset(labels_csv)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    return dataloader
