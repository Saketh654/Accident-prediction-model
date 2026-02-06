from torch.utils.data import DataLoader
from dataset.video_clip_dataset import VideoClipDataset

def get_dataloader(
    labels_csv,
    batch_size=4,
    shuffle=True,
    num_workers=4
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
        pin_memory=True
    )

    return dataloader
