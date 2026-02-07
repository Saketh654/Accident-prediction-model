from dataset.dataloader import get_dataloader

LABELS_CSV = "data/processed/labels_enhanced_npz.csv"

dataloader = get_dataloader(
    labels_csv=LABELS_CSV,
    batch_size=2,
    shuffle=True,
    num_workers=0
)

for clips, labels in dataloader:
    print("Clip batch shape:", clips.shape)
    print("Labels:", labels)
    break
