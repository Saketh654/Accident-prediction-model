"""
preprocessing/split_dataset.py

Run ONCE to produce train/val/test CSVs for both data pipelines.

Splitting strategy
------------------
Split at the VIDEO level, not the clip level.
All clips from a single video are assigned to exactly one split.
This prevents data leakage where the same dashcam footage appears
in both training and evaluation.

Within each class (crash / normal), videos are randomly assigned:
    70% train   15% val   15% test

Output files (in data/processed/)
----------------------------------
NPZ pipeline (3D CNN, CNN+LSTM, CNN+Transformer):
    labels_enhanced_npz_train.csv
    labels_enhanced_npz_val.csv
    labels_enhanced_npz_test.csv

Two-Stream pipeline (Two-Stream CNN / ResNet / Transformer):
    labels_two_stream_train.csv
    labels_two_stream_val.csv
    labels_two_stream_test.csv

Usage
-----
    python preprocessing/split_dataset.py
"""

import os
import pandas as pd
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NPZ_CSV        = os.path.join(ROOT, "data", "processed", "labels_enhanced_npz.csv")
TWO_STREAM_CSV = os.path.join(ROOT, "data", "processed", "labels_two_stream.csv")
OUTPUT_DIR     = os.path.join(ROOT, "data", "processed")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Split ratios ───────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
RANDOM_SEED = 42

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9


# ── Core splitting logic ───────────────────────────────────────────────────────

def split_video_ids(df, train_ratio, val_ratio, seed):
    """
    Splits videos using (video, type) composite keys so that identical
    numeric IDs in crash vs normal subsets are treated as distinct entries.

    Returns three sets of (video, type) tuples: train_keys, val_keys, test_keys.
    """
    rng = np.random.default_rng(seed)

    train_keys = set()
    val_keys   = set()
    test_keys  = set()

    for class_label in sorted(df["type"].unique()):
        videos = sorted(df[df["type"] == class_label]["video"].unique())
        keys   = [(v, class_label) for v in videos]
        keys   = np.array(keys, dtype=object)
        rng.shuffle(keys)

        n       = len(keys)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)
        n_test  = n - n_train - n_val

        train_keys.update(map(tuple, keys[:n_train]))
        val_keys.update(  map(tuple, keys[n_train : n_train + n_val]))
        test_keys.update( map(tuple, keys[n_train + n_val :]))

        print(f"  {class_label:8s}: {n:4d} videos → "
              f"train={n_train}  val={n_val}  test={n_test}")

    assert not (train_keys & val_keys),  "Overlap between train and val!"
    assert not (train_keys & test_keys), "Overlap between train and test!"
    assert not (val_keys   & test_keys), "Overlap between val and test!"

    return train_keys, val_keys, test_keys


def write_splits(df, train_keys, val_keys, test_keys, prefix, output_dir):
    """Filters df by composite key membership and writes three CSVs."""
    composite = list(zip(df["video"], df["type"]))

    df_train = df[[k in train_keys for k in composite]].reset_index(drop=True)
    df_val   = df[[k in val_keys   for k in composite]].reset_index(drop=True)
    df_test  = df[[k in test_keys  for k in composite]].reset_index(drop=True)

    n_assigned = len(df_train) + len(df_val) + len(df_test)
    assert n_assigned == len(df), \
        f"Lost {len(df) - n_assigned} rows during split assignment!"

    for split_name, split_df in [("train", df_train),
                                  ("val",   df_val),
                                  ("test",  df_test)]:
        out_path = os.path.join(output_dir, f"{prefix}_{split_name}.csv")
        split_df.to_csv(out_path, index=False)
        print(f"  Wrote {len(split_df):6d} clips  ->  {out_path}")

    return df_train, df_val, df_test


def print_label_distribution(df_train, df_val, df_test, label):
    print(f"\n  Label distribution for {label}:")
    for name, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        n_crash  = (df["label"] > 0.5).sum()
        n_normal = (df["label"] <= 0.5).sum()
        print(f"    {name:5s}: {len(df):6d} clips  |  "
              f"crash={n_crash} ({100*n_crash/len(df):.1f}%)  "
              f"normal={n_normal} ({100*n_normal/len(df):.1f}%)")


# ── NPZ pipeline ──────────────────────────────────────────────────────────────

def split_npz(input_csv, output_dir, train_ratio, val_ratio, seed):
    print(f"\n{'='*60}")
    print(f"Splitting NPZ pipeline: {input_csv}")
    print(f"{'='*60}")

    df = pd.read_csv(input_csv)
    print(f"Total clips : {len(df)} across {df['video'].nunique()} unique videos")
    print(f"Video-level split by class:")

    train_keys, val_keys, test_keys = split_video_ids(df, train_ratio, val_ratio, seed)

    df_train, df_val, df_test = write_splits(
        df, train_keys, val_keys, test_keys,
        prefix="labels_enhanced_npz",
        output_dir=output_dir
    )

    print_label_distribution(df_train, df_val, df_test, "NPZ")


# ── Two-Stream pipeline ───────────────────────────────────────────────────────

def split_two_stream(input_csv, output_dir, train_ratio, val_ratio, seed):
    print(f"\n{'='*60}")
    print(f"Splitting Two-Stream pipeline: {input_csv}")
    print(f"{'='*60}")

    df = pd.read_csv(input_csv)
    print(f"Total clips : {len(df)} across {df['video'].nunique()} unique videos")
    print(f"Video-level split by class:")

    train_keys, val_keys, test_keys = split_video_ids(df, train_ratio, val_ratio, seed)

    df_train, df_val, df_test = write_splits(
        df, train_keys, val_keys, test_keys,
        prefix="labels_two_stream",
        output_dir=output_dir
    )

    print_label_distribution(df_train, df_val, df_test, "Two-Stream")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Random seed  : {RANDOM_SEED}")
    print(f"Split ratios : train={TRAIN_RATIO}  val={VAL_RATIO}  test={TEST_RATIO}")

    split_npz(
        input_csv=NPZ_CSV,
        output_dir=OUTPUT_DIR,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        seed=RANDOM_SEED
    )

    split_two_stream(
        input_csv=TWO_STREAM_CSV,
        output_dir=OUTPUT_DIR,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        seed=RANDOM_SEED
    )

    print(f"\n{'='*60}")
    print("All splits written. Summary:")
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        if any(fname.endswith(f"_{s}.csv") for s in ["train", "val", "test"]):
            n = len(pd.read_csv(os.path.join(OUTPUT_DIR, fname)))
            print(f"  {fname:<45} {n:>6} clips")
    print(f"{'='*60}")
