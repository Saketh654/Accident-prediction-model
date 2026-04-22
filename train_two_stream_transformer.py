"""
train_two_stream_transformer.py

Two-Stream Transformer training — optimized for RTX 4060 Laptop GPU (8 GB)
and 12th Gen Intel i7-12650H.

Hardware targets
────────────────
GPU  : NVIDIA GeForce RTX 4060 Laptop  — 8 GB GDDR6, 3072 CUDA cores,
       Ada Lovelace (compute cap. 8.9), 128 Tensor cores (4th-gen)
CPU  : Intel Core i7-12650H — 10 cores (6P+4E), 16 logical processors,
       up to 4.7 GHz boost, 24 MB L3

Optimisation decisions
──────────────────────
1.  AMP (float16 autocast)
        Ada Tensor cores run fp16 GEMM natively → ~2× throughput vs fp32.
        GradScaler prevents underflow in fp16 gradients.

2.  torch.compile (PyTorch 2.x)
        Traces the model graph once and emits fused CUDA kernels.
        Disabled by default (add --compile flag) because first-epoch
        compilation takes ~3 min; beneficial for runs ≥ 5 epochs.

3.  Gradient checkpointing
        ResNet18 backbones recompute activations on backward pass instead
        of caching them → saves ~30% VRAM at a ~15% compute cost.
        Enabled by default; disable with grad_ckpt=False for speed.

4.  Fused AdamW
        torch.optim.AdamW with fused=True merges the parameter update
        loop into a single CUDA kernel → ~10-15% optimizer step speedup.

5.  Gradient clipping (max_norm=1.0)
        Prevents gradient explosions with Transformers; essential for
        stable training without extensive LR tuning.

6.  CosineAnnealingLR
        Smoothly decays LR → better final accuracy than step decay.

7.  DataLoader: num_workers=8, persistent_workers, prefetch_factor=3
        8 workers fully utilize i7-12650H's 16 logical processors.
        Persistent workers avoid ~8s re-spawn at each epoch boundary.
        prefetch_factor=3 hides disk latency behind GPU computation.

8.  cudnn.benchmark=True
        Benchmarks convolution algorithms once for fixed input shapes
        (224×224, fixed T) and caches the fastest kernel → free speedup.

9.  non_blocking transfers
        .to(device, non_blocking=True) overlaps H→D PCIe transfer with
        CPU preprocessing of the next batch.

Recommended batch sizes
───────────────────────
    AMP on,  grad_ckpt on  → batch_size=16  (uses ~6.5 GB VRAM)
    AMP on,  grad_ckpt off → batch_size=8   (uses ~7 GB VRAM)
    AMP off, grad_ckpt on  → batch_size=8   (uses ~7.5 GB VRAM)
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from dataset.two_stream_transformer_dataloader import get_two_stream_transformer_dataloader
from models.two_stream_transformer import TwoStreamTransformer

# ─── Defaults (tuned for RTX 4060 Laptop 8GB + AMP) ──────────────────────────
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS         = 15
BATCH_SIZE     = 16       # Safe with AMP + grad checkpointing
LR             = 3e-4
LR_MIN         = 1e-6
WEIGHT_DECAY   = 1e-4
GRAD_CLIP      = 1.0
WARMUP_EPOCHS  = 2        # Linear LR warmup prevents early instability
NUM_WORKERS    = 8        # i7-12650H: 16 logical cores → 8 workers optimal

TRAIN_CSV      = "data/processed/labels_two_stream_train.csv"
VAL_CSV        = "data/processed/labels_two_stream_val.csv"
CHECKPOINT_DIR = "checkpoints"
BEST_CKPT      = os.path.join(CHECKPOINT_DIR, "two_stream_transformer_best.pth")
FINAL_CKPT     = os.path.join(CHECKPOINT_DIR, "two_stream_transformer_final.pth")
HISTORY_PATH   = "two_stream_transformer_loss_history.npy"


# ─── Argument parser ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train Two-Stream Transformer")
    p.add_argument("--epochs",      type=int,   default=EPOCHS)
    p.add_argument("--batch-size",  type=int,   default=BATCH_SIZE)
    p.add_argument("--lr",          type=float, default=LR)
    p.add_argument("--d-model",     type=int,   default=256,
                   help="Transformer hidden dim (256 fits 8GB; try 128 if OOM)")
    p.add_argument("--nhead",       type=int,   default=8)
    p.add_argument("--num-layers",  type=int,   default=4)
    p.add_argument("--dim-ff",      type=int,   default=512)
    p.add_argument("--dropout",     type=float, default=0.1)
    p.add_argument("--fusion",      type=str,   default="concat",
                   choices=["concat", "average"])
    p.add_argument("--no-grad-ckpt",  action="store_true",
                   help="Disable gradient checkpointing (faster, uses more VRAM)")
    p.add_argument("--share-transformer", action="store_true",
                   help="Share Transformer weights between streams (~10M fewer params)")
    p.add_argument("--compile",     action="store_true",
                   help="torch.compile the model (recommended for runs > 5 epochs)")
    p.add_argument("--pos-weight",  type=float, default=3.0,
                   help="BCEWithLogitsLoss positive class weight (for imbalanced data)")
    p.add_argument("--train-csv",   type=str,   default=TRAIN_CSV)
    p.add_argument("--val-csv",     type=str,   default=VAL_CSV)
    return p.parse_args()


# ─── LR scheduler with linear warmup ─────────────────────────────────────────

def build_scheduler(optimizer, warmup_epochs, total_epochs, lr_min):
    """
    Linear warmup for `warmup_epochs`, then CosineAnnealing to lr_min.
    Prevents large early updates that can permanently harm pretrained weights.
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs          # linear ramp 0 → 1
        return 1.0                                       # cosine handles the rest

    warmup   = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    cosine   = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs - warmup_epochs, eta_min=lr_min
    )
    return optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )


# ─── Train one epoch ──────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, scaler, device, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    desc = "  train" if train else "  val  "

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for rgb, flow, labels in tqdm(loader, desc=desc, leave=False):
            rgb    = rgb.to(device, non_blocking=True)
            flow   = flow.to(device, non_blocking=True)
            labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

            with autocast(device, enabled=(device == "cuda")):
                logits = model(rgb, flow)
                loss   = criterion(logits, labels)

            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item()

    return total_loss / len(loader)


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    args = parse_args()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.backends.cudnn.benchmark = True   # free speedup for fixed input shapes

    # ── DataLoaders ───────────────────────────────────────────────────────────
    loader_kw = dict(
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=3,
    )
    train_loader = get_two_stream_transformer_dataloader(
        args.train_csv, batch_size=args.batch_size, shuffle=True,  **loader_kw
    )
    val_loader = get_two_stream_transformer_dataloader(
        args.val_csv,   batch_size=args.batch_size, shuffle=False, **loader_kw
    )

    print(f"Device      : {DEVICE}")
    print(f"Train clips : {len(train_loader.dataset)}")
    print(f"Val   clips : {len(val_loader.dataset)}")
    print(f"Batch size  : {args.batch_size}  |  Epochs: {args.epochs}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = TwoStreamTransformer(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_ff=args.dim_ff,
        dropout=args.dropout,
        fusion=args.fusion,
        grad_ckpt=not args.no_grad_ckpt,
        share_transformer=args.share_transformer,
    ).to(DEVICE)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params    : {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    # torch.compile — fuses kernels, ~20% throughput gain after warmup
    if args.compile:
        print("Compiling model with torch.compile() ... (this takes ~3 min)")
        model = torch.compile(model)

    # ── Loss, optimizer, scheduler ────────────────────────────────────────────
    pos_weight = torch.tensor([args.pos_weight]).to(DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Separate LRs: lower LR for pretrained backbone, higher for new layers
    backbone_params     = []
    non_backbone_params = []
    for name, param in model.named_parameters():
        if "backbone" in name:
            backbone_params.append(param)
        else:
            non_backbone_params.append(param)

    optimizer = optim.AdamW(
        [
            {"params": backbone_params,     "lr": args.lr * 0.1},   # fine-tune gently
            {"params": non_backbone_params, "lr": args.lr},
        ],
        weight_decay=WEIGHT_DECAY,
        fused=True if DEVICE == "cuda" else False,   # fused kernel on CUDA
    )

    scheduler = build_scheduler(optimizer, WARMUP_EPOCHS, args.epochs, LR_MIN)
    scaler    = GradScaler("cuda", enabled=(DEVICE == "cuda"))

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    history       = []

    patience = 3
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        current_lr = scheduler.get_last_lr()
        print(f"\nEpoch [{epoch+1}/{args.epochs}]  "
              f"LR_bb={current_lr[0]:.2e}  LR_new={current_lr[1]:.2e}")

        train_loss = run_epoch(
            model, train_loader, criterion, optimizer, scaler, DEVICE, train=True
        )
        val_loss = run_epoch(
            model, val_loader, criterion, optimizer, scaler, DEVICE, train=False
        )

        scheduler.step()
        history.append((train_loss, val_loss))
        print(f"  train={train_loss:.4f}  val={val_loss:.4f}", end="")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0   # reset counter
            torch.save(model.state_dict(), BEST_CKPT)
            print("  -> saved best ✓")
        else:
            epochs_no_improve += 1
            print(f"  (no improvement: {epochs_no_improve}/{patience})")

            if epochs_no_improve >= patience:
                print("\nEarly stopping triggered")
                break

    # ── Save final checkpoint + loss history ──────────────────────────────────
    torch.save(model.state_dict(), FINAL_CKPT)
    np.save(HISTORY_PATH, np.array(history))

    print(f"\nTraining complete.")
    print(f"Best val loss : {best_val_loss:.4f}")
    print(f"Best checkpoint  : {BEST_CKPT}")
    print(f"Final checkpoint : {FINAL_CKPT}")
    print(f"Loss history     : {HISTORY_PATH}")
