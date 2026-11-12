"""
GPU-Accelerated Training Pipeline for Voice-to-Do Small Language Model.
Optimized for NVIDIA RTX Tensor Cores with mixed-precision (FP16 autocast)
and scaled to 50,000+ diverse training examples.
"""

import os
import time
import math
import torch
from torch.utils.data import DataLoader

from model import VoiceTaskSLM
from tokenizer import VoiceTaskTokenizer
from dataset import VoiceTaskDataset


def evaluate(model, dataloader, device, use_amp: bool = False):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast('cuda', enabled=use_amp):
                _, loss = model(x, y)
            active_tokens = (y != -100).sum().item()
            total_loss += loss.item() * active_tokens
            total_tokens += active_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 20.0))
    return avg_loss, perplexity


def train():
    # 1. Hardware Detection & Optimization
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"\n🚀 NVIDIA GPU ACCELERATION ENABLED:")
        print(f"   • Device: {gpu_name} ({vram_gb:.1f} GB VRAM)")
        print(f"   • Tensor Cores: Active (FP16 Mixed Precision)")
        use_amp = True
        BATCH_SIZE = 128
        NUM_TRAIN = 75000
        NUM_VAL = 4000
        EPOCHS = 6
        LEARNING_RATE = 5e-4
    else:
        device = torch.device("cpu")
        print("\n⚡ Running on CPU")
        use_amp = False
        BATCH_SIZE = 64
        NUM_TRAIN = 20000
        NUM_VAL = 2000
        EPOCHS = 5
        LEARNING_RATE = 4e-4

    WEIGHT_DECAY = 0.01
    MAX_SEQ_LEN = 128
    D_MODEL = 256
    N_LAYERS = 6
    N_HEADS = 8

    # 2. Tokenizer & Dataset
    tokenizer = VoiceTaskTokenizer()
    tokenizer.save("vocab.json")
    print(f"   • Vocabulary Size: {tokenizer.padded_vocab_size}")

    train_dataset = VoiceTaskDataset(tokenizer, num_samples=NUM_TRAIN, max_seq_len=MAX_SEQ_LEN)
    val_dataset = VoiceTaskDataset(tokenizer, num_samples=NUM_VAL, max_seq_len=MAX_SEQ_LEN)

    # Use pinned memory for zero-copy DMA transfer to GPU
    pin_mem = (device.type == "cuda")
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        pin_memory=pin_mem
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=pin_mem
    )

    # 3. Model Architecture
    model = VoiceTaskSLM(
        vocab_size=tokenizer.padded_vocab_size,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        max_seq_len=MAX_SEQ_LEN,
        dropout=0.1
    ).to(device)

    param_count = model.count_parameters()
    print(f"\n🧠 Transformer Architecture (From Scratch):")
    print(f"   • Trainable Parameters: {param_count:,} (~{param_count / 1e6:.1f}M)")
    print(f"   • Layers: {N_LAYERS} | Heads: {N_HEADS} | Embedding: {D_MODEL}")
    print(f"   • Batch Size: {BATCH_SIZE} | Training Samples: {NUM_TRAIN:,}\n")

    # 4. Optimizer, Scaler & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95)
    )
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS * len(train_loader))

    best_val_loss = float('inf')

    # 5. Training Loop
    print("=" * 68)
    print("  EPOCH   |  TRAIN LOSS  |  VAL LOSS  |  VAL PERPLEXITY  |  TIME ")
    print("=" * 68)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        step_count = 0
        start_time = time.time()

        for x, y in train_loader:
            x, y = x.to(device, non_blocking=pin_mem), y.to(device, non_blocking=pin_mem)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=use_amp):
                _, loss = model(x, y)

            scaler.scale(loss).backward()

            # Unscale before gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            step_count += 1

        avg_train_loss = epoch_loss / max(step_count, 1)
        val_loss, val_ppl = evaluate(model, val_loader, device, use_amp=use_amp)
        elapsed = time.time() - start_time

        print(f"  {epoch:02d}/{EPOCHS:02d}   |    {avg_train_loss:.4f}    |   {val_loss:.4f}   |      {val_ppl:.2f}       | {elapsed:.1f}s")

        # Save Best Model Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "epoch": epoch,
                "config": {
                    "vocab_size": tokenizer.padded_vocab_size,
                    "d_model": D_MODEL,
                    "n_layers": N_LAYERS,
                    "n_heads": N_HEADS,
                    "max_seq_len": MAX_SEQ_LEN
                }
            }
            torch.save(checkpoint, "checkpoint_best.pt")

    print("=" * 68)
    print(f"🎉 Training Complete! Best Validation Loss: {best_val_loss:.4f}")
    print(f"💾 Checkpoint saved to: checkpoint_best.pt\n")


if __name__ == "__main__":
    train()
