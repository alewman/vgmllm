"""Training loop for VgmGPT.

Features:
- Mixed precision (bf16 on Ampere+, fp16 fallback)
- Gradient accumulation for effective large batch sizes
- Cosine LR schedule with warmup
- Periodic validation and checkpoint saving
- TensorBoard logging
- Resumable from checkpoint
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .model import VgmGPT, ModelConfig
from .dataset import load_datasets
from .dataset_v4 import load_datasets_v4
from .dataset_v6 import load_datasets_v6

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    # Data
    data_dir: str = "data/prepared"
    seq_len: int = 4096

    # Optimization
    batch_size: int = 4
    grad_accum_steps: int = 8          # effective batch = batch_size * accum
    max_steps: int = 50_000
    lr: float = 3e-4
    min_lr: float = 1e-5
    weight_decay: float = 0.1
    warmup_steps: int = 2000
    grad_clip: float = 1.0

    # Model (overridden by preset)
    model_size: str = "medium"         # small, medium, large

    # Tokenizer / vocabulary
    tokenizer: str = "v6"    # "v3" (legacy), "v4", or "v6"

    # Logging & checkpoints
    log_interval: int = 10
    val_interval: int = 250
    save_interval: int = 1000
    output_dir: str = "runs/default"

    # Hardware
    num_workers: int = 2
    compile_model: bool = False        # torch.compile for speedup (broken on Windows)
    gradient_checkpointing: bool = False  # trade compute for VRAM

    # Stability
    z_loss: float = 0.0   # PaLM-style z-loss weight (λ≈1e-4); 0.0 = off


def _get_lr(step: int, config: TrainConfig) -> float:
    """Cosine decay with linear warmup."""
    if step < config.warmup_steps:
        return config.lr * (step + 1) / config.warmup_steps
    if step >= config.max_steps:
        return config.min_lr

    progress = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_lr + (config.lr - config.min_lr) * cosine


@torch.no_grad()
def _validate(model: VgmGPT, val_loader: DataLoader, device: torch.device,
              max_batches: int = 200) -> float:
    """Run validation and return average loss (capped at max_batches)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids, labels=labels)

        total_loss += out["loss"].item()
        n_batches += 1
        if n_batches >= max_batches:
            break

    model.train()
    return total_loss / max(n_batches, 1)


def train(config: TrainConfig):
    """Main training function."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Add file handler for logs (line-buffered for real-time monitoring)
    _log_path = output_dir / "train.log"
    _log_stream = open(_log_path, "a", encoding="utf-8", buffering=1)
    fh = logging.StreamHandler(_log_stream)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s",
                                       datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(fh)

    # Save config
    (output_dir / "train_config.json").write_text(
        json.dumps(config.__dict__, indent=2), encoding="utf-8"
    )

    # Device setup
    assert torch.cuda.is_available(), "CUDA required for training"
    device = torch.device("cuda")

    # Performance: TF32 for Ampere+ and cuDNN autotuner
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    log.info("Device: %s (%s)", device, torch.cuda.get_device_name())
    log.info("VRAM: %.1f GB", torch.cuda.get_device_properties(0).total_memory / 1e9)

    # Determine dtype
    if torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
        log.info("Using bfloat16 mixed precision")
    else:
        dtype = torch.float16
        log.info("Using float16 mixed precision")

    # Load data
    if config.tokenizer == "v6":
        train_loader, val_loader, _pack_loader, meta = load_datasets_v6(
            data_dir=config.data_dir,
            seq_len=config.seq_len,
            batch_size=config.batch_size,
        )
    elif config.tokenizer == "v4":
        train_loader, val_loader, meta = load_datasets_v4(
            data_dir=config.data_dir,
            seq_len=config.seq_len,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )
    else:
        train_loader, val_loader, meta = load_datasets(
            data_dir=config.data_dir,
            seq_len=config.seq_len,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )
    vocab_size = meta["vocab_size"]

    log.info("Train: %d batches, Val: %d batches",
             len(train_loader), len(val_loader))

    # Model
    from .model import config_small, config_medium, config_large

    model_configs = {
        "small": config_small,
        "medium": config_medium,
        "large": config_large,
    }
    model_cfg = model_configs[config.model_size](
        vocab_size=vocab_size, seq_len=config.seq_len
    )
    if config.gradient_checkpointing:
        model_cfg.gradient_checkpointing = True

    # Save model config
    (output_dir / "model_config.json").write_text(
        json.dumps(model_cfg.__dict__, indent=2), encoding="utf-8"
    )

    model = VgmGPT(model_cfg).to(device)

    if config.compile_model:
        try:
            model = torch.compile(model)
            log.info("Model compiled with torch.compile()")
        except Exception as e:
            log.warning("torch.compile failed, continuing without: %s", e)

    # Optimizer (AdamW with weight decay only on matmul weights)
    decay_params = []
    nodecay_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.dim() >= 2:
                decay_params.append(param)
            else:
                nodecay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=config.lr,
        betas=(0.9, 0.95),
        fused=True,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))

    # TensorBoard
    writer = SummaryWriter(log_dir=str(output_dir / "tb"))

    # Check for existing checkpoint to resume
    start_step = 0
    ckpt_path = output_dir / "latest.pt"
    if ckpt_path.exists():
        log.info("Resuming from %s", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt["model"]
        # Normalise: if checkpoint was saved from a compiled model, strip the
        # _orig_mod. prefix so we can always load into the plain (uncompiled) model.
        if any(k.startswith("_orig_mod.") for k in state_dict):
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            log.error(
                "Checkpoint key mismatch — model config and checkpoint don't agree: %s", e
            )
            raise
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_step = ckpt["step"] + 1
        log.info("Resumed at step %d", start_step)

    # Training loop
    model.train()
    train_iter = iter(train_loader)
    running_loss = 0.0
    step_time_start = time.time()
    loss_ema = None  # exponential moving average for spike detection

    effective_batch = config.batch_size * config.grad_accum_steps
    log.info(
        "Training: %d steps, batch=%d×%d=%d, lr=%.1e, seq_len=%d",
        config.max_steps, config.batch_size, config.grad_accum_steps,
        effective_batch, config.lr, config.seq_len,
    )

    for step in range(start_step, config.max_steps):
        # Update learning rate
        lr = _get_lr(step, config)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Gradient accumulation
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        spike_detected = False

        for micro_step in range(config.grad_accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=dtype):
                out = model(input_ids, labels=labels)
                loss = out["loss"] / config.grad_accum_steps

            micro_loss = loss.item() * config.grad_accum_steps  # unnormalized
            # Use ln(vocab_size)+2 as the floor for the first steps, before loss_ema
            # is seeded.  Without this, any vocab larger than ~e^6 (≈3000 tokens)
            # would trip the 8.0 hard floor on every early micro-step, preventing
            # the EMA from ever being populated and blocking all learning.
            init_floor = math.log(vocab_size) + 2.0
            spike_threshold = max(init_floor, loss_ema * 5.0) if loss_ema is not None else init_floor
            if micro_loss > spike_threshold:
                spike_detected = True
                log.warning(
                    "step=%d micro_step=%d  LOSS SPIKE %.2f > %.2f (5x EMA) — skipping step",
                    step + 1, micro_step, micro_loss, spike_threshold,
                )
                break

            scaler.scale(loss).backward()
            accum_loss += loss.item()

        # Optional z-loss: penalises unnormalized logit scale (PaLM, λ≈1e-4).
        # Keeps the softmax partition function from drifting, which is one cause
        # of training instability in large vocabularies.
        if config.z_loss > 0.0 and not spike_detected:
            with torch.amp.autocast("cuda", dtype=dtype):
                # Re-compute logits for last micro-batch's input (cheap: no backward yet)
                with torch.no_grad():
                    last_logits = model(input_ids)["logits"]
                z = torch.logsumexp(last_logits.float(), dim=-1).pow(2).mean()
            (config.z_loss * z).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        # clip_grad_norm_ returns the *pre-clipping* total gradient norm.
        # If that value is non-finite (NaN/Inf from bf16 overflow) or
        # astronomically large (> 100), something is wrong beyond normal
        # training variance and we skip the optimizer step entirely.
        grad_norm_ok = grad_norm.isfinite() and grad_norm < 100.0

        if spike_detected or not grad_norm_ok:
            if not grad_norm_ok:
                log.warning(
                    "step=%d  GRAD NORM %.2f > 100 after clipping — skipping optimizer step",
                    step + 1, grad_norm,
                )
            # Discard accumulated gradients; do not update weights
            optimizer.zero_grad(set_to_none=True)
            scaler.update()  # keep scaler state consistent
            running_loss += accum_loss  # log what we computed before skip
        else:
            scaler.step(optimizer)
            scaler.update()
            # Update loss EMA only on clean steps
            if accum_loss > 0:
                loss_ema = accum_loss if loss_ema is None else 0.98 * loss_ema + 0.02 * accum_loss

        running_loss += accum_loss

        # Logging
        if (step + 1) % config.log_interval == 0:
            elapsed = time.time() - step_time_start
            ms_per_step = elapsed / config.log_interval * 1000
            avg_loss = running_loss / config.log_interval
            tokens_per_sec = (
                config.seq_len * effective_batch * config.log_interval / elapsed
            )

            log.info(
                "step=%d  loss=%.4f  lr=%.2e  grad_norm=%.2f  "
                "%.0fms/step  %.0f tok/s",
                step + 1, avg_loss, lr, grad_norm.item(),
                ms_per_step, tokens_per_sec,
            )

            writer.add_scalar("train/loss", avg_loss, step + 1)
            writer.add_scalar("train/lr", lr, step + 1)
            writer.add_scalar("train/grad_norm", grad_norm.item(), step + 1)
            writer.add_scalar("train/tokens_per_sec", tokens_per_sec, step + 1)
            writer.flush()

            running_loss = 0.0
            step_time_start = time.time()

        # Validation
        if (step + 1) % config.val_interval == 0:
            val_loss = _validate(model, val_loader, device)
            log.info("step=%d  val_loss=%.4f  val_ppl=%.1f",
                     step + 1, val_loss, math.exp(min(val_loss, 20)))
            writer.add_scalar("val/loss", val_loss, step + 1)
            writer.add_scalar("val/perplexity", math.exp(min(val_loss, 20)), step + 1)
            model.train()

        # Checkpoint
        if (step + 1) % config.save_interval == 0:
            _save_checkpoint(model, optimizer, scaler, step, output_dir)

    # Final checkpoint
    _save_checkpoint(model, optimizer, scaler, config.max_steps - 1, output_dir)

    # Final validation
    val_loss = _validate(model, val_loader, device)
    log.info("Final val_loss=%.4f  val_ppl=%.1f",
             val_loss, math.exp(min(val_loss, 20)))

    writer.close()
    log.info("Training complete. Output: %s", output_dir)


def _save_checkpoint(model, optimizer, scaler, step, output_dir):
    """Save a training checkpoint."""
    ckpt = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
    }
    path = Path(output_dir) / "latest.pt"
    torch.save(ckpt, path)
    # Also save a numbered copy
    numbered = Path(output_dir) / f"step_{step + 1:06d}.pt"
    torch.save(ckpt, numbered)
    log.info("Checkpoint saved: step %d → %s", step + 1, path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Train VgmGPT")
    parser.add_argument("--data-dir", default="data/prepared")
    parser.add_argument("--output-dir", default="runs/default")
    parser.add_argument("--model-size", choices=["small", "medium", "large"],
                        default="medium")
    parser.add_argument("--seq-len", type=int, default=16384)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--val-interval", type=int, default=250)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--compile", action="store_true",
                        help="Enable torch.compile (not supported on Windows)")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Enable gradient checkpointing to reduce VRAM")
    parser.add_argument("--tokenizer", choices=["v3", "v4", "v6"], default="v6",
                        help="Which tokenizer/vocab the data was prepared with")
    parser.add_argument("--z-loss", type=float, default=0.0,
                        help="PaLM z-loss weight (e.g. 1e-4); 0 = off")

    args = parser.parse_args()

    config = TrainConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_size=args.model_size,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        max_steps=args.max_steps,
        lr=args.lr,
        warmup_steps=args.warmup,
        val_interval=args.val_interval,
        save_interval=args.save_interval,
        compile_model=args.compile,
        gradient_checkpointing=args.gradient_checkpointing,
        tokenizer=args.tokenizer,
        z_loss=args.z_loss,
    )

    train(config)


if __name__ == "__main__":
    main()
