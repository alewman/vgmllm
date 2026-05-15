"""GPT-style decoder-only transformer for VGM sequence generation.

Architecture choices for Genesis music on RTX 3090 (24 GB VRAM):
- ~50-80M parameters depending on config
- RoPE positional encoding for length generalization
- Pre-norm (RMSNorm) for training stability
- SwiGLU feed-forward for better parameter efficiency
- Flash Attention 2 when available, else standard SDPA
- bf16/fp16 mixed precision training
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Hyperparameters for the VGM transformer."""
    vocab_size: int = 4096
    seq_len: int = 4096
    n_layers: int = 16
    n_heads: int = 12
    d_model: int = 768
    d_ff: int | None = None     # defaults to 4 * d_model * 2/3 (SwiGLU)
    dropout: float = 0.1
    rope_theta: float = 10000.0
    tie_embeddings: bool = True  # tie token embed to output projection
    gradient_checkpointing: bool = False  # trade compute for VRAM savings

    def __post_init__(self):
        if self.d_ff is None:
            # SwiGLU uses 2/3 * 4d hidden, rounded to nearest 64
            raw = int(4 * self.d_model * 2 / 3)
            self.d_ff = ((raw + 63) // 64) * 64

    @property
    def n_params(self) -> int:
        """Estimate total parameters (approximate)."""
        embed = self.vocab_size * self.d_model
        # Each layer: attention (Q,K,V,O) + FF (gate, up, down) + 2 norms
        attn = 4 * self.d_model * self.d_model
        ff = 3 * self.d_model * self.d_ff  # gate, up, down
        norm = 2 * self.d_model
        per_layer = attn + ff + norm
        total = embed + self.n_layers * per_layer
        if not self.tie_embeddings:
            total += embed  # separate output projection
        return total


# ---------------------------------------------------------------------------
# RoPE (Rotary Positional Embedding)
# ---------------------------------------------------------------------------

def _precompute_rope_freqs(dim: int, max_len: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute complex exponentials for RoPE.

    Returns: (max_len, dim//2) complex64 tensor.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_len, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(angles), angles)  # e^(i*theta)


def _apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding to Q or K.

    Args:
        x: (batch, n_heads, seq_len, head_dim)
        freqs: (seq_len, head_dim//2) complex
    """
    # Reshape to complex pairs
    x_complex = torch.view_as_complex(
        x.float().reshape(*x.shape[:-1], -1, 2)
    )
    # Apply rotation
    freqs = freqs.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim//2)
    x_rotated = x_complex * freqs
    # Back to real
    return torch.view_as_real(x_rotated).reshape(*x.shape).type_as(x)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).type_as(x) * self.weight


# ---------------------------------------------------------------------------
# SwiGLU Feed-Forward
# ---------------------------------------------------------------------------

class SwiGLUFF(nn.Module):
    """SwiGLU feed-forward block: gate * silu(x) then project down."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


# ---------------------------------------------------------------------------
# KV Cache (pre-allocated for efficient generation)
# ---------------------------------------------------------------------------

class KVCache:
    """Pre-allocated KV cache that avoids per-step tensor allocation."""

    def __init__(
        self,
        batch_size: int,
        n_heads: int,
        max_len: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        self.k = torch.zeros(batch_size, n_heads, max_len, head_dim,
                             device=device, dtype=dtype)
        self.v = torch.zeros(batch_size, n_heads, max_len, head_dim,
                             device=device, dtype=dtype)
        self.seq_len = 0

    def update(
        self, k_new: torch.Tensor, v_new: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write new K, V into buffer and return valid slices."""
        T = k_new.shape[2]
        end = self.seq_len + T
        self.k[:, :, self.seq_len:end] = k_new
        self.v[:, :, self.seq_len:end] = v_new
        self.seq_len = end
        return self.k[:, :, :end], self.v[:, :, :end]


# ---------------------------------------------------------------------------
# Multi-Head Self-Attention with RoPE
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    """Multi-head self-attention with RoPE, causal mask, and KV cache."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope_freqs: torch.Tensor,
        kv_cache: KVCache | None = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, KVCache | None]:
        B, T, C = x.shape

        # Project Q, K, V
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, T, n_heads, head_dim)
        q = q.transpose(1, 2)  # (B, n_heads, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply RoPE at correct positions
        q = _apply_rope(q, rope_freqs[start_pos:start_pos + T])
        k = _apply_rope(k, rope_freqs[start_pos:start_pos + T])

        # Update pre-allocated KV cache (zero-copy write)
        if kv_cache is not None:
            k, v = kv_cache.update(k, v)

        # Causal mask needed when processing multiple tokens (training or prefill).
        # Single-token steps (T=1) attend to all cached positions — no mask needed.
        is_causal = T > 1
        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=is_causal,
            dropout_p=self.dropout.p if self.training else 0.0,
        )

        # Reshape and project
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.dropout(self.out(out)), kv_cache


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Pre-norm transformer block with attention + SwiGLU FF."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = Attention(config)
        self.ff_norm = RMSNorm(config.d_model)
        self.ff = SwiGLUFF(config.d_model, config.d_ff, config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope_freqs: torch.Tensor,
        kv_cache: KVCache | None = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, KVCache | None]:
        attn_out, cache = self.attn(
            self.attn_norm(x), rope_freqs, kv_cache, start_pos
        )
        x = x + attn_out
        x = x + self.ff(self.ff_norm(x))
        return x, cache


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------

class VgmGPT(nn.Module):
    """Decoder-only transformer for VGM music generation."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.norm = RMSNorm(config.d_model)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        # Precompute RoPE frequencies — extend beyond seq_len for generation
        max_positions = max(config.seq_len, 65536)
        rope_freqs = _precompute_rope_freqs(
            config.d_model // config.n_heads,
            max_positions,
            config.rope_theta,
        )
        self.register_buffer("rope_freqs", rope_freqs, persistent=False)

        # Initialize weights
        self._init_weights()

        n_params = sum(p.numel() for p in self.parameters())
        log.info("VgmGPT: %d layers, %d heads, d=%d → %.1fM parameters",
                 config.n_layers, config.n_heads, config.d_model, n_params / 1e6)

    def _init_weights(self):
        """Initialize weights using GPT-2 scaled residual init.

        All linear layers get std=0.02.  The residual output projections
        (Attention.out and SwiGLUFF.down) are additionally scaled by
        1/sqrt(2 * n_layers) so that the residual stream variance stays
        ~constant with depth at init.  Without this, deep networks trained
        from scratch accumulate large activations as lr ramps up, causing
        the gradient explosion pattern we observed experimentally.

        Reference: GPT-2 paper §2 / Radford et al. 2019.
        """
        residual_scale = (2 * self.config.n_layers) ** -0.5
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        # Scale down the two residual output projections per block
        for block in self.layers:
            nn.init.normal_(block.attn.out.weight, mean=0.0,
                            std=0.02 * residual_scale)
            nn.init.normal_(block.ff.down.weight, mean=0.0,
                            std=0.02 * residual_scale)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        kv_caches: list[KVCache] | None = None,
        start_pos: int = 0,
        use_cache: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            input_ids: (batch, seq_len) token IDs
            labels: (batch, seq_len) target IDs for loss (-100 = ignore)
            kv_caches: Per-layer KV caches for generation.
            start_pos: Position offset for RoPE (used with KV cache).
            use_cache: If True, return updated KV caches in result.

        Returns:
            Dict with 'logits', optionally 'loss' and 'kv_caches'.
        """
        B, T = input_ids.shape
        if kv_caches is None:
            assert T <= self.config.seq_len, f"Sequence {T} > max {self.config.seq_len}"

        x = self.drop(self.tok_emb(input_ids))

        new_caches = []
        for i, layer in enumerate(self.layers):
            cache_i = kv_caches[i] if kv_caches is not None else None
            if (
                self.config.gradient_checkpointing
                and self.training
                and kv_caches is None
            ):
                x, new_cache = torch_checkpoint(
                    layer, x, self.rope_freqs, cache_i, start_pos,
                    use_reentrant=False,
                )
            else:
                x, new_cache = layer(x, self.rope_freqs, cache_i, start_pos)
            new_caches.append(new_cache)

        x = self.norm(x)
        logits = self.lm_head(x)

        result = {"logits": logits}

        if use_cache:
            result["kv_caches"] = new_caches

        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
            result["loss"] = loss

        return result

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int = 2048,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 0.95,
        eos_token: int = 2,
        repetition_penalty: float = 1.0,
        repetition_window: int = 64,
    ) -> torch.Tensor:
        """Autoregressive generation with KV cache and top-k/top-p sampling.

        Uses KV cache for O(1) per-step computation instead of
        reprocessing the full context each step.

        Args:
            prompt: (1, prompt_len) starting tokens (should start with BOS)
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (lower = more deterministic).
            top_k: Keep only top-k logits before sampling.
            top_p: Nucleus sampling threshold.
            eos_token: Stop when this token is generated.
            repetition_penalty: Penalize recently used tokens. 1.0 = off,
                >1.0 = discourage repeats (1.2-1.5 recommended).
            repetition_window: How many recent tokens to consider for penalty.

        Returns:
            (1, total_len) generated token sequence including prompt.
        """
        self.eval()
        device = prompt.device
        prompt_len = prompt.shape[1]
        max_seq = prompt_len + max_new_tokens

        # Pre-allocate output token buffer
        tokens = torch.zeros(1, max_seq, dtype=torch.long, device=device)
        tokens[:, :prompt_len] = prompt
        gen_len = prompt_len

        # Pre-allocate KV caches (one per layer)
        kv_caches = [
            KVCache(
                1, self.config.n_heads, max_seq,
                self.config.d_model // self.config.n_heads,
                device,
            )
            for _ in range(self.config.n_layers)
        ]

        # Prefill: process entire prompt
        out = self.forward(
            prompt, kv_caches=kv_caches, start_pos=0, use_cache=True,
        )
        logits = out["logits"][:, -1, :]
        cur_pos = prompt_len

        for _ in range(max_new_tokens):
            # Apply repetition penalty before temperature
            if repetition_penalty != 1.0 and gen_len > 0:
                window_start = max(0, gen_len - repetition_window)
                recent = tokens[0, window_start:gen_len].unique()
                penalty_logits = logits[0, recent]
                # Divide positive logits, multiply negative logits
                logits[0, recent] = torch.where(
                    penalty_logits > 0,
                    penalty_logits / repetition_penalty,
                    penalty_logits * repetition_penalty,
                )

            scaled = logits / temperature

            # Top-k filtering
            if top_k > 0:
                topk_vals, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
                scaled[scaled < topk_vals[:, -1:]] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                remove_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[remove_mask] = float("-inf")
                scaled = sorted_logits.scatter(1, sorted_indices, sorted_logits)

            probs = F.softmax(scaled, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            tokens[:, gen_len] = next_token[:, 0]
            gen_len += 1

            if next_token.item() == eos_token:
                break

            # Forward only the new token using pre-allocated KV cache
            out = self.forward(
                next_token, kv_caches=kv_caches,
                start_pos=cur_pos, use_cache=True,
            )
            logits = out["logits"][:, -1, :]
            cur_pos += 1

        return tokens[:, :gen_len]


# ---------------------------------------------------------------------------
# Preset configs
# ---------------------------------------------------------------------------

def config_small(vocab_size: int = 4096, seq_len: int = 4096) -> ModelConfig:
    """Small model (~25M params) for quick experiments."""
    return ModelConfig(
        vocab_size=vocab_size, seq_len=seq_len,
        n_layers=8, n_heads=8, d_model=512, dropout=0.1,
    )


def config_medium(vocab_size: int = 4096, seq_len: int = 4096) -> ModelConfig:
    """Medium model (~55M params) — good balance for RTX 3090."""
    return ModelConfig(
        vocab_size=vocab_size, seq_len=seq_len,
        n_layers=16, n_heads=12, d_model=768, dropout=0.1,
    )


def config_large(vocab_size: int = 4096, seq_len: int = 8192) -> ModelConfig:
    """Large model (~85M params) — pushes RTX 3090 VRAM limits."""
    return ModelConfig(
        vocab_size=vocab_size, seq_len=seq_len,
        n_layers=20, n_heads=16, d_model=1024, dropout=0.1,
    )


import logging
log = logging.getLogger(__name__)
