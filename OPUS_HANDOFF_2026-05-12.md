# VgmGPT Training Handoff — May 12, 2026
## For Claude Opus 4.7 Review

Hi Opus. This is a handoff from a long Sonnet 4.6 session. We've been fighting gradient
explosion across multiple training runs (v5a, v5b, v5c). We believe we've just applied the
correct fix (scaled residual init) and launched v5d. I'd like your opinion on whether the
diagnosis and fix are correct, and whether there are any other architectural or training
concerns we may have missed.

---

## The Project

**VgmGPT**: A decoder-only transformer for generating Sega Genesis (YM2612 + PSG + DAC) music.
The tokenizer represents FM synth events, PSG events, drum hits, bar markers, and composer
identity tokens. Goal: train a model that generates plausible 8-bit/16-bit era music.

- **Model**: 253.5M params, 20 layers, 16 heads, d_model=1024, SwiGLU FFN (d_ff=4096), RoPE
  (θ=10000), weight tying (tok_emb = lm_head), bfloat16 mixed precision
- **Vocab**: 456 tokens (BOS, EOS, NOTE_ON, NOTE_OFF, PITCH_BASE, VEL_BASE, DAC_HIT×8,
  PSG_NOISE, COMPOSER_BASE×...)
- **Dataset**: 2,104M train tokens, 110M val tokens (10,393 VGM files, int16 memmap)
- **Hardware**: RTX 3090 24GB, Windows, Python 3.12, PyTorch (no Triton — compile fails)
- **Sequence length**: 4096 tokens, batch=1×8 grad accum = effective batch size 8

---

## Problem History

### 1. DAC Token Dominance (FIXED)
The original `ym2612.py` had a bug: `_last_dac_pcm_offset` was only updated *inside* the
`if is_new_onset` block. Because each sequential PCM byte (offset N+1) was never equal to
the stored offset N, every byte triggered a new DAC onset token. Result: 91.73% of all
tokens were DAC hits.

**Fix applied**:
- Sequential advance check: `is_sequential = (event.pcm_offset == self._last_dac_pcm_offset + 1)`
- Both `_last_dac_pcm_offset` and `_last_dac_sample_end` now always updated (not gated)
- Added `_MAX_DAC_FRACTION = 0.50` filter in `dataset_v4.py` to drop sample-heavy files
- Dataset rebuilt: DAC now 4.08% of tokens, NOTE_ON=15.26%

### 2. Windows TDR Crash (FIXED)
GPU driver timeout (default 2s TDR) was killing long SDPA kernels. Fixed: `TdrDelay=60`
registry key set in `HKLM:\SYSTEM\CurrentControlSet\Control\GraphicsDrivers`.

### 3. Gradient Explosion — The Core Problem
**Runs v5a, v5b, v5c all showed the same failure pattern**:
- Steps 1–~500: loss drops quickly, grad norms stable and low (1–5)
- Steps ~500–1500: loss continues improving, norms remain acceptable
- Steps ~1300–1800: as lr warmup ramps (warmup was 4000 steps), norms suddenly
  climb into the 30–200+ range
- Once norms exceed ~100, the grad norm guard (added to train.py) skips optimizer steps,
  but the model is already corrupted — loss reverses and recovery is impossible

**Previous lr attempts**: v5a used lr=1e-4 (exploded faster), v5b used lr=3e-5 (same
pattern, slower), v5c used lr=3e-5 with additional guards.

**Root cause identified (Sonnet's diagnosis)**:
`model.py._init_weights` applied `std=0.02` uniformly to ALL `nn.Linear` modules.
It did NOT scale the residual output projections. In a 20-layer model, the residual stream
variance grows proportionally to the number of layers at initialization, causing large
activation norms that destabilize training as the learning rate ramps up during warmup.

---

## The Fix Applied (v5d)

**GPT-2 scaled residual initialization** (Radford et al. 2019, §2):

```python
def _init_weights(self):
    residual_scale = (2 * self.config.n_layers) ** -0.5  # = 1/sqrt(40) ≈ 0.158

    for module in self.modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # Scale residual output projections only
    for block in self.layers:
        nn.init.normal_(block.attn.out.weight, mean=0.0, std=0.02 * residual_scale)
        nn.init.normal_(block.ff.down.weight,  mean=0.0, std=0.02 * residual_scale)
```

The two residual projections (`Attention.out` and `SwiGLUFF.down`) now have `std=0.00316`
instead of `0.02`. Non-residual projections (q, k, v, gate, up) remain at `std=0.02`.

**Verification**: `check_init.py` at project root confirms:
```
out.weight  std = 0.00316  (expect 0.00316) ✓
down.weight std = 0.00316  (expect 0.00316) ✓
qkv.weight  std = 0.01999  (expect ~0.02000) ✓
```

**v5d training has started** from scratch (no checkpoint loaded — prior checkpoints used
broken init). Early results as of ~160 steps:

| step | loss   | lr       | grad_norm |
|------|--------|----------|-----------|
| 10   | 5.8443 | 5.00e-07 | 76.62     |
| 50   | 3.4741 | 2.50e-06 | 21.54     |
| 100  | 2.2747 | 5.00e-06 | 15.99     |
| 150  | 2.0470 | 7.50e-06 | 8.16      |
| 160  | 2.0119 | 8.00e-06 | 10.63     |

Norms are **declining** with loss. This is the opposite of all prior runs (where norms
were flat/small early and then spiked as lr ramped). So far, so good.

The critical danger zone is steps ~500–2000 (lr will be 2.5e-5 to 1e-4). That's where
all previous runs exploded. We need to monitor there.

**v5d launch command**:
```
python -m genesis_music.train \
  --model-size large --data-dir data/prepared_v4 --output-dir runs/v5d \
  --lr 1e-4 --warmup 2000 --max-steps 100000 \
  --save-interval 500 --batch-size 1 --grad-accum 8 \
  --gradient-checkpointing --tokenizer v4
```

Log: `logs/train_v5d.log`

---

## Key Files

| File | Purpose | Status |
|------|---------|--------|
| `src/genesis_music/model.py` | Architecture | **JUST FIXED** — scaled residual init |
| `src/genesis_music/train.py` | Training loop | Fixed — grad norm guard, save_interval=500 default |
| `src/genesis_music/ym2612.py` | VGM→tokens | Fixed — DAC onset detection |
| `src/genesis_music/dataset_v4.py` | Dataset | Fixed — DAC density filter |
| `data/prepared_v4/train.npy` | 2104M int16 tokens | Ready |
| `runs/v5d/` | Current checkpoints | In progress |
| `runs/v5/step1760_v5c_broken_init.pt` | Last v5c checkpoint (broken init) | Archived |
| `check_init.py` | Init verification script | At project root |

---

## Architecture Details (for Opus to review)

```
VgmGPT (253.5M params)
├── tok_emb: Embedding(456, 1024)          # weight-tied with lm_head
├── drop: Dropout(0.1)
├── layers: ModuleList × 20
│   └── TransformerBlock
│       ├── attn_norm: RMSNorm(1024)       # pre-norm
│       ├── attn: Attention
│       │   ├── qkv: Linear(1024, 3072)   # combined QKV, no bias
│       │   ├── out: Linear(1024, 1024)   # ← residual proj, NOW scaled
│       │   └── dropout: Dropout(0.1)
│       ├── ff_norm: RMSNorm(1024)        # pre-norm
│       └── ff: SwiGLUFF
│           ├── gate: Linear(1024, 4096)  # no bias
│           ├── up:   Linear(1024, 4096)  # no bias
│           ├── down: Linear(4096, 1024)  # ← residual proj, NOW scaled
│           └── dropout: Dropout(0.1)
├── norm: RMSNorm(1024)                    # final norm
└── lm_head: (tied to tok_emb)
```

- **RoPE**: θ=10000, no positional embeddings in tok_emb
- **Attention**: PyTorch SDPA (Flash Attention-like kernel), causal masking
- **Weight tying**: lm_head shares weights with tok_emb (saves ~0.5M params)
- **Dropout**: 0.1 in attention and FFN — applied during training

---

## Questions for Opus

1. **Is the scaled residual init sufficient?** We've been fighting explosion for 3 runs.
   The GPT-2 init fix is the textbook answer, but is there anything else in this
   architecture that could cause instability at scale? Specifically:
   - Dropout=0.1 with bfloat16 — any known interactions?
   - RoPE θ=10000 for sequences up to 16384 — is that appropriate?
   - No gradient clipping other than the skip-step guard — should we add `clip_grad_norm_`?

2. **Warmup schedule**: We're using linear warmup over 2000 steps to lr=1e-4. Is that
   appropriate for a 253M param model training from scratch? Previous runs used warmup=4000
   but still exploded. Should warmup be longer (e.g., cosine schedule to a lower peak lr)?

3. **Learning rate**: lr=1e-4 for 253M params, batch=8, seq=4096. Is this too high?
   Chinchilla-style: for ~2B tokens and 253M params, we're roughly at the right compute
   budget. But the architecture is unusual (no positional embeddings in embedding, weight
   tying, custom SwiGLU). Does 1e-4 seem right?

4. **Batch size**: Effective batch=8 sequences × 4096 tokens = 32,768 tokens/batch. For
   a 253M model, this is small. We're constrained by VRAM (RTX 3090 24GB). Gradient
   checkpointing is on. Is there a way to increase effective batch size without OOM?

5. **Alternative**: If this run also explodes, should we fall back to `--model-size medium`
   (16 layers, 12 heads, d_model=768, ~55M params) as a more tractable starting point?
   Or is there a training technique (e.g., µP / maximal update parametrization) that would
   be more robust for this use case?

6. **The `v5c` spiral**: After the init fix, the model started with high but declining
   norms (76→8 in 160 steps, loss 5.84→2.0). Is this expected behavior with properly
   scaled init — or should norms be much lower from step 1 with correct initialization?

---

## Context on the Domain

- VGM files are register-dump format from Sega Genesis hardware emulation
- Tokenization: FM note events (channel, pitch, velocity), timing (bar tokens, implicit
  timing from token order), DAC drum hits (8 sample slots), PSG tones/noise, composer ID
- The model needs to learn: harmonic relationships between FM channels, rhythmic patterns,
  instrument consistency within a "song", and stylistic features of ~10,000 game soundtracks
- This is a from-scratch training (no pretrained weights, no transfer learning)
- Goal: generate 30–60 second musical sequences that sound like Sega Genesis music

---

## What Sonnet Tried and Why It Thinks This Is Fixed

The previous session (Sonnet) was honest that it had missed this bug across 3 failed runs.
The mechanism is clear in retrospect:

With 20 layers and `std=0.02` on all projections including residual ones, the residual
stream at layer L has variance ≈ 1 + L × (4096 × 0.02²) relative to the embedding. At
layer 20, that's a significant amplification. As warmup increases lr from near-zero toward
1e-4, the gradient signal through the full depth amplifies proportionally, causing the
sudden norm explosion around step 1000–1500 (when lr reaches ~2.5e-5 to 5e-5).

The GPT-2 fix (scale by 1/sqrt(2N)) ensures each block adds variance scaled by 1/(2N),
so the total residual variance stays O(1) regardless of depth. With 20 layers, each
residual output projection std is reduced by ~6.3×.

Whether this is *sufficient* — or whether there are other contributing factors — is what
we'd like Opus's opinion on.

---

## Current State (as of writing)

- v5d running: step ~160, loss=2.01, grad_norm=10.63, lr=8e-6
- Training terminal ID: `4e498e17-d236-4503-965c-8815aac91d83` (PowerShell, async)
- The real test: steps 500–2000 (lr ramps 2.5e-5 → 1e-4)
- All prior fixes (DAC onset, TDR, dataset filter, grad norm guard) remain in place

Thank you for reviewing. The user has been very patient through multiple failed runs.
