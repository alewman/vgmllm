# VgmLLM

A GPT-style transformer that learns to generate Sega Genesis / Mega Drive music by predicting YM2612 + SN76489 chip register write sequences — the same low-level data the hardware executes.

> **Status: active research, in-progress training.**  
> The model produces recognisably musical output (multi-channel FM, coordinated phrasing, drums) after ~53 K training steps. Timbre quality is the current bottleneck — see [Current status](#current-status).

---

## Sample outputs

🎵 **[gen_003.mp3](https://github.com/alewman/vgmllm/releases/download/v0.1.0-preview/gen_003.mp3)** — 51 s, 5 FM channels, step 53 499 (v5d run, unprompted)

The first genuinely musical unprompted output — `gen_003` from the v5d run at step 53 499 — features 5 FM channels, ~49 s duration, 667 ms median note duration, and audible coordination between channels. To reproduce it, download the v5d checkpoint from the releases page and run:

```bash
vgmllm-generate --checkpoint runs/v5d/step_053499.pt --vocab-version v4 \
    --patch-lib data/patch_library_v4.json --n 5 --output output/gen.vgm
```

---

## How it works

- **Tokenisation** — VGM files are streams of chip register writes (OPN2/YM2612 + SN76489). VgmLLM maps these to a ~660-token musical-concept vocabulary: note on/off per FM channel, beat-quantised timing, FM patch parameters, PSG tone/noise events. No raw byte tokens.
- **Model** — Decoder-only transformer (~55 M params): RoPE positional encoding, RMSNorm (pre-norm), SwiGLU FFN, scaled residual init, Flash Attention 2 / SDPA. Trained with bf16 mixed precision on an RTX 3090 (24 GB).
- **Training** — Autoregressive next-token prediction on ~19 K VGM files sourced from [vgmrips.net](https://vgmrips.net). Cosine LR schedule, AdamW (β₁=0.9, β₂=0.95), gradient checkpointing.
- **Generation** — Top-k / top-p sampling with KV-cache for O(1) per-step decode; outputs a `.vgm` file playable in any VGM-capable emulator or player.

---

## Current status

Training is iterating through tokeniser versions:

| Version | Vocab | Key change |
|---------|-------|------------|
| v3 | ~30 K tokens | Data-driven raw-register vocab (proof of concept) |
| v4 | 320 tokens | Musical-concept vocab: beat grid, channel roles, FM patch library |
| v6 | 660 tokens | Lossless FM operator parameters (no patch lookup table) |

**Known limitation (GIGO):** the model learns from tokeniser output, not original VGMs. Musical grammar (pitch, rhythm, channel roles) survives tokenisation well — the model learns it. FM timbres are currently degraded by lossy patch encoding, so generated tracks sound structurally musical but tonally imprecise. v6 tokeniser is the active fix.

---

## Quick start

```bash
pip install -e ".[dev]"

# 1. Download & prepare ~19 K VGM files from vgmrips.net
vgmllm-pipeline scrape && vgmllm-pipeline download && vgmllm-pipeline extract

# 2. Build tokeniser vocabulary
vgmllm-tokenizer build-vocab

# 3. Prepare training dataset
vgmllm-dataset prepare

# 4. Train (RTX 3090 recommended; ~10–20 days for 50 K steps at medium size)
vgmllm-train --model-size medium --output-dir runs/myrun

# 5. Generate tracks from a checkpoint
vgmllm-generate --checkpoint runs/myrun/latest.pt --n 5 --output output/gen.vgm
```

**Requirements:**
- Python 3.10+
- PyTorch 2.0+ with CUDA
- [VGMPlay 0.40-9](https://vgmrips.net/forum/viewtopic.php?t=111) — for per-channel WAV rendering (visualisers only)
- ffmpeg in PATH — for MP4 export (visualisers only)
- pygame-ce — for interactive visualiser mode

> **Windows note:** `torch.compile` is disabled by default (broken on Windows). Pass `--compile` explicitly on Linux/WSL.

---

## ML components

| CLI command | Module | Purpose |
|-------------|--------|---------|
| `vgmllm-pipeline` | `data_pipeline` | Download & extract VGM packs from vgmrips.net |
| `vgmllm-tokenizer` | `tokenizer_v6` | Build vocabulary, encode/decode VGM ↔ tokens |
| `vgmllm-dataset` | `dataset_v4` | Build memory-mapped training dataset |
| `vgmllm-train` | `train` | Training loop (bf16, grad accum, cosine LR) |
| `vgmllm-generate` | `generate` | Autoregressive generation with KV-cache |

### Architecture config

| Config | Params | Layers | Heads | d_model | Default seq_len |
|--------|--------|--------|-------|---------|-----------------|
| `config_small` | ~25 M | 8 | 8 | 512 | 4 096 |
| `config_medium` | ~55 M | 16 | 12 | 768 | 8 192 |
| `config_large` | ~85 M | 20 | 16 | 1 024 | 8 192 |

Stability options (off by default, backward-compatible with existing checkpoints):

```python
ModelConfig(qk_norm=True)         # RMSNorm Q & K before dot product
ModelConfig(logit_softcap=30.0)   # Gemma-style tanh logit cap
TrainConfig(z_loss=1e-4)          # PaLM z-loss
```

---

## Visualisation tools

### Combined visualiser (`scripts/vgm_combined.py`)

Full Synthesia piano roll + per-channel oscilloscopes, rendered to MP4 or interactive preview.

**Features:**
- Per-channel WAV isolation via VGMPlay mute masks (FM1–FM6, DAC, PSG Sq1–3, Noise)
- Synthesia-style falling note bars with channel colours
- Real-time oscilloscopes for all active channels with adaptive windowing
- Seamless loop support with phantom notes
- GD3 metadata (title, composer) in header
- 1080p export support

```bash
python scripts/vgm_combined.py path/to/track.vgm --mp4 output.mp4 --vgmplay-dir path/to/VGMPlay
python scripts/vgm_combined.py path/to/track.vgm --mp4 output_1080p.mp4 --width 1920 --height 1080
```

### Oscilloscope (`scripts/vgm_oscilloscope.py`)

Standalone per-channel oscilloscope view with corrscope YAML export.

```bash
python scripts/vgm_oscilloscope.py output/gen.vgm --mp4 --fps 60 --width 1920 --height 1080
```

### Synthesia piano roll (`scripts/vgm_synthesia.py`)

Standalone falling-note visualiser.

### Channel colours

| Channel | Colour |
|---------|--------|
| FM1 | ![#ff5533](https://placehold.co/12x12/ff5533/ff5533.png) Red-orange |
| FM2 | ![#ffcc00](https://placehold.co/12x12/ffcc00/ffcc00.png) Yellow |
| FM3 | ![#44dd44](https://placehold.co/12x12/44dd44/44dd44.png) Green |
| FM4 | ![#22ccdd](https://placehold.co/12x12/22ccdd/22ccdd.png) Cyan |
| FM5 | ![#4488ff](https://placehold.co/12x12/4488ff/4488ff.png) Blue |
| FM6 | ![#cc44ff](https://placehold.co/12x12/cc44ff/cc44ff.png) Purple |
| DAC | ![#ff44aa](https://placehold.co/12x12/ff44aa/ff44aa.png) Pink |
| PSG Sq1 | ![#dc8c32](https://placehold.co/12x12/dc8c32/dc8c32.png) Amber |
| PSG Sq2 | ![#3ca0dc](https://placehold.co/12x12/3ca0dc/3ca0dc.png) Sky blue |
| PSG Sq3 | ![#3cb45a](https://placehold.co/12x12/3cb45a/3cb45a.png) Green |
| Noise | ![#5a7896](https://placehold.co/12x12/5a7896/5a7896.png) Slate |

---

## Package note

The pip distribution name is `vgmllm` and console scripts are `vgmllm-*`.  
The Python import path is `genesis_music` (legacy internal name preserved for checkpoint reproducibility):

```python
from genesis_music.model import VgmGPT, config_medium
```

---

## License

[MIT](LICENSE)

