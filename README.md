# VgmLLM

A GPT-style transformer that learns to generate Sega Genesis / Mega Drive music by predicting YM2612 + SN76489 chip register write sequences — the same low-level data the hardware executes.

> **Status: active research, v7 training in progress.**  
> The model produces recognisably musical output (multi-channel FM, coordinated phrasing, drums) after ~53 K training steps. v7 adds hardware-state tokens, 18× augmentation, and training improvements (LLRD, curriculum learning) targeting better rare-feature and structural learning.

---

## Sample outputs

🎵 **[gen_003.mp3](https://github.com/alewman/vgmllm/releases/download/v0.1.0-preview/gen_003.mp3)** — 51 s, 5 FM channels, step 53 499 (v5d run, unprompted)

The first genuinely musical unprompted output — `gen_003` from the v5d run at step 53 499 — features 5 FM channels, ~49 s duration, 667 ms median note duration, and audible coordination between channels. To reproduce it, download the v5d checkpoint from the releases page and run:

```bash
python -m genesis_music.generate --checkpoint runs/v5d/step_053499.pt \
    --vocab-version v4 --patch-lib data/patch_library_v4.json \
    --n 5 --output output/gen.vgm
```

---

## How it works

- **Tokenisation** — VGM files are streams of chip register writes (OPN2/YM2612 + SN76489). VgmLLM maps these to a 1 024-token musical-concept vocabulary: note on/off per FM channel, beat-quantised timing, lossless FM patch parameters, PSG tone/noise events, and inline hardware-state tokens (panning, LFO, CH3 mode, DAC enable, loop point). No raw byte tokens.
- **Model** — Decoder-only transformer (~114 M params at medium): RoPE positional encoding, RMSNorm (pre-norm), SwiGLU FFN, scaled residual init, SDPA. Trained with bf16 mixed precision on an RTX 3090 (24 GB).
- **Training** — Autoregressive next-token prediction on ~19 K VGM files sourced from [vgmrips.net](https://vgmrips.net). 18× data augmentation (12-key transpose + 4 tempo factors + 2 velocity shifts), cluster-aware oversampling, layer-wise LR decay (LLRD), sequence-length curriculum, and per-token loss upweighting for rare hardware-state tokens.
- **Generation** — Top-k / top-p sampling with KV-cache for O(1) per-step decode; outputs a `.vgm` file playable in any VGM-capable emulator or player.

---

## Current status

Training is iterating through tokeniser versions:

| Version | Vocab | Key change |
|---------|-------|------------|
| v3 | ~30 K tokens | Data-driven raw-register vocab (proof of concept) |
| v4 | 320 tokens | Musical-concept vocab: beat grid, channel roles, FM patch library |
| v6 | 794 tokens | Lossless FM operator parameters (no patch lookup table); meter, game & composer tokens |
| **v7** | **1 024 tokens** | **Hardware-state tokens (PAN/LFO/CH3/DAC/LOOP); SSG-EG; 18× augmentation; curated 63-game map — active run** |

**Core insight (GIGO):** the model trains on tokeniser output, not original VGMs. Musical grammar (pitch, rhythm, channel roles) survives tokenisation well and is learned quickly. Rare hardware features (panning sweeps, LFO vibrato, CH3 special mode) appeared as noise in v6 — v7 encodes each as an explicit inline token and upweights their loss contribution.

---

## Quick start

```bash
pip install -e ".[dev]"

# 1. Download & prepare ~19 K VGM files from vgmrips.net
python -m genesis_music.data_pipeline scrape
python -m genesis_music.data_pipeline download
python -m genesis_music.data_pipeline extract

# 2. Cluster the corpus (assigns hardware-profile cluster IDs, ~30 min)
python -m genesis_music.clusters_v7 --vgm-dir data/vgm --out data/clusters_v7.json

# 3. Prepare v7 training dataset (~35 min on 12 cores)
python -m genesis_music.dataset_v7 \
    --vgm-dir data/vgm \
    --out-dir data/prepared_v7 \
    --cluster-map data/clusters_v7.json \
    --curated-games data/curated_games_v7.json

# 4. Train (RTX 3090 recommended; ~35 hrs for 100 K steps at medium size)
python -m genesis_music.train \
    --data-dir data/prepared_v7 \
    --output-dir runs/v7 \
    --tokenizer v7 \
    --model-size medium \
    --seq-len 8192 \
    --gradient-checkpointing \
    --llrd \
    --rare-token-weight 3.0 \
    --curriculum

# 5. Generate tracks from a checkpoint
python -m genesis_music.generate \
    --checkpoint runs/v7/latest.pt \
    --n 5 --output output/gen.vgm
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

| Module | Purpose |
|--------|---------|
| `data_pipeline` | Download & extract VGM packs from vgmrips.net |
| `tokenizer_v7` | 1 024-token vocab; encode/decode VGM ↔ tokens; hardware-state + SSG-EG tokens |
| `clusters_v7` | Hardware-profile clustering of the VGM corpus (5 clusters) |
| `dataset_v7` | Memory-mapped training dataset with 18× augmentation and cluster oversampling |
| `train` | Training loop: bf16, LLRD, curriculum learning, rare-token loss weighting |
| `generate` | Autoregressive generation with KV-cache, top-k/top-p sampling |

### Training flags (v7)

| Flag | Default | Purpose |
|------|---------|---------|
| `--llrd` | off | Layer-wise LR decay (bottom layers train slower, protecting learned phonetics) |
| `--llrd-decay` | 0.85 | Per-layer LR multiplier (1.0 = top layer) |
| `--rare-token-weight` | 1.0 | Loss multiplier for rare hardware-state tokens (3.0 recommended) |
| `--curriculum` | off | Ramp seq_len from `--curriculum-init` to `--curriculum-target` |
| `--curriculum-init` | 1 024 | Starting sequence length |
| `--curriculum-target` | 8 192 | Final sequence length |
| `--curriculum-warmup` | 8 000 | Steps to reach target |
| `--z-loss` | 0.0 | PaLM-style logit-scale regulariser (try 1e-4) |

### Architecture config

| Config | Params | Layers | Heads | d_model | Default seq_len |
|--------|--------|--------|-------|---------|-----------------|
| `config_small` | ~25 M | 8 | 8 | 512 | 4 096 |
| `config_medium` | ~114 M | 16 | 12 | 768 | 8 192 |
| `config_large` | ~85 M | 20 | 16 | 1 024 | 8 192 |

Optional stability knobs (off by default, backward-compatible):

```python
ModelConfig(qk_norm=True)         # RMSNorm on Q & K before dot product
ModelConfig(logit_softcap=30.0)   # Gemma-style tanh logit cap
TrainConfig(z_loss=1e-4)          # PaLM z-loss (logit scale regulariser)
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

