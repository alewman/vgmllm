# VgmLLM

An AI model for generating Sega Genesis / Mega Drive music via YM2612 + SN76489 register writes in VGM format — plus a suite of visualization tools for analyzing and rendering chiptune music.

## Visualization Tools

### Combined Visualizer (`scripts/vgm_combined.py`)
Full Synthesia piano roll + per-channel oscilloscopes, rendered to MP4 or interactive preview.

**Features:**
- Per-channel WAV isolation via VGMPlay mute masks (FM1–FM6, DAC, PSG Sq1–3, Noise)
- Synthesia-style falling note bars with channel colours
- Real-time oscilloscopes for all active channels with adaptive windowing
- Seamless loop support with phantom notes
- GD3 metadata (title, composer) in header
- Silent pre-roll oscilloscopes before music starts
- 1080p export support

```bash
# Drag-and-drop via combined.bat in VGMPlay_040-9 folder
# Or directly:
python scripts/vgm_combined.py path/to/track.vgm --mp4 output.mp4 --vgmplay-dir path/to/VGMPlay
python scripts/vgm_combined.py path/to/track.vgm --mp4 output_1080p.mp4 --width 1920 --height 1080
```

### Oscilloscope (`scripts/vgm_oscilloscope.py`)
Standalone per-channel oscilloscope view.

### Synthesia Piano Roll (`scripts/vgm_synthesia.py`)
Standalone Synthesia-style falling-note visualizer.

## ML Components

| Script | Purpose |
|--------|---------|
| `vgmllm-pipeline` | Download & prepare VGM dataset |
| `vgmllm-tokenizer` | Tokenize VGM register writes |
| `vgmllm-dataset` | Build training dataset |
| `vgmllm-train` | Train the language model |
| `vgmllm-generate` | Generate new VGM tracks |

## Setup

```bash
pip install -e ".[dev]"
```

Requires:
- Python 3.10+
- [VGMPlay 0.40-9](https://vgmrips.net/forum/viewtopic.php?t=111) for per-channel WAV rendering
- ffmpeg in PATH for MP4 export
- pygame-ce for interactive/render mode

## Channel Colours

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

## License

MIT
