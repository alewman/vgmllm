# Generation Recipe — VgmGPT v6

## The Problem
- Background processes launched via `wsl bash -c "... &"` die when the WSL shell exits
- `nohup` + `disown` don't reliably survive from PowerShell → WSL
- Argument names in `generate.py` differ from what you might guess (see gotchas below)

## The Working Method

### Step 1 — Find the latest checkpoint
```powershell
wsl bash -c "ls /mnt/d/dev/genesis-music-ml/runs/v6_medium/step_*.pt | tail -5"
```

### Step 2 — Write a bash generation script
Create (or copy and edit) a file like `scripts/gen_v6_ken_XXXXX.sh`:

```bash
#!/bin/bash
source /home/alewman/venv/bin/activate
cd /mnt/d/dev/genesis-music-ml
PROMPT="data/vgm/Street_Fighter_II__-_Special_Champion_Edition__Mega_Drive__Genesis___11_-_Ken_s_Theme.vgz"
CKPT="runs/v6_medium/step_XXXXX.pt"
OUTDIR="output/v6_ken_XXXXX"
mkdir -p $OUTDIR logs
for i in 1 2 3 4 5; do
  OUTFILE=$(printf "%s/ken_XXXXX_%03d.vgm" $OUTDIR $i)
  python -m genesis_music.generate \
    --checkpoint $CKPT \
    --vocab-version v6 \
    --game-map data/prepared_v6/game_map_v6.json \
    --dac-slot-map data/prepared_v6/dac_slot_map_v6.json \
    --prompt-vgm $PROMPT \
    --prompt-tokens 256 \
    --max-tokens 8192 \
    --temperature 0.90 \
    --top-k 50 \
    --top-p 0.95 \
    --repetition-penalty 1.20 \
    --output $OUTFILE
done
```

### Step 3 — Write a Python launcher script
Create `scripts/launch_gen_XXXXX.py` (copy `scripts/launch_gen_13k.py` and edit the two paths):

```python
import subprocess, os
from pathlib import Path

base = Path(__file__).parents[1]
script   = str(base / "scripts" / "gen_v6_ken_XXXXX.sh")
log_path = str(base / "logs"    / "gen_v6_ken_XXXXX.log")

(base / "logs").mkdir(exist_ok=True)

with open(log_path, "w") as log:
    p = subprocess.Popen(
        ["bash", script],
        stdout=log, stderr=log,
        stdin=subprocess.DEVNULL,
        start_new_session=True,   # <-- this is the key; survives WSL shell exit
    )
print(f"Launched PID {p.pid} → {log_path}")
```

### Step 4 — Launch
```powershell
wsl bash -c "source /home/alewman/venv/bin/activate && cd /mnt/d/dev/genesis-music-ml && python scripts/launch_gen_XXXXX.py"
```

### Step 5 — Verify it's actually running
```powershell
wsl bash -c "ps aux | grep genesis | grep -v grep"
```
You should see a `python -m genesis_music.generate` process.

### Step 6 — Monitor
```powershell
# Log doesn't flush until first track completes (Python logging buffers)
wsl bash -c "tail -20 /mnt/d/dev/genesis-music-ml/logs/gen_v6_ken_XXXXX.log"
```
Typical timing with training also running on the GPU: **~4–6 min per track**.

---

## Argument Name Gotchas

| What you might write | Actual argument |
|---|---|
| `--prompt` | `--prompt-vgm` |
| `--rep-pen` | `--repetition-penalty` |
| `--rep-window` | `--repetition-window` |

Always verify with:
```powershell
wsl bash -c "source /home/alewman/venv/bin/activate && cd /mnt/d/dev/genesis-music-ml && python -m genesis_music.generate --help"
```

---

## Sampling Parameters — What They Do

| Arg | Value used | Effect |
|---|---|---|
| `--temperature` | 0.90 | Slightly creative, not too random |
| `--top-k` | 50 | Only sample from top 50 tokens at each step |
| `--top-p` | 0.95 | Nucleus sampling — cuts off the long tail |
| `--repetition-penalty` | 1.20 | Discourages exact repetition; sweet spot found empirically |
| `--prompt-tokens` | 256 | How many tokens of the VGM prompt to use as prefix |
| `--max-tokens` | 8192 | Hard ceiling; model will generate until EOS or this limit |

---

## Game Conditioning (no VGM prompt)
To condition on a game without a prompt VGM, use `--game` instead of `--prompt-vgm`:
```bash
python -m genesis_music.generate \
    --checkpoint runs/v6_medium/step_XXXXX.pt \
    --vocab-version v6 \
    --game-map data/prepared_v6/game_map_v6.json \
    --game "street fighter ii': special champion edition" \
    ...
```
Game names must match `data/prepared_v6/game_map_v6.json` exactly (case-insensitive).

---

## Known Issues / Notes
- **Log buffering**: The log file appears empty until the first track finishes. Use `ps aux` to confirm the process is running.
- **GPU sharing**: Training slows from ~13s/step to ~30s/step while generation runs on the same GPU. Both processes continue fine.
- **Track length**: Model doesn't reliably emit EOS; most tracks run to `--max-tokens`. Lower to ~4000 for shorter outputs (~1 min).
- **Output directory**: Created automatically by the script. VGM files are playable in VGMPlay or any Genesis emulator.
