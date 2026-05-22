#!/bin/bash
source /home/alewman/venv/bin/activate
cd /mnt/d/dev/genesis-music-ml
PROMPT="data/vgm/Street_Fighter_II__-_Special_Champion_Edition__Mega_Drive__Genesis___11_-_Ken_s_Theme.vgz"
CKPT="runs/v6_medium/step_017000.pt"
OUTDIR="output/v6_ken_17k"
mkdir -p $OUTDIR logs
for i in 1 2 3 4 5; do
  OUTFILE=$(printf "%s/ken_17k_%03d.vgm" $OUTDIR $i)
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
