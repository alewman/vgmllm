import numpy as np
from genesis_music.tokenizer_v4 import DAC_HIT_BASE, NOTE_ON, BAR, BOS

print("Loading train.npy memmap...", flush=True)
t = np.memmap('data/prepared_v4/train.npy', dtype=np.int16, mode='r')
total = len(t)
print(f"Loaded {total:,} tokens", flush=True)

print("Counting DAC tokens...", flush=True)
dac = int(((t >= DAC_HIT_BASE) & (t < DAC_HIT_BASE + 8)).sum())
note_on = int((t == NOTE_ON).sum())
bar = int((t == BAR).sum())

print()
print("=== Token Distribution ===")
print(f"Total tokens : {total:,}")
print(f"DAC_HIT      : {dac:,} = {100*dac/total:.2f}%")
print(f"NOTE_ON      : {note_on:,} = {100*note_on/total:.2f}%")
print(f"BAR          : {bar:,} = {100*bar/total:.2f}%")

print()
print("Finding BOS positions for song lengths...", flush=True)
positions = np.where(t == BOS)[0]
if len(positions) > 1:
    lengths = np.diff(positions)
    print(f"\n=== Song Token Lengths ({len(lengths):,} songs) ===")
    for pct in [25, 50, 75, 90, 95, 99]:
        print(f"  p{pct:2d}: {int(np.percentile(lengths, pct)):,}")
    print(f"  max:  {lengths.max():,}")
    print(f"  mean: {lengths.mean():.0f}")
else:
    print("No BOS tokens found!")
