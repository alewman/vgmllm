import sys, random
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from genesis_music.vgm_parser import load_vgm
from genesis_music.tokenizer_v2 import encode_events_v2

_m = {"C":0,"C#":1,"D":2,"D#":3,"E":4,"F":5,"F#":6,"G":7,"G#":8,"A":9,"A#":10,"B":11}

def nm(n):
    for i in range(len(n)-1, 0, -1):
        if n[i].isdigit() or (n[i] == "-" and i == len(n)-1):
            return _m.get(n[:i], 0) + (int(n[i:]) + 1) * 12
    return 60

# Check real files
real_files = sorted(Path("data/vgm").glob("*.vgm"))
random.seed(42)
sample = random.sample(real_files, 3)

for f in sample:
    vgm = load_vgm(f)
    tokens = encode_events_v2(vgm.events, include_dac=False)
    ch_active = {}
    intervals = []
    for t in tokens:
        if ":ON:" in t:
            parts = t.split(":")
            ch, note = int(parts[0][2:]), parts[2]
            if note != "X" and ch in ch_active and ch_active[ch]:
                intervals.append(nm(note) - nm(ch_active[ch]))
            if note != "X":
                ch_active[ch] = note
    ic = Counter(intervals)
    stepwise = sum(1 for i in intervals if abs(i) in (1, 2))
    within_oct = sum(1 for i in intervals if abs(i) <= 12)
    print(f"{f.name}: {len(intervals)} intervals, stepwise={stepwise}, octave={within_oct}")
    print(f"  Top: {ic.most_common(10)}")

# Check generated files
print("\n--- Generated ---")
gen_files = sorted(
    list(Path("output/v2_sonic").glob("*.vgm"))
    + list(Path("output/v2_tfiv").glob("*.vgm"))
)
for f in gen_files[:3]:
    vgm = load_vgm(f)
    tokens = encode_events_v2(vgm.events, include_dac=False)
    ch_active = {}
    intervals = []
    for t in tokens:
        if ":ON:" in t:
            parts = t.split(":")
            ch, note = int(parts[0][2:]), parts[2]
            if note != "X" and ch in ch_active and ch_active[ch]:
                intervals.append(nm(note) - nm(ch_active[ch]))
            if note != "X":
                ch_active[ch] = note
    ic = Counter(intervals)
    stepwise = sum(1 for i in intervals if abs(i) in (1, 2))
    within_oct = sum(1 for i in intervals if abs(i) <= 12)
    print(f"{f.name}: {len(intervals)} intervals, stepwise={stepwise}, octave={within_oct}")
    print(f"  Top: {ic.most_common(10)}")
