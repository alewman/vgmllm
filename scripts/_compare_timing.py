"""Compare key-on timing between original and roundtrip VGMs.
Prints the first N timing differences per channel.
"""
import sys, gzip, struct
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent / 'src'))
from pathlib import Path
from genesis_music.vgm_parser import load_vgm
from genesis_music.ym2612 import decode_vgm

def load(path_str):
    p = Path(path_str)
    vgm = load_vgm(p)
    notes, _ = decode_vgm(vgm)
    return notes, vgm.header.total_samples

orig_src = 'data/vgm/Thunder_Force_IV__Lightening_Force___Mega_Drive__Genesis___23_-_Metal_Squad__Stage_8_.vgz'
rt_src   = 'output/roundtrip/metal_squad_roundtrip.vgm'

orig_notes, orig_total = load(orig_src)
rt_notes,   rt_total   = load(rt_src)

print(f"Original : {len(orig_notes)} notes, {orig_total} samples ({orig_total/44100:.3f}s)")
print(f"Roundtrip: {len(rt_notes)} notes, {rt_total} samples ({rt_total/44100:.3f}s)")
print()

# Group by channel
from collections import defaultdict
orig_by_ch = defaultdict(list)
rt_by_ch   = defaultdict(list)
for n in orig_notes:
    orig_by_ch[n.channel].append(n)
for n in rt_notes:
    rt_by_ch[n.channel].append(n)

CH_NAMES = {0:'FM0',1:'FM1',2:'FM2',3:'FM3',4:'FM4',5:'FM5',
            6:'PSG0',7:'PSG1',8:'PSG2',9:'NOISE',10:'DAC'}

for ch in sorted(set(list(orig_by_ch) + list(rt_by_ch))):
    on = orig_by_ch[ch]
    rn = rt_by_ch[ch]
    name = CH_NAMES.get(ch, f'CH{ch}')
    if not on or not rn:
        print(f"{name}: orig={len(on)} rt={len(rn)} — skipping")
        continue
    diffs = []
    for i, (a, b) in enumerate(zip(on, rn)):
        delta = b.sample_on - a.sample_on
        pdiff = b.pitch - a.pitch
        if delta != 0 or pdiff != 0:
            diffs.append((i, a.sample_on/44100, delta, a.pitch, b.pitch))
    if not diffs:
        print(f"{name}: {len(on)} notes — timing/pitch IDENTICAL ✓")
    else:
        print(f"{name}: {len(diffs)}/{len(on)} notes differ")
        for i, t, dt, op, rp in diffs[:8]:
            pitch_tag = f" pitch {op}→{rp}" if op != rp else ""
            print(f"  note {i:4d}  t={t:.3f}s  delta={dt:+d} samples{pitch_tag}")
        if len(diffs) > 8:
            print(f"  ... and {len(diffs)-8} more")
