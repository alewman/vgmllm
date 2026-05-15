"""Scan FM2 original register writes to compare with decoded values."""
import sys, gzip, struct
sys.path.insert(0, 'src')
from pathlib import Path
from genesis_music.vgm_parser import load_vgm
from genesis_music.ym2612 import decode_vgm, fnumber_to_midi

orig = gzip.decompress(Path('data/vgm/Streets_of_Rage_2__Bare_Knuckle_II___Mega_Drive__Genesis___02_-_Go_Straight.vgz').read_bytes())
data_start = 0x34 + struct.unpack_from('<I', orig, 0x34)[0]

# Scan raw FM2 (channel 2, port 0, offset=2) register writes
# 0xA2 = F-num low,  0xA6 = F-num high + block
s = 0; i = data_start
fnum_events = []  # (sample, reg, val)

while i < len(orig):
    cmd = orig[i]
    if cmd == 0x66: break
    elif cmd == 0x67: bs = struct.unpack_from('<I', orig, i+3)[0] & 0x7FFFFFFF; i += 7 + bs
    elif cmd == 0x52:
        reg, val = orig[i+1], orig[i+2]
        if reg in (0xA2, 0xA6):  # FM channel 2 pitch
            fnum_events.append((s, reg, val))
        i += 3
    elif cmd == 0x53: i += 3
    elif cmd == 0x50: i += 2
    elif cmd == 0x61: s += struct.unpack_from('<H', orig, i+1)[0]; i += 3
    elif cmd in (0x62, 0x63): s += 882 if cmd == 0x63 else 735; i += 1
    elif 0x70 <= cmd <= 0x7F: s += cmd - 0x70 + 1; i += 1
    elif 0x80 <= cmd <= 0x8F: s += cmd & 0x0F; i += 1
    elif cmd == 0xE0: i += 5
    else: i += 1

# Pair A2 (low) + A6 (high/block) writes
print('FM2 F-number writes (first 20 key-ons):')
fnum_low = None; fnum_low_smp = None
key_ons = 0
for smp, reg, val in fnum_events:
    if reg == 0xA2:
        fnum_low = val; fnum_low_smp = smp
    elif reg == 0xA6 and fnum_low is not None:
        block = (val >> 3) & 0x07
        fnum_hi = val & 0x07
        fnum = (fnum_hi << 8) | fnum_low
        midi = fnumber_to_midi(fnum, block)
        freq = fnum * (7670454 / 144) / (2 ** (20 - block))
        print(f'  smp={smp:7d} fnum={fnum:4d} block={block} -> midi={midi:3d} freq={freq:7.1f}Hz')
        fnum_low = None
        key_ons += 1
        if key_ons >= 20: break

# Compare with decoded
print()
vgm = load_vgm(Path('data/vgm/Streets_of_Rage_2__Bare_Knuckle_II___Mega_Drive__Genesis___02_-_Go_Straight.vgz'))
notes, patches = decode_vgm(vgm)
fm2 = [n for n in notes if n.channel == 2]
print(f'Decoded FM2 notes (first 10):')
for n in fm2[:10]:
    print(f'  smp={n.sample_on:7d} pitch={n.pitch:3d}')
