"""Scan original VGM and show exact key-on/key-off events for FM1 (enc=1)
plus the B4 pan register, to compare against what we synthesize."""
import sys, gzip, struct
sys.path.insert(0, 'src')
from pathlib import Path
from genesis_music.ym2612 import YM2612_CLOCK, fnumber_to_midi

data = gzip.decompress(Path('data/vgm/Streets_of_Rage_2__Bare_Knuckle_II___Mega_Drive__Genesis___02_-_Go_Straight.vgz').read_bytes())
data_start = 0x34 + struct.unpack_from('<I', data, 0x34)[0]

SR = 44100
s = 0; i = data_start
a1_lo = None
print(f'{"sample":>9}  {"time":>7}  event')
print('-' * 50)
while i < len(data) and s < 3 * SR:
    cmd = data[i]
    if cmd == 0x66:
        break
    elif cmd == 0x52:
        reg, val = data[i+1], data[i+2]
        if reg == 0xA1:   # FM1 F-num lo (port0, ch-offset 1)
            a1_lo = val
        elif reg == 0xA5:  # FM1 F-num hi+block
            blk = (val >> 3) & 7
            fhi = val & 7
            if a1_lo is not None:
                fnum = (fhi << 8) | a1_lo
                midi = fnumber_to_midi(fnum, blk)
                print(f'{s:>9}  {s/SR:>6.3f}s  FM1 F-num blk={blk} fnum={fnum} midi={midi}')
        elif reg == 0xB5:  # FM1 pan/AMS/FMS
            pan = (val >> 6) & 3
            pan_str = {0:'none', 1:'right', 2:'left', 3:'both'}[pan]
            print(f'{s:>9}  {s/SR:>6.3f}s  FM1 pan={pan_str} (0x{val:02X})')
        elif reg == 0x28:
            enc = val & 7
            ops = (val >> 4) & 0xF
            if enc == 1:  # FM1
                label = 'KEY-ON' if ops else 'key-off'
                print(f'{s:>9}  {s/SR:>6.3f}s  FM1 {label}  (0x{val:02X})')
        i += 3
    elif cmd == 0x53:
        i += 3
    elif cmd == 0x50:
        i += 2
    elif cmd == 0x61:
        s += struct.unpack_from('<H', data, i+1)[0]; i += 3
    elif cmd in (0x62, 0x63):
        s += 882 if cmd == 0x63 else 735; i += 1
    elif 0x70 <= cmd <= 0x7F:
        s += cmd - 0x70 + 1; i += 1
    elif 0x80 <= cmd <= 0x8F:
        i += 1
    elif cmd == 0x67:
        bs = struct.unpack_from('<I', data, i+3)[0] & 0x7FFFFFFF
        i += 7 + bs
    elif cmd == 0xE0:
        i += 5
    else:
        i += 1
