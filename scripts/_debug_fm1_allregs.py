"""Dump ALL register writes for FM1 (ch-offset 1, both ports) in original VGM
around the 'brr brr brr' section (samples 30000-90000)."""
import sys, gzip, struct
sys.path.insert(0, 'src')
from pathlib import Path
from genesis_music.ym2612 import YM2612_CLOCK, fnumber_to_midi

SR = 44100
data = gzip.decompress(Path('data/vgm/Streets_of_Rage_2__Bare_Knuckle_II___Mega_Drive__Genesis___02_-_Go_Straight.vgz').read_bytes())
data_start = 0x34 + struct.unpack_from('<I', data, 0x34)[0]

# FM1 = port0, ch-offset 1
# Port0 ch1 registers: 0x31,0x35,0x39,0x3D (DT/MUL), 0x41,0x45,0x49,0x4D (TL),
# 0x51,0x55,0x59,0x5D (AR), 0x61,0x65,0x69,0x6D (AM/DR), 0x71,0x75,0x79,0x7D (SR),
# 0x81,0x85,0x89,0x8D (SL/RR), 0x91,0x95,0x99,0x9D (SSG-EG),
# 0xA1 (F-num lo), 0xA5 (F-num hi/block), 0xB1 (algo/fb), 0xB5 (pan/AMS/FMS)
# Global: 0x28 (key on/off)

FM1_REGS = set()
for base in (0x30,0x40,0x50,0x60,0x70,0x80,0x90):
    for slot in (1,5,9,13):  # ch-offset 1 + slot*4 → 1,5,9,13
        FM1_REGS.add(base + slot)
FM1_REGS |= {0xA1, 0xA5, 0xB1, 0xB5}

s = 0; i = data_start
print(f'{"sample":>9}  {"time":>7}  port  reg   val  meaning')
print('-' * 70)
while i < len(data) and s < 90000:
    cmd = data[i]
    if cmd == 0x66: break
    elif cmd in (0x52, 0x53):
        port = 0 if cmd == 0x52 else 1
        reg, val = data[i+1], data[i+2]
        show = False
        meaning = ''
        if s >= 30000:
            if port == 0:
                if reg in FM1_REGS:
                    show = True
                    meaning = f'FM1 op/patch reg'
                elif reg == 0x28 and (val & 7) == 1:
                    show = True
                    meaning = 'FM1 KEY-ON' if (val >> 4) else 'FM1 key-off'
        if show:
            print(f'{s:>9}  {s/SR:>6.3f}s  p{port}  0x{reg:02X}  0x{val:02X}  {meaning}')
        i += 3
    elif cmd == 0x50: i += 2
    elif cmd == 0x61: s += struct.unpack_from('<H', data, i+1)[0]; i += 3
    elif cmd in (0x62, 0x63): s += 882 if cmd == 0x63 else 735; i += 1
    elif 0x70 <= cmd <= 0x7F: s += cmd - 0x70 + 1; i += 1
    elif 0x80 <= cmd <= 0x8F: i += 1
    elif cmd == 0x67:
        bs = struct.unpack_from('<I', data, i+3)[0] & 0x7FFFFFFF
        i += 7 + bs
    elif cmd == 0xE0: i += 5
    else: i += 1
