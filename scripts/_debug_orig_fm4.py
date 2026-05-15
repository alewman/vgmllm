import sys, gzip, struct
sys.path.insert(0, 'src')
from pathlib import Path
from genesis_music.ym2612 import optimal_block, midi_to_fnumber, YM2612_CLOCK

for midi in [56, 57, 58]:
    blk = optimal_block(midi)
    fn = midi_to_fnumber(midi, blk)
    freq = fn * YM2612_CLOCK / (144 * (1 << (20 - blk)))
    print(f'MIDI {midi}: optimal_block={blk} fnum={fn} freq={freq:.1f} Hz')

print()

data = gzip.decompress(Path('data/vgm/Streets_of_Rage_2__Bare_Knuckle_II___Mega_Drive__Genesis___02_-_Go_Straight.vgz').read_bytes())
data_start = 0x34 + struct.unpack_from('<I', data, 0x34)[0]

s = 0; i = data_start
print('Original FM4 raw writes (samples 320000-360000):')
while i < len(data) and s < 360000:
    cmd = data[i]
    if cmd == 0x66:
        break
    elif cmd == 0x52:
        reg, val = data[i+1], data[i+2]
        if s > 320000 and reg == 0x28:
            enc = val & 7
            ops = (val >> 4) & 0xF
            label = 'key-ON' if ops else 'key-off'
            print(f'  {s:>9}  p0 0x28=0x{val:02X}  {label} enc={enc}')
        i += 3
    elif cmd == 0x53:
        reg, val = data[i+1], data[i+2]
        if s > 320000 and reg in (0xA1, 0xA5):
            if reg == 0xA1:
                print(f'  {s:>9}  p1 0xA1=0x{val:02X}  FM4 F-num lo={val}')
            else:
                blk = (val >> 3) & 7
                fhi = val & 7
                print(f'  {s:>9}  p1 0xA5=0x{val:02X}  FM4 F-num hi={fhi} block={blk}')
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
