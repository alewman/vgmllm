"""Scan key-on/off writes for FM2 to find missing early notes."""
import gzip, struct
from pathlib import Path

orig = gzip.decompress(Path('data/vgm/Streets_of_Rage_2__Bare_Knuckle_II___Mega_Drive__Genesis___02_-_Go_Straight.vgz').read_bytes())
data_start = 0x34 + struct.unpack_from('<I', orig, 0x34)[0]

s = 0; i = data_start
key_events = []  # (sample, ch, is_on)

while i < len(orig):
    cmd = orig[i]
    if cmd == 0x66: break
    elif cmd == 0x67: bs = struct.unpack_from('<I', orig, i+3)[0] & 0x7FFFFFFF; i += 7 + bs
    elif cmd == 0x52:
        reg, val = orig[i+1], orig[i+2]
        if reg == 0x28:
            # Key on/off
            raw_ch = val & 0x07
            # ch encoding: 0=FM0, 1=FM1, 2=FM2, skip 3, 4=FM3, 5=FM4, 6=FM5
            if raw_ch <= 6 and raw_ch != 3:
                ch = raw_ch if raw_ch <= 2 else raw_ch - 1
                op_bits = (val >> 4) & 0x0F
                is_on = op_bits > 0
                key_events.append((s, ch, is_on, op_bits, val))
        i += 3
    elif cmd == 0x53: i += 3
    elif cmd == 0x50: i += 2
    elif cmd == 0x61: s += struct.unpack_from('<H', orig, i+1)[0]; i += 3
    elif cmd in (0x62, 0x63): s += 882 if cmd == 0x63 else 735; i += 1
    elif 0x70 <= cmd <= 0x7F: s += cmd - 0x70 + 1; i += 1
    elif 0x80 <= cmd <= 0x8F: s += cmd & 0x0F; i += 1
    elif cmd == 0xE0: i += 5
    else: i += 1

# Show all FM2 key events
fm2_events = [(s,ch,on,op,v) for s,ch,on,op,v in key_events if ch == 2]
print(f'FM2 key events: {len(fm2_events)} (first 20):')
for s,ch,on,op,v in fm2_events[:20]:
    print(f'  smp={s:7d} {"KEY-ON " if on else "KEY-OFF"} ops={op:04b} byte=0x{v:02X}')

print()
# Show first 10 key events across ALL channels at start
print('First 20 key events (all channels):')
for s,ch,on,op,v in key_events[:20]:
    print(f'  smp={s:7d} ch={ch} {"KEY-ON " if on else "KEY-OFF"} ops={op:04b} byte=0x{v:02X}')
