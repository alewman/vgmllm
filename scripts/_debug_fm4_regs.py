import sys, struct
sys.path.insert(0, 'src')
from pathlib import Path

data = Path('output/roundtrip/channels/ch_FM4.vgm').read_bytes()
data_start = 0x34 + struct.unpack_from('<I', data, 0x34)[0]

# FM4 = port1, channel offset 1
# Registers: 0xA1 (F-num lo), 0xA5 (F-num hi+block), key-on via port0 reg 0x28 (bits 5 = ch4)
s = 0; i = data_start
print('FM4 raw writes (samples 320000-360000):')
print(f'{"sample":>9}  port  reg   val   decoded')
while i < len(data) and s < 360000:
    cmd = data[i]
    if cmd == 0x66: break
    elif cmd == 0x52:
        reg, val = data[i+1], data[i+2]
        if s > 320000:
            if reg == 0x28:
                ch_enc = val & 0x07
                op_bits = (val >> 4) & 0x0F
                print(f'{s:>9}  p0  0x{reg:02X}  0x{val:02X}  key-{"ON" if op_bits else "off"} enc={ch_enc}')
            elif reg in (0xA1, 0xA5):  # port0 ch1 F-num (ch1 global) – not FM4
                pass
        i += 3
    elif cmd == 0x53:
        reg, val = data[i+1], data[i+2]
        if s > 320000:
            if reg == 0xA1:  # port1 ch-offset 1 = FM4
                print(f'{s:>9}  p1  0xA1  0x{val:02X}  FM4 F-num lo={val}')
            elif reg == 0xA5:
                block = (val >> 3) & 0x07
                fhi = val & 0x07
                print(f'{s:>9}  p1  0xA5  0x{val:02X}  FM4 F-num hi={fhi} block={block}')
        i += 3
    elif cmd == 0x50: i += 2
    elif cmd == 0x61: s += struct.unpack_from('<H', data, i+1)[0]; i += 3
    elif cmd in (0x62, 0x63): s += 882 if cmd == 0x63 else 735; i += 1
    elif 0x70 <= cmd <= 0x7F: s += (cmd - 0x70 + 1); i += 1
    elif 0x80 <= cmd <= 0x8F: i += 1
    elif cmd == 0x67:
        bs = struct.unpack_from('<I', data, i+3)[0] & 0x7FFFFFFF
        i += 7 + bs
    elif cmd == 0xE0: i += 5
    else: i += 1
