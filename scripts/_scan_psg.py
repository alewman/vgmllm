"""Scan PSG0 frequency writes in original VGM to understand vibrato pattern."""
import gzip, struct
from pathlib import Path
from collections import Counter

orig = gzip.decompress(Path('data/vgm/Streets_of_Rage_2__Bare_Knuckle_II___Mega_Drive__Genesis___02_-_Go_Straight.vgz').read_bytes())
data_start = 0x34 + struct.unpack_from('<I', orig, 0x34)[0]

s = 0; i = data_start
latch_ch = 0; latch_type = 0
psg_period = [0] * 4

# Track all freq writes to CH0 (PSG0)
ch0_freq_writes = []  # (sample, period, is_data_byte)

while i < len(orig):
    cmd = orig[i]
    if cmd == 0x66: break
    elif cmd == 0x67: bs = struct.unpack_from('<I', orig, i+3)[0] & 0x7FFFFFFF; i += 7 + bs
    elif cmd in (0x52, 0x53, 0x4F): i += 3
    elif cmd == 0x61: s += struct.unpack_from('<H', orig, i+1)[0]; i += 3
    elif cmd in (0x62, 0x63): s += 882 if cmd == 0x63 else 735; i += 1
    elif 0x70 <= cmd <= 0x7F: s += cmd - 0x70 + 1; i += 1
    elif 0x80 <= cmd <= 0x8F: n = cmd & 0x0F; s += n; i += 1
    elif cmd == 0xE0: i += 5
    elif cmd == 0x50:
        byte = orig[i+1]
        if byte & 0x80:
            latch_ch = (byte >> 5) & 0x03
            latch_type = (byte >> 4) & 0x01
            data4 = byte & 0x0F
            if latch_type == 0 and latch_ch < 3:
                psg_period[latch_ch] = (psg_period[latch_ch] & 0x3F0) | data4
                if latch_ch == 0:
                    ch0_freq_writes.append((s, psg_period[0], False))
        else:
            if latch_type == 0 and latch_ch < 3:
                psg_period[latch_ch] = ((byte & 0x3F) << 4) | (psg_period[latch_ch] & 0x0F)
                if latch_ch == 0:
                    ch0_freq_writes.append((s, psg_period[0], True))
        i += 2
    else:
        i += 1

print(f'Total PSG0 freq writes: {len(ch0_freq_writes)}')
latch_only = [(s,p) for s,p,is_data in ch0_freq_writes if not is_data]
data_bytes  = [(s,p) for s,p,is_data in ch0_freq_writes if is_data]
print(f'  LATCH-only (no DATA follows): checking...')
# Find LATCH writes NOT immediately followed by DATA write at same sample
# A "latch-only" write is one where next write at same/adjacent sample is also a latch
paired = set()
for idx,(s,p,is_data) in enumerate(ch0_freq_writes):
    if not is_data:
        # Check if next write is a DATA byte
        if idx+1 < len(ch0_freq_writes) and ch0_freq_writes[idx+1][2]:
            paired.add(idx)
latch_standalone = [ch0_freq_writes[i] for i in range(len(ch0_freq_writes)) if not ch0_freq_writes[i][2] and i not in paired]
print(f'  Standalone LATCH writes (vibrato): {len(latch_standalone)}')
print(f'  Paired LATCH+DATA writes: {len(paired)}')
print(f'  DATA-only writes: {len(data_bytes)}')

# Show first 20 writes to see vibrato pattern
print('\nFirst 30 PSG0 freq writes:')
for s, p, is_data in ch0_freq_writes[:30]:
    tag = 'DATA' if is_data else 'LTCH'
    freq = 3579545 / (32 * p) if p > 0 else 0
    print(f'  smp={s:7d} period={p:4d} freq={freq:7.1f}Hz [{tag}]')
