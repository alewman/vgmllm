"""Measure exact sample-level timing of key-off/key-on for FM1 in both
original and our synthesized VGM to find any micro-timing differences."""
import sys, gzip, struct
sys.path.insert(0, 'src')
from pathlib import Path

SR = 44100

def scan_keyon_events(data, data_start, enc_target, max_s=5*SR):
    """Return list of (sample, 'on'|'off') for the given key-on encoding."""
    events = []
    s = 0; i = data_start
    while i < len(data) and s < max_s:
        cmd = data[i]
        if cmd == 0x66: break
        elif cmd in (0x52, 0x53):
            reg, val = data[i+1], data[i+2]
            if reg == 0x28:
                enc = val & 7
                if enc == enc_target:
                    ops = (val >> 4) & 0xF
                    events.append((s, 'on' if ops else 'off'))
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
    return events

# Original
orig = gzip.decompress(Path('data/vgm/Streets_of_Rage_2__Bare_Knuckle_II___Mega_Drive__Genesis___02_-_Go_Straight.vgz').read_bytes())
orig_start = 0x34 + struct.unpack_from('<I', orig, 0x34)[0]
orig_ev = scan_keyon_events(orig, orig_start, enc_target=1)  # FM1 enc=1

# Ours (ch_FM1)
ours = Path('output/roundtrip/channels/ch_FM1.vgm').read_bytes()
ours_start = 0x34 + struct.unpack_from('<I', ours, 0x34)[0]
ours_ev = scan_keyon_events(ours, ours_start, enc_target=1)

def print_events(events, label):
    print(f'\n{label}:')
    print(f'  {"#":>3}  {"sample":>8}  {"time":>7}  type  gap_from_prev_off')
    last_off = None
    idx = 0
    for s, kind in events:
        gap = (s - last_off) if (last_off is not None and kind == 'on') else None
        gap_str = f'{gap:+d} samples' if gap is not None else ''
        print(f'  {idx:>3}  {s:>8}  {s/SR:>6.3f}s  {kind:<4}  {gap_str}')
        if kind == 'off': last_off = s
        idx += 1

print_events(orig_ev, 'ORIGINAL')
print_events(ours_ev, 'OURS')
