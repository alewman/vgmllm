"""Regenerate v10 VGMs: fixed F-number formula + PSG vibrato + verbatim DAC."""
import sys, gzip, struct
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent / 'src'))
from pathlib import Path
from genesis_music.vgm_parser import load_vgm
from genesis_music.ym2612 import decode_vgm, CH_DAC, CH_PSG_0, CH_PSG_1, CH_PSG_2, CH_PSG_NOISE
from genesis_music.vgm_synth import synthesise_vgm

src = Path('data/vgm/Streets_of_Rage_2__Bare_Knuckle_II___Mega_Drive__Genesis___02_-_Go_Straight.vgz')
orig = gzip.decompress(src.read_bytes())
data_start = 0x34 + struct.unpack_from('<I', orig, 0x34)[0]


def scan_dac(data, ds):
    events = []
    s = 0; i = ds
    while i < len(data):
        cmd = data[i]
        if cmd == 0x66: break
        elif cmd == 0x67: bs = struct.unpack_from('<I', data, i+3)[0] & 0x7FFFFFFF; i += 7 + bs
        elif cmd in (0x52, 0x53, 0x4F): i += 3
        elif cmd == 0x50: i += 2
        elif cmd == 0x61: s += struct.unpack_from('<H', data, i+1)[0]; i += 3
        elif cmd in (0x62, 0x63): s += 882 if cmd == 0x63 else 735; i += 1
        elif 0x70 <= cmd <= 0x7F: s += cmd - 0x70 + 1; i += 1
        elif 0x80 <= cmd <= 0x8F: n = cmd & 0x0F; events.append((s, 'write', n)); s += n; i += 1
        elif cmd == 0xE0: events.append((s, 'seek', struct.unpack_from('<I', data, i+1)[0])); i += 5
        else: i += 1
    return events


# Extract PCM bank
pcm_data = None
i = data_start
while i < len(orig):
    cmd = orig[i]
    if cmd == 0x66: break
    elif cmd == 0x67:
        bs = struct.unpack_from('<I', orig, i+3)[0] & 0x7FFFFFFF
        pcm_data = bytes(orig[i+7:i+7+bs]); break
    elif cmd in (0x52, 0x53, 0x4F): i += 3
    elif cmd == 0x50: i += 2
    elif cmd == 0x61: i += 3
    elif cmd in (0x62, 0x63): i += 1
    elif 0x70 <= cmd <= 0x8F: i += 1
    elif cmd == 0xE0: i += 5
    else: i += 1

dac_stream = scan_dac(orig, data_start)
print(f'DAC stream: {len(dac_stream)} events')

vgm = load_vgm(src)
notes, patches = decode_vgm(vgm)
total_samples = vgm.header.total_samples
print(f'Decoded: {len(notes)} total notes')

non_dac = [n for n in notes if n.channel != CH_DAC]

# Full mix v10
full = synthesise_vgm(non_dac, total_samples, patches, pcm_data=pcm_data, dac_stream=dac_stream)
Path('output/roundtrip/go_straight_direct_v10.vgm').write_bytes(full)
print(f'v10 full mix: {len(full)} bytes')

# Per-channel files
ch_map = {
    0: 'FM0', 1: 'FM1', 2: 'FM2', 3: 'FM3', 4: 'FM4', 5: 'FM5',
    CH_PSG_0: 'PSG0', CH_PSG_1: 'PSG1', CH_PSG_2: 'PSG2',
    CH_PSG_NOISE: 'NOISE',
}

for ch_id, ch_name in ch_map.items():
    ch_notes = [n for n in notes if n.channel == ch_id]
    if not ch_notes:
        continue
    ch_vgm = synthesise_vgm(ch_notes, total_samples, patches if ch_id <= 5 else {})
    Path(f'output/roundtrip/channels/ch_{ch_name}.vgm').write_bytes(ch_vgm)
    print(f'ch_{ch_name}: {len(ch_notes)} notes, {len(ch_vgm)} bytes')

# DAC isolation
dac_only = synthesise_vgm([], total_samples, {}, pcm_data=pcm_data, dac_stream=dac_stream)
Path('output/roundtrip/channels/ch_DAC.vgm').write_bytes(dac_only)
print(f'ch_DAC: {len(dac_only)} bytes')

print('Done.')
