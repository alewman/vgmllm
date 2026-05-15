"""Regenerate v9 VGMs: PSG vibrato fix + verbatim DAC stream."""
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
non_dac = [n for n in notes if n.channel != CH_DAC]

# Count PSG0 vibrato entries
psg0_notes = [n for n in notes if n.channel == CH_PSG_0]
total_bends = sum(len(n.pitch_envelope) for n in psg0_notes)
print(f'PSG0: {len(psg0_notes)} notes, {total_bends} vibrato bend entries (avg {total_bends/len(psg0_notes):.1f}/note)')

# Full mix v9
full = synthesise_vgm(non_dac, total_samples, patches, pcm_data=pcm_data, dac_stream=dac_stream)
Path('output/roundtrip/go_straight_direct_v9.vgm').write_bytes(full)
print(f'v9 full mix: {len(full)} bytes')

# Per-channel files - regenerate PSG channels with vibrato fix
for ch_id, ch_name in [(CH_PSG_0, 'PSG0'), (CH_PSG_1, 'PSG1'), (CH_PSG_2, 'PSG2')]:
    ch_notes = [n for n in notes if n.channel == ch_id]
    ch_vgm = synthesise_vgm(ch_notes, total_samples, {})
    Path(f'output/roundtrip/channels/ch_{ch_name}.vgm').write_bytes(ch_vgm)
    bends = sum(len(n.pitch_envelope) for n in ch_notes)
    print(f'ch_{ch_name}: {len(ch_notes)} notes, {bends} bends, {len(ch_vgm)} bytes')

print('Done.')
