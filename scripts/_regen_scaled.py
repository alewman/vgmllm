"""Quick test: scale FM note durations to hear if notes-running-together is the issue.

Usage:
    python scripts/_regen_scaled.py 0.7    # 70% of original duration
    python scripts/_regen_scaled.py 0.5    # 50% of original duration
"""
import sys, gzip, struct
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent / 'src'))
from pathlib import Path
from genesis_music.vgm_parser import load_vgm
from genesis_music.ym2612 import decode_vgm, CH_DAC, CH_PSG_0, CH_PSG_1, CH_PSG_2, CH_PSG_NOISE, NoteEvent
from genesis_music.vgm_synth import synthesise_vgm

scale = float(sys.argv[1]) if len(sys.argv) > 1 else 0.7

src = Path('data/vgm/Streets_of_Rage_2__Bare_Knuckle_II___Mega_Drive__Genesis___02_-_Go_Straight.vgz')
orig = gzip.decompress(src.read_bytes())
data_start = 0x34 + struct.unpack_from('<I', orig, 0x34)[0]


def scan_dac(data, ds):
    events = []; s = 0; i = ds
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
vgm = load_vgm(src)
notes, patches = decode_vgm(vgm)
total_samples = vgm.header.total_samples

# Scale FM note durations only (leave PSG/NOISE alone — they already sound OK)
def scale_note(n: NoteEvent, s: float) -> NoteEvent:
    if 0 <= n.channel <= 5:
        dur = n.sample_off - n.sample_on
        new_off = n.sample_on + max(1, int(dur * s))
        return NoteEvent(
            channel=n.channel, pitch=n.pitch, velocity=n.velocity,
            sample_on=n.sample_on, sample_off=new_off,
            patch=n.patch, dac_sample_id=n.dac_sample_id,
            pitch_envelope=n.pitch_envelope,
        )
    return n

slug = str(scale).replace('.', 'p')
scaled_notes = [scale_note(n, scale) for n in notes if n.channel != CH_DAC]

# Full mix
out_dir = Path('output/roundtrip/scaled')
out_dir.mkdir(parents=True, exist_ok=True)
out = synthesise_vgm(scaled_notes, total_samples, patches, pcm_data=pcm_data, dac_stream=dac_stream)
out_path = out_dir / f'go_straight_scaled_{slug}.vgm'
out_path.write_bytes(out)
print(f'Full mix: {out_path}  ({len(out)} bytes, FM notes at {int(scale*100)}% duration)')

# Per-channel
ch_map = {
    0: 'FM0', 1: 'FM1', 2: 'FM2', 3: 'FM3', 4: 'FM4', 5: 'FM5',
    CH_PSG_0: 'PSG0', CH_PSG_1: 'PSG1', CH_PSG_2: 'PSG2',
    CH_PSG_NOISE: 'NOISE',
}
ch_dir = out_dir / 'channels'
ch_dir.mkdir(exist_ok=True)
for ch_id, ch_name in ch_map.items():
    ch_notes = [n for n in scaled_notes if n.channel == ch_id]
    if not ch_notes:
        continue
    ch_vgm = synthesise_vgm(ch_notes, total_samples, patches if ch_id <= 5 else {})
    p = ch_dir / f'ch_{ch_name}_{slug}.vgm'
    p.write_bytes(ch_vgm)
    print(f'  ch_{ch_name}: {len(ch_notes)} notes  →  {p.name}')

# DAC isolation (unchanged)
dac_only = synthesise_vgm([], total_samples, {}, pcm_data=pcm_data, dac_stream=dac_stream)
(ch_dir / f'ch_DAC_{slug}.vgm').write_bytes(dac_only)
print(f'  ch_DAC: {len(dac_only)} bytes')
print('Done.')
