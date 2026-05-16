"""Print a human-readable summary of any VGM/VGZ file.

Usage:
    python scripts/vgm_info.py <file.vgm> [file2.vgm ...]
    python scripts/vgm_info.py output/v5d/free_*.vgm
"""
import sys, gzip, struct
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent / 'src'))
from pathlib import Path
from collections import Counter
from genesis_music.vgm_parser import load_vgm
from genesis_music.ym2612 import decode_vgm, CH_PSG_0, CH_PSG_1, CH_PSG_2, CH_PSG_NOISE, CH_DAC

CH_NAMES = {
    0: 'FM0', 1: 'FM1', 2: 'FM2', 3: 'FM3', 4: 'FM4', 5: 'FM5',
    CH_PSG_0: 'PSG0', CH_PSG_1: 'PSG1', CH_PSG_2: 'PSG2',
    CH_PSG_NOISE: 'NOISE', CH_DAC: 'DAC',
}

NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

def midi_name(m):
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"

def bpm_guess(median_samples):
    """Guess BPM from median note duration — try common beat subdivisions."""
    if median_samples <= 0:
        return '?'
    dur_s = median_samples / 44100
    # Try quarter, eighth, half note
    candidates = []
    for beats in (0.5, 1.0, 2.0):
        bpm = 60 * beats / dur_s
        if 40 <= bpm <= 300:
            candidates.append(f'{bpm:.0f}')
    return ' / '.join(candidates) if candidates else f'({60/dur_s:.0f} bpm if 1 beat)'

def summarise(path: Path):
    p = path
    raw = gzip.decompress(p.read_bytes()) if p.suffix.lower() == '.vgz' else p.read_bytes()
    total_samples = struct.unpack_from('<I', raw, 0x18)[0]
    loop_samples  = struct.unpack_from('<I', raw, 0x20)[0]
    loop_offset   = struct.unpack_from('<I', raw, 0x1C)[0]
    ym2612_clock  = struct.unpack_from('<I', raw, 0x2C)[0]

    vgm = load_vgm(p)
    notes, patches = decode_vgm(vgm)

    duration_s = total_samples / 44100
    loop_s     = loop_samples / 44100 if loop_samples else 0
    has_loop   = loop_offset != 0

    print(f"\n{'='*60}")
    print(f"  {p.name}  ({p.stat().st_size/1024:.1f} KB)")
    print(f"{'='*60}")
    print(f"  Duration : {duration_s:.1f}s"
          + (f"  (loop from {(total_samples-loop_samples)/44100:.1f}s)" if has_loop else "  (no loop)"))
    print(f"  YM2612   : {ym2612_clock/1e6:.4f} MHz")
    print(f"  Notes    : {len(notes)} total")
    print()

    # Group by channel
    from collections import defaultdict
    by_ch = defaultdict(list)
    for n in notes:
        by_ch[n.channel].append(n)

    fm_channels = sorted(ch for ch in by_ch if 0 <= ch <= 5)
    other_channels = sorted(ch for ch in by_ch if ch > 5)

    # FM channels
    if fm_channels:
        print(f"  FM Channels ({len(fm_channels)} active):")
        for ch in fm_channels:
            ch_notes = by_ch[ch]
            pitches = [n.pitch for n in ch_notes if n.pitch >= 0]
            durs    = sorted([n.duration_samples for n in ch_notes if n.duration_samples > 0])
            median  = durs[len(durs)//2] if durs else 0
            tl_count = sum(1 for n in ch_notes if n.tl_envelope)

            # Patch usage
            patch_ids = [n.patch for n in ch_notes if n.patch is not None]
            unique_patches = len(set(id(p) for p in patch_ids))

            pitch_str = ''
            if pitches:
                pitch_str = f"{midi_name(min(pitches))}–{midi_name(max(pitches))}"

            bpm_str = bpm_guess(median)

            print(f"    {CH_NAMES[ch]:4s}: {len(ch_notes):4d} notes  "
                  f"pitch {pitch_str:8s}  "
                  f"dur {median/44100*1000:5.0f}ms  "
                  f"≈{bpm_str} bpm  "
                  f"{unique_patches} patch(es)"
                  + (f"  TL-fade:{tl_count}" if tl_count else ""))
        print()

    # PSG / Noise / DAC
    if other_channels:
        print(f"  Other Channels:")
        for ch in other_channels:
            ch_notes = by_ch[ch]
            name = CH_NAMES.get(ch, f'CH{ch}')
            if ch == CH_DAC:
                print(f"    {name:5s}: {len(ch_notes):4d} hits")
            elif ch == CH_PSG_NOISE:
                print(f"    {name:5s}: {len(ch_notes):4d} hits")
            else:
                pitches = [n.pitch for n in ch_notes if n.pitch >= 0]
                durs    = sorted([n.duration_samples for n in ch_notes if n.duration_samples > 0])
                median  = durs[len(durs)//2] if durs else 0
                pitch_str = f"{midi_name(min(pitches))}–{midi_name(max(pitches))}" if pitches else ''
                print(f"    {name:5s}: {len(ch_notes):4d} notes  pitch {pitch_str:8s}  dur {median/44100*1000:.0f}ms")
        print()

    # Overall feel
    all_fm = [n for ch in fm_channels for n in by_ch[ch]]
    if all_fm:
        all_durs = sorted([n.duration_samples for n in all_fm if n.duration_samples > 0])
        overall_median = all_durs[len(all_durs)//2] if all_durs else 0
        print(f"  Overall median note dur: {overall_median/44100*1000:.0f}ms  →  BPM guess: {bpm_guess(overall_median)}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for arg in sys.argv[1:]:
        for p in sorted(Path('.').glob(arg)) or [Path(arg)]:
            if p.exists():
                summarise(p)
            else:
                print(f"Not found: {p}")
