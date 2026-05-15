"""Piano-roll comparison tool: original VGM vs reconstructed VGM.

Usage:
    python scripts/piano_roll.py --channel FM0
    python scripts/piano_roll.py --channel FM1 FM2 FM3
    python scripts/piano_roll.py --all
    python scripts/piano_roll.py --channel FM0 --start 0 --end 30

Outputs PNG files to output/piano_roll/.
"""

import sys, argparse, math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import matplotlib
matplotlib.use('Agg')  # non-interactive backend for file output
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

from genesis_music.vgm_parser import load_vgm
from genesis_music.ym2612 import (
    decode_vgm, CH_DAC, CH_PSG_0, CH_PSG_1, CH_PSG_2, CH_PSG_NOISE,
)

SAMPLE_RATE = 44100

CHANNEL_NAMES = {
    0: 'FM0', 1: 'FM1', 2: 'FM2', 3: 'FM3', 4: 'FM4', 5: 'FM5',
    CH_PSG_0: 'PSG0', CH_PSG_1: 'PSG1', CH_PSG_2: 'PSG2',
    CH_PSG_NOISE: 'NOISE', CH_DAC: 'DAC',
}
NAME_TO_CH = {v: k for k, v in CHANNEL_NAMES.items()}

SRC_VGZ = Path('data/vgm/Streets_of_Rage_2__Bare_Knuckle_II___Mega_Drive__Genesis___02_-_Go_Straight.vgz')
REGEN_DIR = Path('output/roundtrip/channels')
OUT_DIR   = Path('output/piano_roll')

MIDI_NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']


def midi_name(n: int) -> str:
    return f'{MIDI_NOTE_NAMES[n % 12]}{n // 12 - 1}'


def seconds_to_samples(sec: float) -> int:
    return int(sec * SAMPLE_RATE)


def load_original_notes(ch_id: int, start_s: int, end_s: int):
    vgm = load_vgm(SRC_VGZ)
    notes, _ = decode_vgm(vgm)
    return [n for n in notes
            if n.channel == ch_id
            and n.sample_off > start_s
            and n.sample_on < end_s]


def load_regen_notes(ch_name: str, ch_id: int, start_s: int, end_s: int):
    vgm_path = REGEN_DIR / f'ch_{ch_name}.vgm'
    if not vgm_path.exists():
        print(f'  WARNING: {vgm_path} not found, skipping regen side')
        return []
    vgm = load_vgm(vgm_path)
    notes, _ = decode_vgm(vgm)
    return [n for n in notes
            if n.channel == ch_id
            and n.sample_off > start_s
            and n.sample_on < end_s]


def plot_piano_roll(ax, notes, start_s: int, end_s: int, color: str, label: str):
    """Draw notes as horizontal bars on a piano-roll axes."""
    if not notes:
        ax.text(0.5, 0.5, 'No notes in range', ha='center', va='center',
                transform=ax.transAxes, color='gray', fontsize=10)
        ax.set_title(label, fontsize=11, color=color)
        return

    pitches = [n.pitch for n in notes if n.pitch >= 0]
    if not pitches:
        ax.set_title(label, fontsize=11, color=color)
        return

    for note in notes:
        if note.pitch < 0:
            continue
        x0 = max(note.sample_on,  start_s) / SAMPLE_RATE
        x1 = min(note.sample_off, end_s)   / SAMPLE_RATE
        if x1 <= x0:
            continue
        width = max(x1 - x0, 0.005)        # minimum visible width
        rect = mpatches.FancyBboxPatch(
            (x0, note.pitch - 0.4), width, 0.8,
            boxstyle='round,pad=0.02',
            facecolor=color, edgecolor='none', alpha=0.75,
        )
        ax.add_patch(rect)

    p_lo = max(0,   min(pitches) - 2)
    p_hi = min(127, max(pitches) + 2)
    ax.set_xlim(start_s / SAMPLE_RATE, end_s / SAMPLE_RATE)
    ax.set_ylim(p_lo, p_hi)
    ax.set_ylabel('MIDI note', fontsize=9)

    # Y axis: note names every semitone, label every C
    ax.yaxis.set_major_locator(ticker.MultipleLocator(12))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda v, _: midi_name(int(v)) if 0 <= int(v) <= 127 else ''))

    # Horizontal grid at C notes only
    for c in range(p_lo, p_hi + 1):
        if c % 12 == 0:
            ax.axhline(c, color='#cccccc', linewidth=0.5, zorder=0)

    ax.set_title(f'{label}  ({len(notes)} notes)', fontsize=11, color=color)
    ax.tick_params(axis='y', labelsize=8)
    ax.tick_params(axis='x', labelsize=8)
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.grid(axis='x', color='#eeeeee', linewidth=0.5)
    ax.set_facecolor('#1a1a2e')


def make_comparison(ch_name: str, start_sec: float, end_sec: float):
    ch_id   = NAME_TO_CH.get(ch_name.upper())
    if ch_id is None:
        print(f'Unknown channel: {ch_name}')
        return

    start_s = seconds_to_samples(start_sec)
    end_s   = seconds_to_samples(end_sec)

    print(f'Loading {ch_name} (t={start_sec:.1f}s–{end_sec:.1f}s)...')
    orig_notes  = load_original_notes(ch_id, start_s, end_s)
    regen_notes = load_regen_notes(ch_name.upper(), ch_id, start_s, end_s)

    print(f'  original: {len(orig_notes)} notes   regen: {len(regen_notes)} notes')

    # Stats
    orig_pitches  = sorted(set(n.pitch for n in orig_notes  if n.pitch >= 0))
    regen_pitches = sorted(set(n.pitch for n in regen_notes if n.pitch >= 0))
    print(f'  original pitches: {orig_pitches}')
    print(f'  regen    pitches: {regen_pitches}')

    fig, axes = plt.subplots(2, 1, figsize=(18, 7), sharex=True)
    fig.suptitle(f'Piano Roll: {ch_name}  |  {start_sec:.1f}s – {end_sec:.1f}s',
                 fontsize=13, fontweight='bold')

    plot_piano_roll(axes[0], orig_notes,  start_s, end_s, '#4fc3f7', 'ORIGINAL')
    plot_piano_roll(axes[1], regen_notes, start_s, end_s, '#ef9a9a', 'RECONSTRUCTED')

    # Shared x-axis time labels
    axes[1].set_xlabel('Time (s)', fontsize=9)

    plt.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    slug = f'{ch_name.upper()}_{int(start_sec)}-{int(end_sec)}s'
    out_path = OUT_DIR / f'pianoroll_{slug}.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
    plt.close(fig)
    print(f'  -> {out_path}')
    return out_path


def main():
    parser = argparse.ArgumentParser(description='Piano-roll VGM comparison tool')
    parser.add_argument('--channel', '-c', nargs='+',
                        help='Channel name(s): FM0-FM5, PSG0-PSG2, NOISE')
    parser.add_argument('--all', '-a', action='store_true',
                        help='Generate all FM + PSG channels')
    parser.add_argument('--start', '-s', type=float, default=0.0,
                        help='Start time in seconds (default: 0)')
    parser.add_argument('--end', '-e', type=float, default=30.0,
                        help='End time in seconds (default: 30)')
    args = parser.parse_args()

    if args.all:
        channels = ['FM0','FM1','FM2','FM3','FM4','PSG0','PSG1','PSG2']
    elif args.channel:
        channels = [c.upper() for c in args.channel]
    else:
        parser.print_help()
        return

    for ch in channels:
        make_comparison(ch, args.start, args.end)

    print('Done.')


if __name__ == '__main__':
    main()
