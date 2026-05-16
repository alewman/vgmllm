#!/usr/bin/env python3
"""vgm_piano_roll.py — Piano roll visualizer for VGM files.

Renders FM and PSG note events as a piano roll (time on X, MIDI pitch on Y),
coloured per channel.  DAC events are shown as a thin strip at the bottom
since they have no pitch.

Modes
-----
  (default)    Show interactive matplotlib window
  --png PATH   Save to PNG instead of showing
  --width W    Figure width in inches (default 20)
  --height H   Figure height in inches (default 10)

Usage
-----
  cd d:\\dev\\genesis-music-ml
  python scripts/vgm_piano_roll.py data/vgm/some_song.vgz
  python scripts/vgm_piano_roll.py output/roundtrip/some_song_v6_roundtrip.vgm
  python scripts/vgm_piano_roll.py data/vgm/some_song.vgz --png output/some_song_roll.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "src"))

from genesis_music.vgm_parser import load_vgm
from genesis_music.ym2612 import (
    CH_DAC, CH_FM_0, CH_PSG_0, CH_PSG_1, CH_PSG_2, CH_PSG_NOISE,
    decode_vgm,
)

SAMPLE_RATE = 44_100

# ── channel display config ────────────────────────────────────────────────────

CHANNEL_INFO = {
    0: ("FM 1", "#e05c5c"),
    1: ("FM 2", "#e08c3a"),
    2: ("FM 3", "#d4c830"),
    3: ("FM 4", "#5cbf5c"),
    4: ("FM 5", "#3ab0d4"),
    5: ("FM 6", "#7b6fe0"),
    CH_DAC:       ("DAC",   "#888888"),
    CH_PSG_0:     ("PSG 1", "#d47bbf"),
    CH_PSG_1:     ("PSG 2", "#bf9fd4"),
    CH_PSG_2:     ("PSG 3", "#94d4bf"),
    CH_PSG_NOISE: ("Noise", "#aaaaaa"),
}

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def midi_to_name(n: int) -> str:
    return f"{NOTE_NAMES[n % 12]}{n // 12 - 1}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Piano roll for VGM files")
    parser.add_argument("vgm", help="Path to .vgm / .vgz file")
    parser.add_argument("--png", metavar="PATH", help="Save PNG instead of showing")
    parser.add_argument("--width",  type=float, default=22, help="Figure width in inches")
    parser.add_argument("--height", type=float, default=10, help="Figure height in inches")
    args = parser.parse_args()

    vgm_path = Path(args.vgm)
    if not vgm_path.exists():
        sys.exit(f"File not found: {vgm_path}")

    print(f"Loading {vgm_path.name} …")
    vgm  = load_vgm(vgm_path)
    notes, _ = decode_vgm(vgm)
    total_samples = vgm.header.total_samples
    total_sec = total_samples / SAMPLE_RATE

    print(f"  {len(notes)} note events, {total_sec:.1f}s")

    # ── separate pitched from DAC/noise ──────────────────────────────────────

    pitched = [n for n in notes if n.pitch >= 0 and n.channel != CH_DAC]
    dac     = [n for n in notes if n.channel == CH_DAC]
    noise   = [n for n in notes if n.channel == CH_PSG_NOISE]

    # pitch range with a small margin
    if pitched:
        p_min = max(0,   min(n.pitch for n in pitched) - 2)
        p_max = min(127, max(n.pitch for n in pitched) + 2)
    else:
        p_min, p_max = 36, 84

    pitch_span = p_max - p_min

    # ── figure ───────────────────────────────────────────────────────────────

    fig, ax = plt.subplots(figsize=(args.width, args.height))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    # subtle horizontal lane lines at each octave boundary
    for octave in range(11):
        y = octave * 12
        if p_min <= y <= p_max:
            ax.axhline(y, color="#2a2a3e", linewidth=0.5, zorder=0)

    # ── draw pitched notes ───────────────────────────────────────────────────

    for note in pitched:
        t_on  = note.sample_on  / SAMPLE_RATE
        t_off = note.sample_off / SAMPLE_RATE if note.sample_off >= 0 else total_sec
        dur   = max(t_off - t_on, 0.02)  # minimum visible width

        color = CHANNEL_INFO.get(note.channel, ("?", "#ffffff"))[1]

        # subtle velocity shading: scale alpha 0.55–1.0
        vel_alpha = 0.55 + 0.45 * (note.velocity / 15) if note.velocity > 0 else 0.65

        rect = mpatches.FancyBboxPatch(
            (t_on, note.pitch - 0.4),
            dur, 0.8,
            boxstyle="round,pad=0.01",
            linewidth=0,
            facecolor=color,
            alpha=vel_alpha,
            zorder=2,
        )
        ax.add_patch(rect)

    # ── DAC strip at bottom ──────────────────────────────────────────────────

    dac_y = p_min - 2.5
    dac_color = CHANNEL_INFO[CH_DAC][1]
    for note in dac:
        t_on  = note.sample_on  / SAMPLE_RATE
        t_off = note.sample_off / SAMPLE_RATE if note.sample_off >= 0 else t_on + 0.05
        dur   = max(t_off - t_on, 0.02)
        rect = mpatches.FancyBboxPatch(
            (t_on, dac_y - 0.45),
            dur, 0.9,
            boxstyle="round,pad=0.01",
            linewidth=0,
            facecolor=dac_color,
            alpha=0.75,
            zorder=2,
        )
        ax.add_patch(rect)

    # noise strip just below DAC
    noise_y = p_min - 4.5
    noise_color = CHANNEL_INFO[CH_PSG_NOISE][1]
    for note in noise:
        t_on  = note.sample_on  / SAMPLE_RATE
        t_off = note.sample_off / SAMPLE_RATE if note.sample_off >= 0 else t_on + 0.05
        dur   = max(t_off - t_on, 0.02)
        rect = mpatches.FancyBboxPatch(
            (t_on, noise_y - 0.45),
            dur, 0.9,
            boxstyle="round,pad=0.01",
            linewidth=0,
            facecolor=noise_color,
            alpha=0.6,
            zorder=2,
        )
        ax.add_patch(rect)

    # ── Y axis: pitch labels ─────────────────────────────────────────────────

    yticks = list(range(p_min, p_max + 1, 12))
    ax.set_yticks(yticks)
    ax.set_yticklabels([midi_to_name(y) for y in yticks], color="#cccccc", fontsize=8)

    # extra ticks for DAC / Noise strips
    extra_ticks  = []
    extra_labels = []
    if dac:
        extra_ticks.append(dac_y)
        extra_labels.append("DAC")
    if noise:
        extra_ticks.append(noise_y)
        extra_labels.append("Noise")
    if extra_ticks:
        combined_ticks  = yticks + extra_ticks
        combined_labels = [midi_to_name(y) for y in yticks] + extra_labels
        ax.set_yticks(combined_ticks)
        ax.set_yticklabels(combined_labels, color="#cccccc", fontsize=8)

    # ── X axis: time ─────────────────────────────────────────────────────────

    ax.set_xlim(0, total_sec)
    ax.set_xlabel("Time (s)", color="#cccccc", fontsize=9)
    ax.xaxis.label.set_color("#cccccc")
    ax.tick_params(colors="#cccccc", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")

    y_bottom = (noise_y if noise else dac_y if dac else p_min) - 1.5
    ax.set_ylim(y_bottom, p_max + 1)

    # ── legend ───────────────────────────────────────────────────────────────

    present_channels = sorted(set(n.channel for n in notes))
    legend_handles = []
    for ch in present_channels:
        label, color = CHANNEL_INFO.get(ch, (f"ch{ch}", "#ffffff"))
        legend_handles.append(mpatches.Patch(color=color, label=label))

    ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=8,
        framealpha=0.3,
        facecolor="#1a1a2e",
        edgecolor="#444466",
        labelcolor="#cccccc",
    )

    ax.set_title(
        vgm_path.stem.replace("_", " "),
        color="#eeeeff",
        fontsize=11,
        pad=8,
    )

    plt.tight_layout()

    if args.png:
        out = Path(args.png)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
        print(f"Saved: {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
