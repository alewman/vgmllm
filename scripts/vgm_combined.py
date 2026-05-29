#!/usr/bin/env python3
"""vgm_combined.py — Combined Synthesia + Oscilloscope visualizer for VGM files.

The Synthesia falling-notes visualizer runs as the background layer.
A grid of compact per-channel oscilloscope boxes (4 per row, semi-transparent)
overlays the falling-note zone — notes fall *behind* the boxes.
A thin header bar at the top shows the track title and timestamp.
The piano keyboard at the bottom is always unobstructed.

Modes
-----
  (default)    Interactive pygame window with audio (requires VGMPlay)
  --mp4 PATH   Render every frame and mux into an MP4 via ffmpeg

Usage
-----
  cd d:\\dev\\genesis-music-ml
  python scripts/vgm_combined.py output/v5d/gen_003.vgm
  python scripts/vgm_combined.py output/v5d/gen_003.vgm --mp4 output/v5d/gen_003_combined.mp4
  python scripts/vgm_combined.py data/vgm/some_song.vgz --width 1920 --height 1080 --fps 60
  python scripts/vgm_combined.py data/vgm/some_song.vgz --no-osc   # synthesia only

Layout (default 1280×720)
--------------------------
  ┌───────────────────────────────────────────────┐  ← top
  │  VgmLLM Combined · <title>       0:12.345     │  ← HEADER_H = 36 px (header bar)
  ├───────────────────────────────────────────────┤
  │  [FM1] [FM2] [FM3] [FM4]   ← oscilloscope    │
  │  [FM5] [FM6] [DAC] [PSG1]    boxes, 4/row     │  ← OSC_BOX_H = 110 px / row
  ├───────────────────────────────────────────────┤
  │   D  N  │  ← DAC/Noise strip                  │
  │   A  o  │  (falling notes pass behind         │
  │   C  i  │   the oscilloscope grid above)       │
  │   │  se │                                      │
  ├───────────────────────────────────────────────┤
  │              piano keyboard                    │  ← piano_h  = 140 px
  └───────────────────────────────────────────────┘  ← bottom

Z-order (back to front):
  1. Background fill
  2. Falling note bars
  3. Piano keyboard
  4. Oscilloscope grid (semi-transparent dark boxes, opaque traces + title chips)
  5. Header bar (fully opaque rounded rectangle)
"""

from __future__ import annotations

import argparse
import math
import random
import shutil
import subprocess
import sys
import time as _time
from pathlib import Path
from typing import Optional

import numpy as np

# ── path setup ─────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent   # scripts/
_ROOT = _HERE.parent                      # genesis-music-ml/
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_HERE))            # so sibling-script imports work

_DEFAULT_VGMPLAY_DIR = _ROOT.parent / "VGMPlay_040-9"

# ── imports from sibling scripts ───────────────────────────────────────────────
from vgm_oscilloscope import (           # noqa: E402
    VGMPlayRenderer,
    Channel,
    _make_channel_table,
    load_wav_mono,
    is_active,
    build_vgm_info,
    _make_is_active,
    _dynamic_title,
    get_oscilloscope_window,
    adaptive_window_samples,
    VGM_RATE,
    AUDIO_RATE,
)
from vgm_synthesia import (              # noqa: E402
    _parse_notes,
    NoteRect,
    CHANNEL_COLORS,
    _NOTE_IS_WHITE,
    _white_index,
    _note_key_info,
    _white_key_midi,
    BG_COLOR,
    PIANO_BG_COLOR,
    WHITE_KEY_CLR,
    BLACK_KEY_CLR,
    DIVIDER_COLOR,
    CH_DAC,
    CH_PSG_NOISE,
)

# ── layout constants ───────────────────────────────────────────────────────────
HEADER_H   = 36     # px — top title + timestamp bar
OSC_BOX_H  = 110    # px — default box height (4-col tier); overridden dynamically
OSC_MARGIN = 8      # px — gap between boxes / screen edges
OSC_PAD    = 4      # px — inner padding inside box
CHIP_H     = 15     # px — opaque title chip at top of each box

APP_NAME   = "VgmLLM"

# Ordered list of (channel_modes key, short display label) for the LCD slot row.
# Slots are rendered right-to-left from the right edge in this order, so the
# rightmost slot in the list ends up closest to the right edge.
# DAC and SPC are handled separately (not in channel_modes).
_OSC_MODE_SLOTS: tuple[tuple[str, str], ...] = (
    ("● L",     "L"),
    ("● R",     "R"),
    ("● VIB",   "VIB"),
    ("● TRE",   "TRE"),
    ("● SSG",   "SSG"),
    ("● PERIO", "PERIO"),
    ("● CH2",   "CH2"),
)

# Colour for an unlit LCD segment (dark, barely-there ghost).
_LCD_DIM = (42, 42, 60)

# ── particle system (Idea 3) ───────────────────────────────────────────────────
_MAX_PARTICLES = 1200  # hard cap to keep export-mode fast


class _Particle:
    """Spark particle for the piano-key multi-channel overlap effect."""
    __slots__ = ("x", "y", "vx", "vy", "r", "g", "b", "life")

    def __init__(self, x: float, y: float, vx: float, vy: float,
                 color: tuple) -> None:
        self.x, self.y   = x, y
        self.vx, self.vy = vx, vy
        self.r, self.g, self.b = color
        self.life = 1.0


def _osc_cols(n_channels: int) -> int:
    """Dynamic column count: ≤4 → 2 cols, 5-9 → 3 cols, 10+ → 4 cols."""
    if n_channels <= 4:
        return 2
    elif n_channels <= 9:
        return 3
    else:
        return 4


def _osc_box_h(n_rows: int) -> int:
    """Box height by row count: fewer rows → taller boxes."""
    if n_rows <= 1:
        return 180
    elif n_rows == 2:
        return 150
    elif n_rows == 3:
        return 130
    else:
        return 110


# ── colour helpers ─────────────────────────────────────────────────────────────

def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


# ── drawing: background (falling notes + piano) ───────────────────────────────

def _draw_background(
    surface,
    rects:       list[NoteRect],
    t:           float,          # current playback position (seconds)
    lookahead:   float,          # seconds visible above the piano
    screen_w:    int,
    screen_h:    int,
    piano_h:     int,
    top_y:       int,            # y where the falling zone begins (below header)
    dac_strip_w: int,
    pitch_min:   int,
    pitch_max:   int,
    white_w:     float,
    black_w:     float,
    min_white:   int,
    piano_x0:    float,
    font_key,
    particles: list | None = None,  # mutable spark list (updated in-place)
    dt:          float        = 0.0,  # seconds since last frame (for particles)
    bpm:         float        = 0.0,  # detected tempo for bar/beat grid lines
) -> None:
    """Draw the synthesia background: fill, falling note bars, and piano keyboard."""
    import pygame

    piano_y    = screen_h - piano_h
    fall_h     = piano_y - top_y
    if fall_h <= 0:
        return

    px_per_sec = fall_h / lookahead

    surface.fill(BG_COLOR)

    t_vis_start = t
    t_vis_end   = t + lookahead

    def _time_to_y(ts: float) -> float:
        # ts==t → piano_y (bottom of fall zone)
        # ts==t+lookahead → top_y (top of fall zone)
        return piano_y - (ts - t) * px_per_sec

    # ── bar / beat grid lines ──────────────────────────────────────────────────
    if bpm > 0.0:
        import math as _math
        beat_sec = 60.0 / bpm
        bar_sec  = beat_sec * 4          # assume 4/4
        # which beat index is the first one visible?
        first_beat = int(_math.floor(t_vis_start / beat_sec))
        beat_idx   = first_beat
        while True:
            bt = beat_idx * beat_sec
            if bt > t_vis_end:
                break
            y = int(_time_to_y(bt))
            if top_y <= y <= piano_y:
                is_downbeat = (beat_idx % 4 == 0)
                if is_downbeat:
                    pygame.draw.line(surface, (55, 55, 88),
                                     (dac_strip_w, y), (screen_w, y), 1)
                    bar_num = beat_idx // 4 + 1
                    lbl = font_key.render(str(bar_num), True, (80, 80, 115))
                    surface.blit(lbl, (dac_strip_w + 3, y - lbl.get_height() - 1))
                else:
                    pygame.draw.line(surface, (32, 32, 54),
                                     (dac_strip_w, y), (screen_w, y), 1)
            beat_idx += 1

    # ── falling note bars ──────────────────────────────────────────────────────
    # Pre-group pitched notes by pitch to detect multi-channel overlap (Idea 1).
    _pitch_groups: dict[int, list] = {}
    for nr in rects:
        if nr.t_off < t_vis_start or nr.t_on > t_vis_end:
            continue
        if nr.pitch >= 0 and nr.channel != CH_DAC:
            _pitch_groups.setdefault(nr.pitch, []).append(nr)

    for nr in rects:
        if nr.t_off < t_vis_start or nr.t_on > t_vis_end:
            continue

        clipped_on  = max(nr.t_on,  t_vis_start)
        clipped_off = min(nr.t_off, t_vis_end)

        y_bottom = _time_to_y(clipped_on)
        y_top    = _time_to_y(clipped_off)
        bar_h    = max(y_bottom - y_top, 2)

        r_c, g_c, b_c = nr.color

        if nr.pitch >= 0 and nr.channel != CH_DAC:
            rel_x, key_w, _ = _note_key_info(nr.pitch, min_white, white_w, black_w)
            bx = piano_x0 + rel_x
            bw = max(key_w - 2, 2)

            # Find every visible note at this pitch that overlaps this one in time.
            same_time = [
                o for o in _pitch_groups[nr.pitch]
                if o.t_on < nr.t_off and o.t_off > nr.t_on
            ]
            N   = len(same_time)
            idx = next(i for i, o in enumerate(same_time) if o is nr)

            if N == 1:
                # ── single channel: normal rounded bar ─────────────────────────
                pygame.draw.rect(surface, (r_c, g_c, b_c),
                                 pygame.Rect(int(bx), int(y_top), int(bw), int(bar_h)),
                                 border_radius=3)
                if bar_h > 4:
                    pygame.draw.rect(surface,
                                     (min(r_c+80,255), min(g_c+80,255), min(b_c+80,255)),
                                     pygame.Rect(int(bx), int(y_top), int(bw), 3),
                                     border_radius=3)
            else:
                # ── multi-channel: wide solid bands with narrow blend zones ────────
                # 32 internal px per channel; 5-px smoothstep blend per seam (≈16%).
                # Every note draws its OWN time rect with the shared gradient so
                # notes that start/end at different times are all fully visible.
                # (The overlapping region is drawn twice but produces identical pixels.)
                _S, _B = 32, 5
                grad_s = pygame.Surface((N * _S, 1))
                for ci, o in enumerate(same_time):
                    bs = ci * _S
                    bl = _B if ci > 0     else 0
                    br = _B if ci < N - 1 else 0
                    grad_s.fill(o.color, (bs + bl, 0, _S - bl - br, 1))
                    for px in range(bl):
                        tb = (px + 0.5) / bl
                        tb = tb * tb * (3.0 - 2.0 * tb)  # smoothstep
                        pc = same_time[ci - 1].color
                        grad_s.set_at(
                            (bs + px, 0),
                            tuple(int(pc[k] + (o.color[k] - pc[k]) * tb)
                                  for k in range(3)),
                        )
                    for px in range(br):
                        tb = (px + 0.5) / br
                        tb = tb * tb * (3.0 - 2.0 * tb)  # smoothstep
                        nc = same_time[ci + 1].color
                        grad_s.set_at(
                            (bs + _S - br + px, 0),
                            tuple(int(o.color[k] + (nc[k] - o.color[k]) * tb)
                                  for k in range(3)),
                        )
                surface.blit(
                    pygame.transform.smoothscale(grad_s, (int(bw), int(bar_h))),
                    (int(bx), int(y_top)),
                )
                # Additive glow strip at the leading (top) edge of this note
                if bar_h > 3:
                    mixed = tuple(
                        min(255, sum(o.color[i] for o in same_time))
                        for i in range(3)
                    )
                    glow_h = min(4, int(bar_h))
                    glow_s = pygame.Surface((int(bw), glow_h), pygame.SRCALPHA)
                    glow_s.fill((*mixed, 220))
                    surface.blit(glow_s, (int(bx), int(y_top)))

        else:
            # DAC / PSG_NOISE → left strip.
            # PSG tone channels with pitch=-1 (period=0/mute) are discarded here
            # so they don't pollute the DAC strip.
            if nr.channel not in (CH_DAC, CH_PSG_NOISE):
                continue
            strip_slot = [CH_DAC, CH_PSG_NOISE].index(nr.channel)
            slot_w = dac_strip_w // 2
            x      = strip_slot * slot_w
            w      = slot_w - 2
            pygame.draw.rect(surface, (r_c, g_c, b_c),
                             pygame.Rect(int(x), int(y_top), int(w), int(bar_h)),
                             border_radius=2)

    # ── piano keyboard ─────────────────────────────────────────────────────────
    pygame.draw.rect(surface, PIANO_BG_COLOR,
                     pygame.Rect(0, piano_y, screen_w, piano_h))
    pygame.draw.line(surface, (80, 80, 130),
                     (0, piano_y), (screen_w, piano_y), 2)

    active_pitches: set[int]        = set()
    active_ch:      dict[int, list] = {}   # pitch → [channel_ids]  (multi-aware)
    active_vel:     dict[int, dict] = {}   # pitch → {channel_id → velocity 0-15}
    for nr in rects:
        if nr.t_on <= t < nr.t_off and nr.pitch >= 0:
            active_pitches.add(nr.pitch)
            active_ch.setdefault(nr.pitch, []).append(nr.channel)
            active_vel.setdefault(nr.pitch, {})[nr.channel] = nr.velocity

    def _key_color(ch_ids: list, boost: int, fallback) -> tuple:
        """Additive light-mix piano key colour: more channels → brighter key."""
        return (
            min(255, sum(CHANNEL_COLORS.get(c, (150,150,150))[0] for c in ch_ids) + boost),
            min(255, sum(CHANNEL_COLORS.get(c, (150,150,150))[1] for c in ch_ids) + boost),
            min(255, sum(CHANNEL_COLORS.get(c, (150,150,150))[2] for c in ch_ids) + boost),
        )

    def _key_gradient(ch_ids: list, boost: int,
                      kx: int, ky: int, kw: int, kh: int) -> None:
        """Wide-band gradient across a piano key for multi-channel notes."""
        N_k      = len(ch_ids)
        _S, _B   = 32, 5
        gs       = pygame.Surface((N_k * _S, 1))
        for ci, ch_id in enumerate(ch_ids):
            base = CHANNEL_COLORS.get(ch_id, (200, 200, 200))
            lit  = tuple(min(255, c + boost) for c in base)
            bs   = ci * _S
            bl   = _B if ci > 0       else 0
            br   = _B if ci < N_k - 1 else 0
            gs.fill(lit, (bs + bl, 0, _S - bl - br, 1))
            for px in range(bl):
                tb = (px + 0.5) / bl; tb = tb * tb * (3.0 - 2.0 * tb)
                pc = tuple(min(255, c + boost)
                           for c in CHANNEL_COLORS.get(ch_ids[ci - 1], (200, 200, 200)))
                gs.set_at((bs + px, 0),
                          tuple(int(pc[k] + (lit[k] - pc[k]) * tb) for k in range(3)))
            for px in range(br):
                tb = (px + 0.5) / br; tb = tb * tb * (3.0 - 2.0 * tb)
                nc = tuple(min(255, c + boost)
                           for c in CHANNEL_COLORS.get(ch_ids[ci + 1], (200, 200, 200)))
                gs.set_at((bs + _S - br + px, 0),
                          tuple(int(lit[k] + (nc[k] - lit[k]) * tb) for k in range(3)))
        surface.blit(pygame.transform.smoothscale(gs, (kw, kh)), (kx, ky))

    white_key_h = piano_h - 4
    black_key_h = int(white_key_h * 0.60)
    n_whites    = _white_index(pitch_max) - min_white + 1

    # White keys
    for wi in range(n_whites):
        x    = int(piano_x0 + wi * white_w)
        w    = int(white_w) - 1
        midi = _white_key_midi(wi + min_white)
        if midi is None:
            continue
        if midi in active_pitches:
            wch = active_ch[midi]
            if len(wch) > 1:
                _key_gradient(wch, 60, x, piano_y + 2, w, white_key_h - 2)
            else:
                pygame.draw.rect(surface, _key_color(wch, 60, WHITE_KEY_CLR),
                                 pygame.Rect(x, piano_y + 2, w, white_key_h - 2),
                                 border_radius=2)
        else:
            pygame.draw.rect(surface, WHITE_KEY_CLR,
                             pygame.Rect(x, piano_y + 2, w, white_key_h - 2),
                             border_radius=2)
        pygame.draw.rect(surface, (80, 80, 100),
                         pygame.Rect(x, piano_y + 2, w, white_key_h - 2),
                         width=1, border_radius=2)
        # Middle C marker: small triangle tab at the bottom of MIDI 60 (C4)
        if midi == 60 and pitch_min <= 60 <= pitch_max:
            tab_cx = x + w // 2
            tab_b  = piano_y + white_key_h - 1   # bottom of key area
            tab_h  = min(7, white_key_h // 5)
            pygame.draw.polygon(
                surface, (80, 80, 110),   # dark blue-grey, readable on white
                [(tab_cx - tab_h, tab_b),
                 (tab_cx + tab_h, tab_b),
                 (tab_cx,         tab_b - tab_h)],
            )

    # Black keys
    for wi in range(n_whites):
        midi_white = _white_key_midi(wi + min_white)
        if midi_white is None:
            continue
        midi_black = midi_white + 1
        if midi_black <= pitch_max and not _NOTE_IS_WHITE[midi_black % 12]:
            bkx = int(piano_x0 + wi * white_w + white_w * 0.55)
            bkw = int(black_w)
            if midi_black in active_pitches:
                bch = active_ch[midi_black]
                if len(bch) > 1:
                    _key_gradient(bch, 40, bkx, piano_y + 2, bkw, black_key_h)
                else:
                    pygame.draw.rect(surface, _key_color(bch, 40, BLACK_KEY_CLR),
                                     pygame.Rect(bkx, piano_y + 2, bkw, black_key_h),
                                     border_radius=2)
            else:
                pygame.draw.rect(surface, BLACK_KEY_CLR,
                                 pygame.Rect(bkx, piano_y + 2, bkw, black_key_h),
                                 border_radius=2)

    # ── DAC / Noise strip labels ───────────────────────────────────────────────
    if dac_strip_w > 0:
        pygame.draw.line(surface, DIVIDER_COLOR,
                         (dac_strip_w, top_y), (dac_strip_w, screen_h), 1)
        for i, (ch_id, label) in enumerate([(CH_DAC, "DAC"), (CH_PSG_NOISE, "Noise")]):
            slot_w = dac_strip_w // 2
            r2, g2, b2 = CHANNEL_COLORS.get(ch_id, (150, 150, 150))
            if font_key:
                txt = font_key.render(label, True, (r2, g2, b2))
                surface.blit(txt, (i * slot_w + 2, top_y + 4))

    # ── particle sparks — update, spawn, draw (Idea 3) ────────────────────────
    if particles is not None and dt > 0.0:
        # Move, apply gravity, age, and prune existing sparks
        survivors = []
        for p in particles:
            p.life -= dt * 0.7   # slower fade → long arcs
            if p.life <= 0.0:
                continue
            p.x  += p.vx * dt
            p.y  += p.vy * dt
            p.vy += 130.0 * dt  # gravity
            survivors.append(p)
        particles[:] = survivors

        # Spawn new sparks from each actively-held pitch × channel
        if len(particles) < _MAX_PARTICLES:
            for pitch, ch_ids in active_ch.items():
                rel_x, key_w, _ = _note_key_info(pitch, min_white, white_w, black_w)
                kx = piano_x0 + rel_x + key_w * 0.5
                ky = float(piano_y)
                for ch_id in ch_ids:
                    # velocity 0–15 → normalised 0.0–1.0
                    # louder notes fire more sparks, faster, over a wider spread
                    vel_norm = active_vel.get(pitch, {}).get(ch_id, 15) / 15.0
                    # spawn probability scales with volume: vel=0 → 20%, vel=15 → 80%
                    if random.random() > (0.20 + 0.60 * vel_norm):
                        continue
                    base_col = CHANNEL_COLORS.get(ch_id, (200, 200, 200))
                    speed    = 0.35 + 0.65 * vel_norm   # 35–100% of max speed
                    count    = max(1, round(vel_norm * 4))  # 1–4 sparks
                    for _ in range(random.randint(count, count + 1)):
                        if len(particles) >= _MAX_PARTICLES:
                            break
                        particles.append(_Particle(
                            kx + random.uniform(-key_w * 0.5 * vel_norm,
                                                 key_w * 0.5 * vel_norm),
                            ky,
                            random.uniform(-50.0 * speed,  50.0 * speed),
                            random.uniform(-170.0 * speed, -50.0 * speed),
                            base_col,
                        ))

        # Draw: outer body + bright core
        for p in particles:
            fade       = p.life
            r_p        = int(p.r * fade)
            g_p        = int(p.g * fade)
            b_p        = int(p.b * fade)
            size       = max(2, int(fade * 4))
            cx_i, cy_i = int(p.x), int(p.y)
            # soft outer halo
            pygame.draw.circle(surface, (r_p // 2, g_p // 2, b_p // 2),
                               (cx_i, cy_i), size + 1)
            # main body
            pygame.draw.circle(surface, (r_p, g_p, b_p), (cx_i, cy_i), size)
            # bright core
            pygame.draw.circle(
                surface,
                (min(255, r_p + 110), min(255, g_p + 110), min(255, b_p + 110)),
                (cx_i, cy_i), max(1, size - 1),
            )

    # ── playhead marker ────────────────────────────────────────────────────────
    pygame.draw.line(surface, (200, 200, 255),
                     (dac_strip_w, piano_y - 1), (screen_w, piano_y - 1), 2)


# ── drawing: oscilloscope grid ─────────────────────────────────────────────────

def _draw_osc_grid(
    surface,
    channels:             list[Channel],
    pos_sample:           int,
    sample_rate:          int,
    grid_x:               int,
    grid_y:               int,
    grid_w:               int,
    cols:                 int,
    window_s:             float,
    autoscale:            bool,
    vgm_pos:              int,
    dac_is_active,
    ch3_special_is_active,
    font_sm,
    music_started:        bool = True,
    box_h:                int  = OSC_BOX_H,
    channel_modes:        dict | None = None,
) -> None:
    """Overlay a grid of compact oscilloscope boxes on the surface."""
    import pygame

    n = len(channels)
    if n == 0:
        return

    default_win = int(window_s * sample_rate)
    col_w       = grid_w // cols   # pixel width of one cell

    for idx, ch in enumerate(channels):
        col = idx % cols
        row = idx // cols
        bx  = grid_x + col * col_w
        by  = grid_y + row * box_h

        inner_w = col_w - 2
        inner_h = box_h - 2

        # ── box surface (SRCALPHA for semi-transparent background) ─────────────
        box = pygame.Surface((inner_w, inner_h), pygame.SRCALPHA)
        pygame.draw.rect(box, (10, 10, 20, 150),
                         pygame.Rect(0, 0, inner_w, inner_h), border_radius=4)

        # ── trace geometry ─────────────────────────────────────────────────────
        trace_y0 = CHIP_H + 1 + OSC_PAD
        trace_h  = inner_h - trace_y0 - OSC_PAD
        cy       = trace_y0 + trace_h // 2

        # centre line
        pygame.draw.line(box, (40, 40, 60),
                         (OSC_PAD, cy), (inner_w - OSC_PAD, cy), 1)

        # ── oscilloscope trace ─────────────────────────────────────────────────
        win_n = default_win   # default, may be updated below
        if music_started and ch.audio is not None and len(ch.audio) > 0:
            if autoscale:
                win_n = adaptive_window_samples(
                    ch.audio, pos_sample, sample_rate, default_samples=default_win)
            else:
                win_n = default_win

            wnd = get_oscilloscope_window(
                ch.audio, pos_sample, win_n, triggered=ch.use_trigger)

            amp = float(np.max(np.abs(wnd))) if len(wnd) > 0 else 1.0
            if amp < 1e-5:
                amp = 1.0

            n_pts = min(len(wnd), inner_w - OSC_PAD * 2)
            if n_pts >= 2:
                xs = np.linspace(OSC_PAD, inner_w - OSC_PAD, n_pts, dtype=np.float32)
                ys = (cy - (wnd[:n_pts] / amp) * (trace_h / 2)).astype(np.float32)
                ys = np.clip(ys, trace_y0, trace_y0 + trace_h - 1)

                pts = list(zip(xs.astype(int).tolist(), ys.astype(int).tolist()))
                r, g, b = _hex_to_rgb(ch.color)
                pygame.draw.aalines(box, (r, g, b, 255), False, pts)

        # ── title chip (opaque dark strip over trace) ──────────────────────────
        pygame.draw.rect(box, (20, 20, 30, 230),
                         pygame.Rect(1, 1, inner_w - 2, CHIP_H), border_radius=3)

        if font_sm:
            r, g, b = _hex_to_rgb(ch.color)
            ms_str = f"{win_n / sample_rate * 1000:.0f}ms"

            # ── LEFT: "FM3 · 8ms" in channel colour ───────────────────────────
            name_ms_surf = font_sm.render(f"{ch.name} · {ms_str}", True, (r, g, b))
            box.blit(name_ms_surf, (OSC_PAD, 2))

            # ── RIGHT: LCD segment row — every slot always present, dim or lit ─
            # Slots are laid out right-to-left so each label is at a fixed pixel
            # column every frame (Game & Watch / calculator display effect).
            SLOT_GAP = 5
            ch_modes = (channel_modes or {}).get(ch.ch_id, {})

            # Build slot list in right-to-left render order:
            #   last item rendered → rightmost on screen
            slots: list[tuple[str, bool]] = []
            # DAC / SPC come first in the list → rendered leftmost in the right block
            if ch.ch_id == 5:  # FM6
                slots.append(("DAC", dac_is_active(vgm_pos)))
            if ch.ch_id == 2:  # FM3
                slots.append(("SPC", ch3_special_is_active(vgm_pos)))
            for key, short in _OSC_MODE_SLOTS:
                if key in ch_modes:
                    slots.append((short, ch_modes[key](vgm_pos)))

            # Render right-to-left: iterate in reverse so rightmost slot lands
            # closest to the right edge.
            rx = inner_w - OSC_PAD
            for label_txt, active in reversed(slots):
                col  = (r, g, b) if active else _LCD_DIM
                surf = font_sm.render(label_txt, True, col)
                rx  -= surf.get_width()
                box.blit(surf, (rx, 2))
                rx  -= SLOT_GAP

        surface.blit(box, (bx + 1, by + 1))


# ── drawing: header bar ────────────────────────────────────────────────────────

def _draw_header(
    surface,
    title:      str,
    t:          float,
    screen_w:   int,
    n_channels: int,
    total_sec:  float,
    composer:   str,
    font_hdr,
    font_mono,
) -> None:
    """Draw the rounded title + timestamp bar at the very top."""
    import pygame

    bar_rect = pygame.Rect(8, 6, screen_w - 16, HEADER_H - 6)

    bar_surf = pygame.Surface((bar_rect.width, bar_rect.height), pygame.SRCALPHA)
    pygame.draw.rect(bar_surf, (15, 15, 25, 210),
                     pygame.Rect(0, 0, bar_rect.width, bar_rect.height),
                     border_radius=8)
    surface.blit(bar_surf, (bar_rect.x, bar_rect.y))

    vy = bar_rect.y + bar_rect.height // 2

    if font_hdr:
        t_mins, t_secs = divmod(total_sec, 60)
        meta  = f"{n_channels}ch  ·  {int(t_mins)}:{int(t_secs):02d}"
        by    = f"  ·  {composer}" if composer else ""
        label = f"VgmLLM  ·  Synthesia + Oscilloscope  ·  {title}{by}  ·  {meta}"
        txt   = font_hdr.render(label, True, (190, 190, 215))
        surface.blit(txt, (bar_rect.x + 10, vy - txt.get_height() // 2))

    if font_mono:
        mins, secs = divmod(max(t, 0.0), 60)
        ts_str  = f"Aubrey Lewman  ·  {int(mins)}:{secs:06.3f}"
        ts_surf = font_mono.render(ts_str, True, (140, 140, 165))
        tx = bar_rect.right - ts_surf.get_width() - 10
        surface.blit(ts_surf, (tx, vy - ts_surf.get_height() // 2))


# ── master render call ─────────────────────────────────────────────────────────

def _render_combined(
    surface,
    rects, t, lookahead,
    screen_w, screen_h, piano_h,
    dac_strip_w,
    pitch_min, pitch_max, white_w, black_w, min_white, piano_x0,
    channels, pos_sample, sample_rate, vgm_pos,
    dac_is_active, ch3_special_is_active,
    window_s, autoscale, cols,
    title, n_channels, total_sec, composer,
    font_hdr, font_mono, font_sm, font_key,
    no_osc: bool = False,
    channel_modes: dict | None = None,
    particles: list | None = None,
    dt: float = 0.0,
    bpm: float = 0.0,
) -> None:
    """Composite one complete frame onto `surface`."""
    top_y = HEADER_H  # the falling zone starts here

    # 1 – synthesia background
    _draw_background(
        surface, rects, t, lookahead,
        screen_w, screen_h, piano_h, top_y,
        dac_strip_w, pitch_min, pitch_max,
        white_w, black_w, min_white, piano_x0, font_key,
        particles=particles, dt=dt, bpm=bpm,
    )

    # 2 – oscilloscope grid overlay (behind header, in front of falling notes)
    if not no_osc and channels:
        grid_x = dac_strip_w + OSC_MARGIN
        grid_y = top_y + OSC_MARGIN
        grid_w = screen_w - grid_x - OSC_MARGIN
        n_rows = math.ceil(len(channels) / cols)
        box_h  = _osc_box_h(n_rows)
        _draw_osc_grid(
            surface, channels, pos_sample, sample_rate,
            grid_x, grid_y, grid_w, cols,
            window_s, autoscale, vgm_pos,
            dac_is_active, ch3_special_is_active, font_sm,
            music_started=(t >= 0),
            box_h=box_h,
            channel_modes=channel_modes,
        )

    # 3 – header on top of everything
    _draw_header(surface, title, t, screen_w, n_channels, total_sec, composer, font_hdr, font_mono)


# ── setup / WAV rendering ──────────────────────────────────────────────────────

def _build_looped_wav(src: Path, t_loop_start: float, extra_sec: float, out: Path) -> None:
    """Append the first `extra_sec` seconds of the loop body to `src`, writing `out`."""
    import wave as _wave
    with _wave.open(str(src), "rb") as wf:
        sr      = wf.getframerate()
        n_ch    = wf.getnchannels()
        sw      = wf.getsampwidth()
        n_tot   = wf.getnframes()
        raw_all = wf.readframes(n_tot)

    frame_size      = n_ch * sw
    loop_start_fr   = int(t_loop_start * sr)
    extra_fr        = int(extra_sec * sr)
    loop_slice_fr   = min(extra_fr, n_tot - loop_start_fr)   # don't go past EOF
    loop_slice_fr   = max(loop_slice_fr, 0)

    loop_bytes = raw_all[loop_start_fr * frame_size :
                         (loop_start_fr + loop_slice_fr) * frame_size]

    with _wave.open(str(out), "wb") as wf:
        wf.setnchannels(n_ch)
        wf.setsampwidth(sw)
        wf.setframerate(sr)
        wf.writeframes(raw_all + loop_bytes)


def _prepare(args, vgm_path: Path):
    """
    Parse VGM, render per-channel WAVs (cached), then initialise pygame fonts.
    Returns a tuple of everything needed for rendering.
    """
    # ── 1. Parse notes (no pygame needed) ─────────────────────────────────────
    print(f"Loading {vgm_path.name} …")
    rects, total_sec, t_loop_start, bpm = _parse_notes(vgm_path)
    print(f"  {len(rects)} note events, {total_sec:.1f}s  loop@{t_loop_start:.1f}s  ~{bpm:.1f} BPM")

    pitched = [r for r in rects if r.pitch >= 0 and r.channel != CH_DAC]
    if not pitched:
        sys.exit("No pitched notes found — cannot build piano keyboard.")

    pitch_min  = (min(r.pitch for r in pitched) // 12) * 12
    pitch_max  = ((max(r.pitch for r in pitched) // 12) + 1) * 12 - 1
    min_white  = _white_index(pitch_min)
    max_white  = _white_index(pitch_max)
    n_whites   = max_white - min_white + 1

    DAC_STRIP_W   = 100
    piano_pixel_w = args.width - DAC_STRIP_W
    white_w       = piano_pixel_w / n_whites
    black_w       = white_w * 0.60
    piano_x0      = float(DAC_STRIP_W)

    # ── 2. Render WAVs (no pygame needed) ─────────────────────────────────────
    wav_dir   = _ROOT / "output" / "roundtrip" / (vgm_path.stem + "_wavs")
    mix_wav: Optional[Path] = None

    try:
        renderer = VGMPlayRenderer(args.vgmplay_dir)
        mix_wav  = renderer.render_mix(vgm_path, wav_dir)
    except FileNotFoundError as e:
        print(f"  Warning: {e} — running without audio / per-channel WAVs")
        renderer = None

    active_ids, dac_ranges, ch3_ranges, per_ch_modes = build_vgm_info(vgm_path)
    dac_is_active         = _make_is_active(dac_ranges)
    ch3_special_is_active = _make_is_active(ch3_ranges)
    channel_modes = {
        ch_id: {mode: _make_is_active(ranges) for mode, ranges in modes.items()}
        for ch_id, modes in per_ch_modes.items()
    }

    channels: list[Channel] = []
    if renderer is not None:
        all_chs = _make_channel_table()
        for ch in all_chs:
            if not args.all_channels and ch.ch_id not in active_ids:
                continue
            wav_path = renderer.render_channel(vgm_path, ch, wav_dir)
            if wav_path is None:
                continue
            audio, _rate = load_wav_mono(wav_path)
            if args.all_channels or is_active(audio):
                ch.wav_path = wav_path
                ch.audio    = audio
                channels.append(ch)

    if not channels:
        print("  Warning: no oscilloscope channels loaded; --no-osc implied")

    print(f"  {len(channels)} osc channels: {[c.name for c in channels]}")

    # Infer sample rate from mix wav
    sample_rate = AUDIO_RATE
    if mix_wav and mix_wav.exists():
        _, sample_rate = load_wav_mono(mix_wav)

    # ── 2b. Phantom notes + looped mix WAV (seamless loop visuals & audio) ────
    mix_looped_wav: Optional[Path] = None
    if t_loop_start > 0.0 and total_sec > t_loop_start:
        loop_dur = total_sec - t_loop_start
        phantom  = [
            NoteRect(r.channel, r.pitch, r.t_on + loop_dur, r.t_off + loop_dur, r.color)
            for r in rects if r.t_on >= t_loop_start
        ]
        rects = rects + phantom
        print(f"  {len(phantom)} phantom notes added for seamless loop visuals")

        if mix_wav and mix_wav.exists():
            # Full loop body + tail — enough for MP4 export and interactive
            extra_sec      = loop_dur + args.lookahead + 1.0
            mix_looped_wav = wav_dir / (vgm_path.stem + "_mix_1loop.wav")
            if not mix_looped_wav.exists():
                _build_looped_wav(mix_wav, t_loop_start, extra_sec, mix_looped_wav)
                print(f"  Built looped mix WAV: {mix_looped_wav.name}")
            else:
                print(f"  [cached] {mix_looped_wav.name}")

    # ── 3. GD3 metadata (title + composer) ─────────────────────────────────────
    from genesis_music.vgm_parser import load_vgm as _load_vgm
    try:
        _vgm = _load_vgm(vgm_path)
        _gd3 = getattr(_vgm, "gd3", None)
        composer    = (_gd3.author_en.strip()     if _gd3 and _gd3.author_en.strip()     else "")
        track_title = (_gd3.track_name_en.strip() if _gd3 and _gd3.track_name_en.strip() else vgm_path.stem)
    except Exception:
        composer    = ""
        track_title = vgm_path.stem
    if composer:
        print(f"  Composer: {composer}")

    # ── 4. Init pygame + fonts ─────────────────────────────────────────────────
    import pygame
    pygame.init()
    font_hdr  = pygame.font.SysFont("segoeui",   14)
    font_mono = pygame.font.SysFont("consolas",  13)
    font_sm   = pygame.font.SysFont("segoeui",   10)
    font_key  = pygame.font.SysFont("monospace", 10)

    return (
        rects, total_sec, t_loop_start, bpm,
        pitch_min, pitch_max, min_white, n_whites,
        white_w, black_w, piano_x0, DAC_STRIP_W,
        channels, mix_wav, mix_looped_wav, sample_rate,
        dac_is_active, ch3_special_is_active,
        composer, track_title,
        font_hdr, font_mono, font_sm, font_key,
        channel_modes,
    )


# ── interactive mode ───────────────────────────────────────────────────────────

def _interactive(
    args, vgm_path,
    rects, total_sec, t_loop_start, bpm,
    pitch_min, pitch_max, min_white, n_whites,
    white_w, black_w, piano_x0, dac_strip_w,
    channels, mix_wav, mix_looped_wav, sample_rate,
    dac_is_active, ch3_special_is_active,
    composer, track_title,
    font_hdr, font_mono, font_sm, font_key,
    channel_modes=None,
):
    import pygame

    screen = pygame.display.set_mode((args.width, args.height))
    pygame.display.set_caption(f"{APP_NAME} — {track_title}")
    clock = pygame.time.Clock()

    has_audio = mix_wav is not None and mix_wav.exists()
    if has_audio:
        pygame.mixer.init(frequency=sample_rate, size=-16, channels=2, buffer=2048)
        pygame.mixer.music.load(str(mix_wav))

    window_s  = args.window
    autoscale = not args.no_autoscale
    title     = track_title

    t         = -args.lookahead / 2   # start with notes already halfway down
    t_end     = total_sec + 1.0
    paused    = False

    has_loop = t_loop_start > 0.0

    def _start_audio_at(pos_sec: float) -> None:
        if not has_audio:
            return
        pygame.mixer.music.stop()
        if pos_sec >= 0:
            pygame.mixer.music.play(start=pos_sec)

    _start_audio_at(t)
    audio_started = t >= 0

    if has_loop:
        print(f"Loop point: {t_loop_start:.2f}s (loop body {total_sec - t_loop_start:.2f}s)")
    print("Controls: SPACE=pause  LEFT/RIGHT=seek 5s  Q/ESC=quit")

    particles: list = []   # spark particles for Idea 3
    spf = 1.0 / args.fps

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); return
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    pygame.quit(); return
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    if has_audio:
                        if paused:
                            pygame.mixer.music.pause()
                        else:
                            pygame.mixer.music.unpause()
                elif event.key == pygame.K_RIGHT:
                    t = min(t + 5.0, t_end)
                    _start_audio_at(max(t, 0))
                    audio_started = t >= 0
                elif event.key == pygame.K_LEFT:
                    t = max(t - 5.0, -args.lookahead)
                    _start_audio_at(max(t, 0))
                    audio_started = t >= 0

        if not paused:
            if has_audio and audio_started:
                pos_ms = pygame.mixer.music.get_pos()
                if pos_ms >= 0:
                    t = pos_ms / 1000.0
                else:
                    # audio finished — advance by clock so loop condition triggers
                    dt = clock.tick(args.fps) / 1000.0
                    t += dt
            else:
                dt = clock.tick(args.fps) / 1000.0
                t += dt
                if has_audio and not audio_started and t >= 0:
                    pygame.mixer.music.play(start=0)
                    audio_started = True
                    t = 0.0
        else:
            clock.tick(30)

        if t >= t_end:
            # Seamless loop: jump to the VGM loop point (not back to pre-roll)
            t = t_loop_start
            _start_audio_at(t_loop_start)
            audio_started = True

        pos_sample = int(max(t, 0.0) * sample_rate)
        vgm_pos    = int(pos_sample * VGM_RATE / sample_rate)

        _render_combined(
            screen, rects, t, args.lookahead,
            args.width, args.height, args.piano_height,
            dac_strip_w, pitch_min, pitch_max, white_w, black_w, min_white, piano_x0,
            channels, pos_sample, sample_rate, vgm_pos,
            dac_is_active, ch3_special_is_active,
            window_s, autoscale, _osc_cols(len(channels)), title, len(channels), total_sec, composer,
            font_hdr, font_mono, font_sm, font_key,
            no_osc=args.no_osc,
            channel_modes=channel_modes,
            particles=particles,
            dt=spf,
            bpm=bpm,
        )

        pygame.display.flip()
        if not (has_audio and audio_started and not paused):
            clock.tick(args.fps)


# ── MP4 encoder helpers ──────────────────────────────────────────────────────────────────────────────

def _probe_amf(ffmpeg_bin: str) -> bool:
    """Return True if the h264_amf encoder is usable on this system."""
    try:
        r = subprocess.run(
            [ffmpeg_bin, "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", "color=black:s=128x128:r=30:d=0.1",
             "-vf", "format=yuv420p",
             "-frames:v", "1", "-c:v", "h264_amf", "-f", "null", "-"],
            capture_output=True, timeout=10,
        )
        return r.returncode == 0
    except Exception:
        return False


def _video_enc_args(args, ffmpeg_bin: str) -> list[str]:
    """Return the ffmpeg video encoder arguments based on --amf flag."""
    if getattr(args, 'amf', False):
        print("  Probing h264_amf (AMD APU hardware encoder) …", end=" ", flush=True)
        if _probe_amf(ffmpeg_bin):
            print("available ✓  (encoding on Radeon APU)")
            return [
                "-vcodec",  "h264_amf",
                "-pix_fmt", "yuv420p",
                "-quality", "quality",
                "-rc",      "cqp",
                "-qp_i",    "18",
                "-qp_p",    "20",
            ]
        print("not available — falling back to libx264")
    return [
        "-vcodec",  "libx264",
        "-pix_fmt", "yuv420p",
        "-crf",     "18",
        "-preset",  "fast",
    ]


# ── MP4 export mode ────────────────────────────────────────────────────────────

def _export_mp4(
    args, vgm_path,
    rects, total_sec, t_loop_start, bpm,
    pitch_min, pitch_max, min_white, n_whites,
    white_w, black_w, piano_x0, dac_strip_w,
    channels, mix_wav, mix_looped_wav, sample_rate,
    dac_is_active, ch3_special_is_active,
    composer, track_title,
    font_hdr, font_mono, font_sm, font_key,
    channel_modes=None,
):
    import pygame

    out_path = Path(args.mp4)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    surface      = pygame.Surface((args.width, args.height))
    spf          = 1.0 / args.fps
    # Render intro (half-lookahead pre-roll) + full first loop + one extra loop pass + tail
    loop_dur     = (total_sec - t_loop_start) if t_loop_start > 0.0 else 0.0
    total_frames = int((total_sec + loop_dur + args.lookahead / 2) * args.fps)
    window_s     = args.window
    autoscale    = not args.no_autoscale
    title        = track_title
    # Prefer looped WAV for gapless audio; fall back to original mix
    audio_wav = (mix_looped_wav if (mix_looped_wav and mix_looped_wav.exists())
                 else mix_wav)
    has_mix   = audio_wav is not None and audio_wav.exists()

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        sys.exit("ffmpeg not found in PATH — cannot export MP4.")

    ffmpeg_cmd = [
        ffmpeg, "-y",
        "-f",      "rawvideo",
        "-vcodec", "rawvideo",
        "-s",      f"{args.width}x{args.height}",
        "-pix_fmt","rgb24",
        "-r",      str(args.fps),
        "-i",      "pipe:0",
    ]
    if has_mix:
        # Delay audio by half-lookahead so it starts when notes reach the piano
        ffmpeg_cmd += ["-itsoffset", f"{args.lookahead / 2:.6f}", "-i", str(audio_wav)]
    ffmpeg_cmd += _video_enc_args(args, ffmpeg)
    if has_mix:
        ffmpeg_cmd += ["-acodec", "aac", "-b:a", "192k", "-shortest"]
    ffmpeg_cmd.append(str(out_path))

    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)
    t0 = _time.time()
    print(f"Exporting {total_frames} frames @ {args.fps}fps → {out_path}")

    particles: list = []   # spark particles for Idea 3

    try:
        for frame_i in range(total_frames):
            t          = -args.lookahead / 2 + frame_i * spf
            pos_sample = int(max(t, 0.0) * sample_rate)
            vgm_pos    = int(pos_sample * VGM_RATE / sample_rate)

            _render_combined(
                surface, rects, t, args.lookahead,
                args.width, args.height, args.piano_height,
                dac_strip_w, pitch_min, pitch_max, white_w, black_w, min_white, piano_x0,
                channels, pos_sample, sample_rate, vgm_pos,
                dac_is_active, ch3_special_is_active,
                window_s, autoscale, _osc_cols(len(channels)), title, len(channels), total_sec, composer,
                font_hdr, font_mono, font_sm, font_key,
                no_osc=args.no_osc,
                channel_modes=channel_modes,
                particles=particles,
                dt=spf,
                bpm=bpm,
            )

            raw = pygame.surfarray.array3d(surface)
            proc.stdin.write(raw.transpose(1, 0, 2).tobytes())

            if frame_i % (args.fps * 5) == 0:
                elapsed   = _time.time() - t0
                pct       = frame_i / total_frames * 100
                remaining = (elapsed / max(frame_i, 1)) * (total_frames - frame_i)
                print(f"  {pct:5.1f}%  t={max(t,0):.1f}s/{total_sec:.1f}s  "
                      f"ETA {remaining:.0f}s", end="\r", flush=True)

    finally:
        proc.stdin.close()
        proc.wait()

    elapsed = _time.time() - t0
    print(f"\n  Done → {out_path}  ({elapsed:.1f}s)")
    pygame.quit()


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combined Synthesia + Oscilloscope visualizer for VGM files."
    )
    parser.add_argument("vgm",
                        help="Path to .vgm or .vgz file")
    parser.add_argument("--mp4",          metavar="PATH",
                        help="Export to MP4 instead of interactive view")
    parser.add_argument("--fps",          type=int,   default=60)
    parser.add_argument("--width",        type=int,   default=1280)
    parser.add_argument("--height",       type=int,   default=720)
    parser.add_argument("--lookahead",    type=float, default=4.0,
                        help="Seconds of notes visible above keyboard (default 4)")
    parser.add_argument("--piano-height", type=int,   default=140, dest="piano_height",
                        help="Piano keyboard height in pixels (default 140)")
    parser.add_argument("--window",       type=float, default=0.08, metavar="SEC",
                        help="Oscilloscope window width in seconds (default 0.08)")
    parser.add_argument("--all-channels", action="store_true", dest="all_channels",
                        help="Include apparently inactive channels in oscilloscope grid")
    parser.add_argument("--no-autoscale", action="store_true", dest="no_autoscale",
                        help="Disable per-channel amplitude normalization in oscilloscope")
    parser.add_argument("--no-osc",       action="store_true", dest="no_osc",
                        help="Disable oscilloscope overlay (Synthesia background only)")
    parser.add_argument("--amf",          action="store_true",
                        help="Use AMD AMF hardware encoder (h264_amf) on the APU; "
                             "auto-falls back to libx264 if unavailable")
    parser.add_argument("--vgmplay-dir",  type=Path, default=_DEFAULT_VGMPLAY_DIR,
                        dest="vgmplay_dir",
                        help=f"VGMPlay directory (default: {_DEFAULT_VGMPLAY_DIR})")
    args = parser.parse_args()

    vgm_path = Path(args.vgm).resolve()
    if not vgm_path.exists():
        sys.exit(f"File not found: {vgm_path}")

    (
        rects, total_sec, t_loop_start, bpm,
        pitch_min, pitch_max, min_white, n_whites,
        white_w, black_w, piano_x0, dac_strip_w,
        channels, mix_wav, mix_looped_wav, sample_rate,
        dac_is_active, ch3_special_is_active,
        composer, track_title,
        font_hdr, font_mono, font_sm, font_key,
        channel_modes,
    ) = _prepare(args, vgm_path)

    if args.mp4:
        _export_mp4(
            args, vgm_path,
            rects, total_sec, t_loop_start, bpm,
            pitch_min, pitch_max, min_white, n_whites,
            white_w, black_w, piano_x0, dac_strip_w,
            channels, mix_wav, mix_looped_wav, sample_rate,
            dac_is_active, ch3_special_is_active,
            composer, track_title,
            font_hdr, font_mono, font_sm, font_key,
            channel_modes=channel_modes,
        )
    else:
        _interactive(
            args, vgm_path,
            rects, total_sec, t_loop_start, bpm,
            pitch_min, pitch_max, min_white, n_whites,
            white_w, black_w, piano_x0, dac_strip_w,
            channels, mix_wav, mix_looped_wav, sample_rate,
            dac_is_active, ch3_special_is_active,
            composer, track_title,
            font_hdr, font_mono, font_sm, font_key,
            channel_modes=channel_modes,
        )


if __name__ == "__main__":
    main()
