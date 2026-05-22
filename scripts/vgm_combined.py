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

# Box height per column tier: wider grid (fewer cols) → taller boxes
_BOX_H_FOR_COLS = {2: 160, 3: 130, 4: 110}

APP_NAME   = "VgmLLM"


def _osc_cols(n_channels: int) -> int:
    """Dynamic column count: ≤6 → 2 cols, 7-9 → 3 cols, >9 → 4 cols."""
    if n_channels <= 6:
        return 2
    elif n_channels <= 9:
        return 3
    else:
        return 4


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

    # ── falling note bars ──────────────────────────────────────────────────────
    for nr in rects:
        if nr.t_off < t_vis_start or nr.t_on > t_vis_end:
            continue

        clipped_on  = max(nr.t_on,  t_vis_start)
        clipped_off = min(nr.t_off, t_vis_end)

        y_bottom = _time_to_y(clipped_on)
        y_top    = _time_to_y(clipped_off)
        bar_h    = max(y_bottom - y_top, 2)

        r, g, b  = nr.color

        if nr.pitch >= 0 and nr.channel != CH_DAC:
            rel_x, key_w, _ = _note_key_info(nr.pitch, min_white, white_w, black_w)
            x = piano_x0 + rel_x
            w = max(key_w - 2, 2)
            pygame.draw.rect(surface, (r, g, b),
                             pygame.Rect(int(x), int(y_top), int(w), int(bar_h)),
                             border_radius=3)
            if bar_h > 4:
                pygame.draw.rect(surface,
                                 (min(r+80, 255), min(g+80, 255), min(b+80, 255)),
                                 pygame.Rect(int(x), int(y_top), int(w), 3),
                                 border_radius=3)
        else:
            # DAC / noise → left strip
            strip_slot = ([CH_DAC, CH_PSG_NOISE].index(nr.channel)
                          if nr.channel in (CH_DAC, CH_PSG_NOISE) else 0)
            slot_w = dac_strip_w // 2
            x      = strip_slot * slot_w
            w      = slot_w - 2
            pygame.draw.rect(surface, (r, g, b),
                             pygame.Rect(int(x), int(y_top), int(w), int(bar_h)),
                             border_radius=2)

    # ── piano keyboard ─────────────────────────────────────────────────────────
    pygame.draw.rect(surface, PIANO_BG_COLOR,
                     pygame.Rect(0, piano_y, screen_w, piano_h))
    pygame.draw.line(surface, (80, 80, 130),
                     (0, piano_y), (screen_w, piano_y), 2)

    active_pitches: set[int]    = set()
    active_ch:      dict[int, int] = {}
    for nr in rects:
        if nr.t_on <= t < nr.t_off and nr.pitch >= 0:
            active_pitches.add(nr.pitch)
            active_ch[nr.pitch] = nr.channel

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
            base = CHANNEL_COLORS.get(active_ch[midi], WHITE_KEY_CLR)
            color = tuple(min(c + 60, 255) for c in base)
        else:
            color = WHITE_KEY_CLR
        pygame.draw.rect(surface, color,
                         pygame.Rect(x, piano_y + 2, w, white_key_h - 2),
                         border_radius=2)
        pygame.draw.rect(surface, (80, 80, 100),
                         pygame.Rect(x, piano_y + 2, w, white_key_h - 2),
                         width=1, border_radius=2)

    # Black keys
    for wi in range(n_whites):
        midi_white = _white_key_midi(wi + min_white)
        if midi_white is None:
            continue
        midi_black = midi_white + 1
        if midi_black <= pitch_max and not _NOTE_IS_WHITE[midi_black % 12]:
            if midi_black in active_pitches:
                base  = CHANNEL_COLORS.get(active_ch[midi_black], BLACK_KEY_CLR)
                color = tuple(min(c + 40, 255) for c in base)
            else:
                color = BLACK_KEY_CLR
            bx = int(piano_x0 + wi * white_w + white_w * 0.55)
            bw = int(black_w)
            pygame.draw.rect(surface, color,
                             pygame.Rect(bx, piano_y + 2, bw, black_key_h),
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

        label = _dynamic_title(ch, vgm_pos, dac_is_active, ch3_special_is_active)
        if font_sm:
            # left: window width in ms (muted grey)
            ms_str  = f"{win_n / sample_rate * 1000:.0f}ms"
            ms_surf = font_sm.render(ms_str, True, (90, 90, 120))
            box.blit(ms_surf, (OSC_PAD, 2))
            # right: channel name in channel colour
            r, g, b = _hex_to_rgb(ch.color)
            txt = font_sm.render(label, True, (r, g, b))
            box.blit(txt, (inner_w - txt.get_width() - OSC_PAD, 2))

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
) -> None:
    """Composite one complete frame onto `surface`."""
    top_y = HEADER_H  # the falling zone starts here

    # 1 – synthesia background
    _draw_background(
        surface, rects, t, lookahead,
        screen_w, screen_h, piano_h, top_y,
        dac_strip_w, pitch_min, pitch_max,
        white_w, black_w, min_white, piano_x0, font_key,
    )

    # 2 – oscilloscope grid overlay (behind header, in front of falling notes)
    if not no_osc and channels:
        grid_x = dac_strip_w + OSC_MARGIN
        grid_y = top_y + OSC_MARGIN
        grid_w = screen_w - grid_x - OSC_MARGIN
        box_h  = _BOX_H_FOR_COLS.get(cols, OSC_BOX_H)
        _draw_osc_grid(
            surface, channels, pos_sample, sample_rate,
            grid_x, grid_y, grid_w, cols,
            window_s, autoscale, vgm_pos,
            dac_is_active, ch3_special_is_active, font_sm,
            music_started=(t >= 0),
            box_h=box_h,
        )

    # 3 – header on top of everything
    _draw_header(surface, title, t, screen_w, n_channels, total_sec, composer, font_hdr, font_mono)


# ── setup / WAV rendering ──────────────────────────────────────────────────────

def _build_looped_wav(src: Path, t_loop_start: float, extra_sec: float, out: Path) -> None:
    """Append the first `extra_sec` seconds of the loop body to `src`, writing `out`.

    Works at raw-bytes level (no re-encode, preserves original sample format).
    Used to give the MP4 export gapless audio through the phantom-note tail.
    """
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
    rects, total_sec, t_loop_start = _parse_notes(vgm_path)
    print(f"  {len(rects)} note events, {total_sec:.1f}s  loop@{t_loop_start:.1f}s")

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

    active_ids, dac_ranges, ch3_ranges = build_vgm_info(vgm_path)
    dac_is_active         = _make_is_active(dac_ranges)
    ch3_special_is_active = _make_is_active(ch3_ranges)

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
        rects, total_sec, t_loop_start,
        pitch_min, pitch_max, min_white, n_whites,
        white_w, black_w, piano_x0, DAC_STRIP_W,
        channels, mix_wav, mix_looped_wav, sample_rate,
        dac_is_active, ch3_special_is_active,
        composer, track_title,
        font_hdr, font_mono, font_sm, font_key,
    )


# ── interactive mode ───────────────────────────────────────────────────────────

def _interactive(
    args, vgm_path,
    rects, total_sec, t_loop_start,
    pitch_min, pitch_max, min_white, n_whites,
    white_w, black_w, piano_x0, dac_strip_w,
    channels, mix_wav, mix_looped_wav, sample_rate,
    dac_is_active, ch3_special_is_active,
    composer, track_title,
    font_hdr, font_mono, font_sm, font_key,
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

    t         = -args.lookahead
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
        )

        pygame.display.flip()
        if not (has_audio and audio_started and not paused):
            clock.tick(args.fps)


# ── MP4 export mode ────────────────────────────────────────────────────────────

def _export_mp4(
    args, vgm_path,
    rects, total_sec, t_loop_start,
    pitch_min, pitch_max, min_white, n_whites,
    white_w, black_w, piano_x0, dac_strip_w,
    channels, mix_wav, mix_looped_wav, sample_rate,
    dac_is_active, ch3_special_is_active,
    composer, track_title,
    font_hdr, font_mono, font_sm, font_key,
):
    import pygame

    out_path = Path(args.mp4)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    surface      = pygame.Surface((args.width, args.height))
    spf          = 1.0 / args.fps
    # Render intro + full first loop + one extra loop pass + tail
    loop_dur     = (total_sec - t_loop_start) if t_loop_start > 0.0 else 0.0
    total_frames = int((total_sec + loop_dur + args.lookahead) * args.fps)
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
        # Delay audio by lookahead so it starts when notes reach the piano
        ffmpeg_cmd += ["-itsoffset", f"{args.lookahead:.6f}", "-i", str(audio_wav)]
    ffmpeg_cmd += [
        "-vcodec",  "libx264",
        "-pix_fmt", "yuv420p",
        "-crf",     "18",
        "-preset",  "fast",
    ]
    if has_mix:
        ffmpeg_cmd += ["-acodec", "aac", "-b:a", "192k", "-shortest"]
    ffmpeg_cmd.append(str(out_path))

    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)
    t0 = _time.time()
    print(f"Exporting {total_frames} frames @ {args.fps}fps → {out_path}")

    try:
        for frame_i in range(total_frames):
            t          = -args.lookahead + frame_i * spf
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
    parser.add_argument("--vgmplay-dir",  type=Path, default=_DEFAULT_VGMPLAY_DIR,
                        dest="vgmplay_dir",
                        help=f"VGMPlay directory (default: {_DEFAULT_VGMPLAY_DIR})")
    args = parser.parse_args()

    vgm_path = Path(args.vgm).resolve()
    if not vgm_path.exists():
        sys.exit(f"File not found: {vgm_path}")

    (
        rects, total_sec, t_loop_start,
        pitch_min, pitch_max, min_white, n_whites,
        white_w, black_w, piano_x0, dac_strip_w,
        channels, mix_wav, mix_looped_wav, sample_rate,
        dac_is_active, ch3_special_is_active,
        composer, track_title,
        font_hdr, font_mono, font_sm, font_key,
    ) = _prepare(args, vgm_path)

    if args.mp4:
        _export_mp4(
            args, vgm_path,
            rects, total_sec, t_loop_start,
            pitch_min, pitch_max, min_white, n_whites,
            white_w, black_w, piano_x0, dac_strip_w,
            channels, mix_wav, mix_looped_wav, sample_rate,
            dac_is_active, ch3_special_is_active,
            composer, track_title,
            font_hdr, font_mono, font_sm, font_key,
        )
    else:
        _interactive(
            args, vgm_path,
            rects, total_sec, t_loop_start,
            pitch_min, pitch_max, min_white, n_whites,
            white_w, black_w, piano_x0, dac_strip_w,
            channels, mix_wav, mix_looped_wav, sample_rate,
            dac_is_active, ch3_special_is_active,
            composer, track_title,
            font_hdr, font_mono, font_sm, font_key,
        )


if __name__ == "__main__":
    main()
