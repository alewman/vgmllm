#!/usr/bin/env python3
"""vgm_synthesia.py — Synthesia-style falling-notes visualizer for VGM files.

Notes fall as coloured bars from the top of the screen toward a piano keyboard
at the bottom.  When a bar reaches the keyboard the corresponding key lights up.
DAC events appear as a narrow strip to the left of the piano, also falling.

Modes
-----
  (default)    Interactive pygame window with audio (requires VGMPlay)
  --mp4 PATH   Render every frame and mux into an MP4 via ffmpeg (no audio)

Usage
-----
  cd d:\\dev\\genesis-music-ml
  python scripts/vgm_synthesia.py data/vgm/some_song.vgz
  python scripts/vgm_synthesia.py data/vgm/some_song.vgz --mp4 output/roll.mp4
  python scripts/vgm_synthesia.py data/vgm/some_song.vgz --lookahead 6 --width 1280 --height 720
  python scripts/vgm_synthesia.py data/vgm/some_song.vgz --vgmplay-dir D:/dev/VGMPlay_040-9
"""

from __future__ import annotations

import argparse
import gzip
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "src"))

from genesis_music.vgm_parser import load_vgm
from genesis_music.ym2612 import (
    CH_DAC, CH_FM_0, CH_PSG_0, CH_PSG_1, CH_PSG_2, CH_PSG_NOISE,
    decode_vgm,
)

VGM_RATE    = 44_100
AUDIO_RATE  = 48_000   # VGMPlay renders at 48 kHz
_DEFAULT_VGMPLAY_DIR = _ROOT.parent / "VGMPlay_040-9"

# ── channel colours (match oscilloscope) ─────────────────────────────────────

CHANNEL_COLORS: dict[int, tuple[int, int, int]] = {
    0:            (255,  85,  51),   # FM1  red-orange
    1:            (255, 204,   0),   # FM2  yellow
    2:            ( 68, 221,  68),   # FM3  green
    3:            ( 34, 204, 221),   # FM4  cyan
    4:            ( 68, 136, 255),   # FM5  blue
    5:            (204,  68, 255),   # FM6  purple
    CH_DAC:       (255,  68, 170),   # DAC  pink
    CH_PSG_0:     (255, 224, 178),   # PSG1 peach
    CH_PSG_1:     (179, 229, 252),   # PSG2 light blue
    CH_PSG_2:     (200, 230, 201),   # PSG3 mint
    CH_PSG_NOISE: (144, 164, 174),   # Noise blue-grey
}

CHANNEL_NAMES: dict[int, str] = {
    0: "FM1", 1: "FM2", 2: "FM3", 3: "FM4", 4: "FM5", 5: "FM6",
    CH_DAC: "DAC", CH_PSG_0: "PSG1", CH_PSG_1: "PSG2",
    CH_PSG_2: "PSG3", CH_PSG_NOISE: "Noise",
}

BG_COLOR       = ( 18,  18,  35)
PIANO_BG_COLOR = ( 28,  28,  48)
WHITE_KEY_CLR  = (230, 230, 230)
BLACK_KEY_CLR  = ( 30,  30,  40)
DIVIDER_COLOR  = ( 50,  50,  80)

# piano layout helpers
_NOTE_IS_WHITE = [True,False,True,False,True,True,False,True,False,True,False,True]
#                  C    C#   D   D#    E   F   F#   G   G#   A   A#   B

def _white_index(midi: int) -> int:
    """Global index of the white key at-or-left-of `midi`."""
    octave   = midi // 12
    semitone = midi % 12
    return octave * 7 + sum(1 for i in range(semitone) if _NOTE_IS_WHITE[i])

def _note_key_info(midi: int, min_white: int, white_w: float, black_w: float
                   ) -> tuple[float, float, bool]:
    """Return (x_left, width, is_white) in pixel units relative to piano origin."""
    is_white = _NOTE_IS_WHITE[midi % 12]
    wi       = _white_index(midi) - min_white
    if is_white:
        return wi * white_w, white_w - 1, True
    else:
        # _white_index for a black key returns the NEXT white key's index,
        # so step back by 1 to align with the left neighbour (matches piano drawing)
        return (wi - 1) * white_w + white_w * 0.55, black_w, False


@dataclass
class NoteRect:
    channel: int
    pitch:   int            # -1 for DAC
    t_on:    float          # seconds
    t_off:   float          # seconds
    color:   tuple[int,int,int]


# ── VGMPlay mix renderer (minimal — full-mix only) ───────────────────────────

class _MixRenderer:
    """Renders a full-mix WAV via VGMPlay for audio playback."""

    def __init__(self, vgmplay_dir: Path):
        self.exe = vgmplay_dir / "VGMPlay.exe"
        self.ini = vgmplay_dir / "VGMPlay.ini"
        if not self.exe.exists():
            raise FileNotFoundError(f"VGMPlay.exe not found: {self.exe}")

    def _patch_ini(self, original: str) -> str:
        # If ini is empty, just create the minimal [General] block
        if not original.strip():
            return "[General]\nLogSound = 1\nSndOut = -1\n"
        lines = original.splitlines()
        result: list[str] = []
        current_section: Optional[str] = None
        log_patched = False
        sndout_patched = False
        for line in lines:
            stripped = line.strip()
            m = re.match(r"^\[(\w+)\]$", stripped)
            if m:
                current_section = m.group(1)
                result.append(line)
                continue
            if current_section == "General" and re.match(r"LogSound\s*=", stripped):
                result.append("LogSound = 1")
                log_patched = True
                continue
            if current_section == "General" and re.match(r"SndOut\s*=", stripped):
                result.append("SndOut = -1")
                sndout_patched = True
                continue
            result.append(line)
        if not log_patched or not sndout_patched:
            final: list[str] = []
            for line in result:
                final.append(line)
                if line.strip() == "[General]":
                    if not log_patched:
                        final.append("LogSound = 1")
                    if not sndout_patched:
                        final.append("SndOut = -1")
            result = final
        return "\n".join(result)

    def render_mix(self, vgm_path: Path, out_dir: Path) -> Optional[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_wav = out_dir / f"{vgm_path.stem}_mix.wav"
        if out_wav.exists():
            print(f"  [cached] {out_wav.name}")
            return out_wav
        print("  Rendering audio mix via VGMPlay...", end=" ", flush=True)
        t0 = time.time()

        # VGMPlay outputs WAV next to the input file; .vgz must be decompressed first
        tmp_dir: Optional[tempfile.TemporaryDirectory] = None
        work_path = vgm_path
        if vgm_path.suffix.lower() == ".vgz":
            tmp_dir = tempfile.TemporaryDirectory()
            work_path = Path(tmp_dir.name) / (vgm_path.stem + ".vgm")
            with gzip.open(vgm_path, "rb") as f_in, open(work_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        original_ini = self.ini.read_bytes()
        try:
            patched = self._patch_ini(original_ini.decode("utf-8", errors="replace"))
            self.ini.write_text(patched, encoding="utf-8")
            subprocess.run([str(self.exe), str(work_path)],
                           cwd=str(self.exe.parent), capture_output=True, timeout=600)
        finally:
            self.ini.write_bytes(original_ini)

        expected_wav = work_path.parent / (work_path.stem + ".wav")
        if expected_wav.exists():
            shutil.move(str(expected_wav), str(out_wav))
            if tmp_dir:
                tmp_dir.cleanup()
            print(f"{time.time()-t0:.1f}s")
            return out_wav

        if tmp_dir:
            tmp_dir.cleanup()
        print("FAILED (no audio)")
        return None


def _parse_notes(vgm_path: Path) -> tuple[list[NoteRect], float]:
    vgm   = load_vgm(vgm_path)
    notes, _ = decode_vgm(vgm)
    total_sec = vgm.header.total_samples / VGM_RATE

    rects: list[NoteRect] = []
    for n in notes:
        t_on  = n.sample_on  / VGM_RATE
        t_off = n.sample_off / VGM_RATE if n.sample_off >= 0 else t_on + 0.08
        # clamp duration
        t_off = max(t_off, t_on + 0.04)
        color = CHANNEL_COLORS.get(n.channel, (200, 200, 200))
        rects.append(NoteRect(n.channel, n.pitch, t_on, t_off, color))

    return rects, total_sec


def _render_frame(
    surface,          # pygame.Surface
    rects:     list[NoteRect],
    t:         float,           # current playback time (seconds)
    lookahead: float,           # seconds visible above piano
    screen_w:  int,
    screen_h:  int,
    piano_h:   int,             # height of keyboard area
    dac_strip_w: int,           # width of DAC / noise strip on the left
    pitch_min: int,
    pitch_max: int,
    white_w:   float,
    black_w:   float,
    min_white: int,
    piano_x0:  float,           # left edge of piano in screen coords
    font,
) -> None:
    import pygame

    fall_h      = screen_h - piano_h   # pixel height of falling zone
    px_per_sec  = fall_h / lookahead   # pixels per second of lookahead

    surface.fill(BG_COLOR)

    # ── draw falling note bars ────────────────────────────────────────────────

    t_visible_start = t              # oldest visible time (bottom of fall zone)
    t_visible_end   = t + lookahead  # newest visible time (top of fall zone)

    # pre-split into pitched vs dac/noise for separate regions
    for nr in rects:
        if nr.t_off < t_visible_start or nr.t_on > t_visible_end:
            continue

        clipped_on  = max(nr.t_on,  t_visible_start)
        clipped_off = min(nr.t_off, t_visible_end)

        # y=0 is top; piano is at bottom → notes arriving at piano are at bottom
        # time t is at the piano line; time t+lookahead is at the top
        def _time_to_y(ts: float) -> float:
            return fall_h - (ts - t) * px_per_sec

        y_bottom = _time_to_y(clipped_on)
        y_top    = _time_to_y(clipped_off)
        bar_h    = max(y_bottom - y_top, 2)

        r, g, b  = nr.color

        if nr.pitch >= 0 and nr.channel != CH_DAC:
            # pitched note → position over piano key
            rel_x, key_w, is_white = _note_key_info(nr.pitch, min_white, white_w, black_w)
            x = piano_x0 + rel_x
            w = max(key_w - 2, 2)

            # Draw bar with rounded top
            bar_rect = pygame.Rect(int(x), int(y_top), int(w), int(bar_h))
            pygame.draw.rect(surface, (r, g, b, 200), bar_rect, border_radius=3)

            # bright highlight strip at leading edge
            if bar_h > 4:
                pygame.draw.rect(surface, (min(r+80,255), min(g+80,255), min(b+80,255)),
                                 pygame.Rect(int(x), int(y_top), int(w), 3),
                                 border_radius=3)
        else:
            # DAC / noise → left strip
            ch_index = list(CHANNEL_COLORS.keys()).index(nr.channel) if nr.channel in CHANNEL_COLORS else 0
            strip_slot = ([CH_DAC, CH_PSG_NOISE].index(nr.channel)
                          if nr.channel in (CH_DAC, CH_PSG_NOISE) else 0)
            slot_w = dac_strip_w // 2
            x = strip_slot * slot_w
            w = slot_w - 2
            bar_rect = pygame.Rect(int(x), int(y_top), int(w), int(bar_h))
            pygame.draw.rect(surface, (r, g, b), bar_rect, border_radius=2)

    # ── piano keyboard ────────────────────────────────────────────────────────

    piano_y = fall_h
    pygame.draw.rect(surface, PIANO_BG_COLOR,
                     pygame.Rect(0, piano_y, screen_w, piano_h))
    pygame.draw.line(surface, (80, 80, 130), (0, piano_y), (screen_w, piano_y), 2)

    # figure out which notes are active right now (for key lighting)
    active_pitches: set[int] = set()
    active_ch:    dict[int, int] = {}   # pitch → channel
    for nr in rects:
        if nr.t_on <= t < nr.t_off and nr.pitch >= 0:
            active_pitches.add(nr.pitch)
            active_ch[nr.pitch] = nr.channel

    white_key_h = piano_h - 4
    black_key_h = int(white_key_h * 0.60)

    # draw white keys first
    n_whites = _white_index(pitch_max) - min_white + 1
    for wi in range(n_whites):
        x   = int(piano_x0 + wi * white_w)
        w   = int(white_w) - 1
        # find which MIDI pitch this white key is
        midi = _white_key_midi(wi + min_white)
        if midi is None:
            continue
        is_active = midi in active_pitches
        if is_active:
            ch    = active_ch[midi]
            color = CHANNEL_COLORS.get(ch, (255, 255, 200))
            # brighten
            color = tuple(min(c + 60, 255) for c in color)
        else:
            color = WHITE_KEY_CLR
        pygame.draw.rect(surface, color,
                         pygame.Rect(x, piano_y + 2, w, white_key_h - 2),
                         border_radius=2)
        pygame.draw.rect(surface, (80, 80, 100),
                         pygame.Rect(x, piano_y + 2, w, white_key_h - 2),
                         width=1, border_radius=2)

    # draw black keys on top
    for wi in range(n_whites):
        midi_white = _white_key_midi(wi + min_white)
        if midi_white is None:
            continue
        # check if there's a black key to the right of this white key
        # (i.e., midi_white+1 is a black key and is in our range)
        midi_black = midi_white + 1
        if (midi_black <= pitch_max and
                not _NOTE_IS_WHITE[midi_black % 12]):
            is_active = midi_black in active_pitches
            if is_active:
                ch    = active_ch[midi_black]
                color = CHANNEL_COLORS.get(ch, (255, 255, 100))
            else:
                color = BLACK_KEY_CLR
            x = int(piano_x0 + wi * white_w + white_w * 0.55)
            w = int(black_w)
            pygame.draw.rect(surface, color,
                             pygame.Rect(x, piano_y + 2, w, black_key_h),
                             border_radius=2)

    # ── DAC/Noise label strip header ─────────────────────────────────────────
    if dac_strip_w > 0:
        pygame.draw.line(surface, DIVIDER_COLOR,
                         (dac_strip_w, 0), (dac_strip_w, screen_h), 1)
        for i, (ch_id, label) in enumerate([(CH_DAC, "DAC"), (CH_PSG_NOISE, "Noise")]):
            slot_w = dac_strip_w // 2
            r, g, b = CHANNEL_COLORS.get(ch_id, (150, 150, 150))
            if font:
                txt = font.render(label, True, (r, g, b))
                surface.blit(txt, (i * slot_w + 2, 4))

    # ── legend ────────────────────────────────────────────────────────────────
    if font:
        x_leg = dac_strip_w + 8
        y_leg = 8
        for ch_id, name in CHANNEL_NAMES.items():
            if ch_id in (CH_DAC, CH_PSG_NOISE):
                continue
            color = CHANNEL_COLORS.get(ch_id, (200, 200, 200))
            pygame.draw.rect(surface, color, pygame.Rect(x_leg, y_leg, 12, 10))
            if font:
                txt = font.render(name, True, (200, 200, 220))
                surface.blit(txt, (x_leg + 16, y_leg))
            x_leg += 60

    # ── playhead line ─────────────────────────────────────────────────────────
    pygame.draw.line(surface, (200, 200, 255, 180),
                     (dac_strip_w, fall_h - 1), (screen_w, fall_h - 1), 2)


def _white_key_midi(white_global_index: int) -> Optional[int]:
    """Convert a global white-key index to a MIDI note number."""
    octave         = white_global_index // 7
    white_in_oct   = white_global_index % 7
    # white key semitone offsets within octave: C=0,D=2,E=4,F=5,G=7,A=9,B=11
    WHITE_SEMITONES = [0, 2, 4, 5, 7, 9, 11]
    if white_in_oct >= len(WHITE_SEMITONES):
        return None
    return octave * 12 + WHITE_SEMITONES[white_in_oct]


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesia-style VGM piano visualizer")
    parser.add_argument("vgm",       help="Path to .vgm or .vgz file")
    parser.add_argument("--mp4",     metavar="PATH", help="Export to MP4 instead of live view")
    parser.add_argument("--fps",     type=int, default=60)
    parser.add_argument("--width",   type=int, default=1280)
    parser.add_argument("--height",  type=int, default=720)
    parser.add_argument("--lookahead", type=float, default=4.0,
                        help="Seconds of notes visible above keyboard (default 4)")
    parser.add_argument("--piano-height", type=int, default=120, dest="piano_height")
    parser.add_argument("--vgmplay-dir", type=Path, default=_DEFAULT_VGMPLAY_DIR,
                        dest="vgmplay_dir", help="Path to VGMPlay_040-9 directory")
    args = parser.parse_args()

    vgm_path = Path(args.vgm)
    if not vgm_path.exists():
        sys.exit(f"File not found: {vgm_path}")

    print(f"Loading {vgm_path.name} …")
    rects, total_sec = _parse_notes(vgm_path)
    print(f"  {len(rects)} note events, {total_sec:.1f}s")

    pitched = [r for r in rects if r.pitch >= 0 and r.channel != CH_DAC]
    if not pitched:
        sys.exit("No pitched notes found.")

    pitch_min  = min(r.pitch for r in pitched)
    pitch_max  = max(r.pitch for r in pitched)
    # expand to full octave boundaries
    pitch_min  = (pitch_min // 12) * 12
    pitch_max  = ((pitch_max // 12) + 1) * 12 - 1

    min_white  = _white_index(pitch_min)
    max_white  = _white_index(pitch_max)
    n_whites   = max_white - min_white + 1

    DAC_STRIP_W = 100
    piano_pixel_w = args.width - DAC_STRIP_W
    white_w    = piano_pixel_w / n_whites
    black_w    = white_w * 0.60
    piano_x0   = DAC_STRIP_W  # where the piano starts on screen

    import pygame
    pygame.init()
    font = pygame.font.SysFont("monospace", 10)

    # Try to render audio mix for interactive mode
    mix_wav: Optional[Path] = None
    if not args.mp4:
        try:
            renderer = _MixRenderer(args.vgmplay_dir)
            wav_dir  = _ROOT / "output" / "roundtrip" / (vgm_path.stem + "_wavs")
            mix_wav  = renderer.render_mix(vgm_path, wav_dir)
        except FileNotFoundError as e:
            print(f"  Warning: {e} — running without audio")

    if args.mp4:
        _export_mp4(args, rects, total_sec, pitch_min, pitch_max,
                    min_white, n_whites, white_w, black_w, piano_x0,
                    DAC_STRIP_W, font, vgm_path)
    else:
        _interactive(args, rects, total_sec, mix_wav, total_sec, pitch_min, pitch_max,
                     min_white, n_whites, white_w, black_w, piano_x0,
                     DAC_STRIP_W, font, vgm_path)


def _interactive(args, rects, total_sec, mix_wav, song_dur, pitch_min, pitch_max,
                 min_white, n_whites, white_w, black_w, piano_x0,
                 dac_strip_w, font, vgm_path):
    import pygame

    screen = pygame.display.set_mode((args.width, args.height))
    pygame.display.set_caption(f"Synthesia — {vgm_path.stem}")
    clock  = pygame.time.Clock()

    # ── audio setup ──────────────────────────────────────────────────────────
    has_audio = mix_wav is not None and mix_wav.exists()
    if has_audio:
        pygame.mixer.init(frequency=AUDIO_RATE, size=-16, channels=2, buffer=2048)
        pygame.mixer.music.load(str(mix_wav))

    # t = playback position in seconds (negative = pre-roll before song starts)
    t      = -args.lookahead
    t_end  = song_dur + 1.0
    paused = False

    def _start_audio_at(pos_sec: float) -> None:
        """Start or restart audio playback from pos_sec (>=0)."""
        if not has_audio:
            return
        pygame.mixer.music.stop()
        if pos_sec >= 0:
            pygame.mixer.music.play(start=pos_sec)
        # negative pos_sec = pre-roll, don't start audio yet

    _start_audio_at(t)

    print("Controls: SPACE=pause, LEFT/RIGHT=seek 5s, Q/ESC=quit")
    audio_started = t >= 0

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
                # Drive t from audio clock for perfect sync
                pos_ms = pygame.mixer.music.get_pos()
                if pos_ms >= 0:
                    t = pos_ms / 1000.0
                else:
                    # audio finished
                    t = t_end
            else:
                dt = clock.tick(args.fps) / 1000.0
                t += dt
                # kick off audio once we reach t=0 in pre-roll
                if has_audio and not audio_started and t >= 0:
                    pygame.mixer.music.play(start=0)
                    audio_started = True
                    t = 0.0
        else:
            clock.tick(30)

        if t > t_end:
            # loop: restart
            t = -args.lookahead
            audio_started = False
            if has_audio:
                pygame.mixer.music.stop()

        _render_frame(screen, rects, t, args.lookahead,
                      args.width, args.height, args.piano_height,
                      dac_strip_w, pitch_min, pitch_max,
                      white_w, black_w, min_white, piano_x0, font)

        pygame.display.flip()
        if not (has_audio and audio_started and not paused):
            clock.tick(args.fps)


def _export_mp4(args, rects, total_sec, pitch_min, pitch_max,
                min_white, n_whites, white_w, black_w, piano_x0,
                dac_strip_w, font, vgm_path):
    import pygame
    import numpy as np

    out_path = Path(args.mp4)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    surface = pygame.Surface((args.width, args.height))
    spf     = 1.0 / args.fps
    total_frames = int((total_sec + args.lookahead) * args.fps)

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{args.width}x{args.height}",
        "-pix_fmt", "rgb24",
        "-r", str(args.fps),
        "-i", "pipe:0",
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        str(out_path),
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    print(f"Exporting {total_frames} frames -> {out_path} ...")
    for frame_i in range(total_frames):
        t = -args.lookahead + frame_i * spf
        _render_frame(surface, rects, t, args.lookahead,
                      args.width, args.height, args.piano_height,
                      dac_strip_w, pitch_min, pitch_max,
                      white_w, black_w, min_white, piano_x0, font)
        raw = pygame.surfarray.array3d(surface)
        # surfarray gives (w,h,3), ffmpeg wants (h,w,3) row-major
        proc.stdin.write(raw.transpose(1, 0, 2).tobytes())
        if frame_i % (args.fps * 5) == 0:
            print(f"  {t:.1f}s / {total_sec:.1f}s")

    proc.stdin.close()
    proc.wait()
    print(f"Saved: {out_path}")
    pygame.quit()


if __name__ == "__main__":
    main()
