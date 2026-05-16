#!/usr/bin/env python3
"""vgm_oscilloscope.py — Multi-channel oscilloscope visualizer for VGM files.

Renders each YM2612 / SN76489 channel to an isolated WAV via VGMPlay, then
visualizes them as oscilloscope traces.

Modes
-----
  (default)    Render per-channel WAVs + show static matplotlib preview
  --play       Live oscilloscope player (pygame audio + matplotlib display)
  --mp4        Export oscilloscope video to MP4 via ffmpeg
  --corrscope  Write corrscope YAML config for corrscope tool

Usage
-----
  cd d:\\dev\\genesis-music-ml
  python scripts/vgm_oscilloscope.py output/v5d/gen_003.vgm
  python scripts/vgm_oscilloscope.py output/v5d/gen_003.vgm --mp4
  python scripts/vgm_oscilloscope.py output/v5d/gen_003.vgm --play
  python scripts/vgm_oscilloscope.py output/v5d/gen_003.vgm --corrscope
  python scripts/vgm_oscilloscope.py output/v5d/gen_003.vgm --mp4 --fps 60 --width 1920 --height 1080

Notes
-----
  corrscope (https://github.com/corrscope/corrscope) is the de-facto YouTube
  oscilloscope tool for chiptune — use --corrscope + `pip install corrscope`
  for best visual results.  This script handles all the VGMPlay plumbing so
  corrscope gets clean per-channel WAVs.
"""

from __future__ import annotations

import argparse
import bisect
import io
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────

_HERE = Path(__file__).resolve().parent           # scripts/
_ROOT = _HERE.parent                              # genesis-music-ml/
sys.path.insert(0, str(_ROOT / "src"))

_DEFAULT_VGMPLAY_DIR = _ROOT.parent / "VGMPlay_040-9"

# VGM internal timing is 44100 Hz; rendered audio is 48000 Hz
VGM_RATE = 44100
AUDIO_RATE = 48000
APP_NAME = "VgmLLM Scope"

from genesis_music.vgm_parser import load_vgm
from genesis_music.ym2612 import (
    decode_vgm,
    CH_DAC, CH_FM_0, CH_PSG_0, CH_PSG_1, CH_PSG_2, CH_PSG_NOISE,
)

# ── channel table ─────────────────────────────────────────────────────────────
# (ym2612_ch_id, ini_section, vgmplay_bit, n_chip_bits, display_name, hex_color)
# vgmplay_bit is the bit index in that chip's VGMPlay MuteMask

_CH_TABLE = [
    # YM2612: channels 0-5 FM + 6 DAC (7 channels total → 7-bit mask)
    (0,          "YM2612",  0, 7, "YM2612 FM1",     "#ff5533"),
    (1,          "YM2612",  1, 7, "YM2612 FM2",     "#ffcc00"),
    (2,          "YM2612",  2, 7, "YM2612 FM3",     "#44dd44"),
    (3,          "YM2612",  3, 7, "YM2612 FM4",     "#22ccdd"),
    (4,          "YM2612",  4, 7, "YM2612 FM5",     "#4488ff"),
    (5,          "YM2612",  5, 7, "YM2612 FM6",     "#cc44ff"),
    (CH_DAC,     "YM2612",  6, 7, "YM2612 DAC",     "#ff44aa"),
    # SN76496: channels 0-2 tone + 3 noise (4-bit mask)
    (CH_PSG_0,   "SN76496", 0, 4, "SN76489 Sq1",   "#ffe0b2"),
    (CH_PSG_1,   "SN76496", 1, 4, "SN76489 Sq2",   "#b3e5fc"),
    (CH_PSG_2,   "SN76496", 2, 4, "SN76489 Sq3",   "#c8e6c9"),
    (CH_PSG_NOISE,"SN76496",3, 4, "SN76489 Noise", "#90a4ae"),
]

# Sections that carry the other chip (used when muting sibling chips)
_SIBLING_SECTIONS = {"YM2612": ("SN76496", 4), "SN76496": ("YM2612", 7)}


@dataclass
class Channel:
    ch_id: int              # ym2612.py CH_* constant
    ini_section: str        # "[YM2612]" or "[SN76496]"
    vgmplay_bit: int        # bit index in that chip's MuteMask
    n_chip_bits: int        # total channels in chip (mask width)
    name: str
    color: str
    wav_path: Optional[Path] = None
    audio: Optional[np.ndarray] = None   # float32 mono, shape (N,)

    @property
    def isolate_mask(self) -> int:
        """MuteMask value that keeps ONLY this channel active."""
        full = (1 << self.n_chip_bits) - 1
        return full & ~(1 << self.vgmplay_bit)

    @property
    def mute_all_mask(self) -> int:
        return (1 << self.n_chip_bits) - 1

    @property
    def file_stem(self) -> str:
        return self.name.replace(" ", "").lower()

    @property
    def use_trigger(self) -> bool:
        """False for percussive/noise channels that shouldn't be trigger-locked."""
        return self.ch_id not in (CH_DAC, CH_PSG_NOISE)


def _make_channel_table() -> list[Channel]:
    return [Channel(*row) for row in _CH_TABLE]


# ── VGMPlay renderer ──────────────────────────────────────────────────────────

class VGMPlayRenderer:
    """Calls VGMPlay.exe with a patched ini to render isolated channel WAVs."""

    def __init__(self, vgmplay_dir: Path):
        self.exe = vgmplay_dir / "VGMPlay.exe"
        self.ini = vgmplay_dir / "VGMPlay.ini"
        if not self.exe.exists():
            raise FileNotFoundError(f"VGMPlay.exe not found: {self.exe}\n"
                                    f"Pass --vgmplay-dir to specify its location.")

    def _patch_ini(self, original: str, ym2612_mask: int, psg_mask: int) -> str:
        """Return patched ini text with LogSound=1, SndOut=-1 and per-chip MuteMasks."""
        # If ini is empty, build the minimal block then append chip sections
        if not original.strip():
            original = (
                "[General]\nLogSound = 1\nSndOut = -1\n"
                f"[YM2612]\nMuteMask = 0x{ym2612_mask:02X}\n"
                f"[SN76496]\nMuteMask = 0x{psg_mask:02X}\n"
            )
            return original
        lines = original.splitlines()
        result: list[str] = []
        current_section: Optional[str] = None
        log_sound_patched = False
        sndout_patched = False

        for line in lines:
            stripped = line.strip()

            # Section header
            m = re.match(r"^\[(\w+)\]$", stripped)
            if m:
                current_section = m.group(1)
                result.append(line)
                # Insert MuteMask immediately after section header
                if current_section == "YM2612":
                    result.append(f"MuteMask = 0x{ym2612_mask:02X}")
                elif current_section == "SN76496":
                    result.append(f"MuteMask = 0x{psg_mask:02X}")
                continue

            # Skip any pre-existing MuteMask lines in controlled sections
            if current_section in ("YM2612", "SN76496") and re.match(r"MuteMask\s*=", stripped):
                continue

            # Patch LogSound in [General]
            if current_section == "General" and re.match(r"LogSound\s*=", stripped):
                result.append("LogSound = 1")
                log_sound_patched = True
                continue

            # Patch SndOut in [General] — -1 = no audio device (silent, faster than real-time)
            if current_section == "General" and re.match(r"SndOut\s*=", stripped):
                result.append("SndOut = -1")
                sndout_patched = True
                continue

            result.append(line)

        # If LogSound / SndOut lines didn't exist, insert them after [General] header
        if not log_sound_patched or not sndout_patched:
            final: list[str] = []
            for line in result:
                final.append(line)
                if line.strip() == "[General]":
                    if not log_sound_patched:
                        final.append("LogSound = 1")
                    if not sndout_patched:
                        final.append("SndOut = -1")
            result = final

        return "\n".join(result)

    def _run_vgmplay(self, vgm_path: Path, ym2612_mask: int, psg_mask: int) -> Optional[Path]:
        """
        Run VGMPlay with patched ini.  Returns the WAV path VGMPlay produced
        (next to the VGM file), or None if the WAV was not created.
        """
        original_ini = self.ini.read_bytes()
        try:
            patched = self._patch_ini(original_ini.decode("utf-8", errors="replace"),
                                      ym2612_mask, psg_mask)
            self.ini.write_text(patched, encoding="utf-8")
            subprocess.run(
                [str(self.exe), str(vgm_path)],
                cwd=str(self.exe.parent),
                capture_output=True,
                timeout=600,
            )
        finally:
            self.ini.write_bytes(original_ini)

        expected_wav = vgm_path.parent / (vgm_path.stem + ".wav")
        return expected_wav if expected_wav.exists() else None

    def render_mix(self, vgm_path: Path, out_dir: Path) -> Optional[Path]:
        """Render the full stereo mix with all channels active."""
        out_dir.mkdir(parents=True, exist_ok=True)
        out_wav = out_dir / f"{vgm_path.stem}_mix.wav"
        if out_wav.exists():
            print(f"  [cached] {out_wav.name}")
            return out_wav
        print(f"  Rendering mix …", end=" ", flush=True)
        t0 = time.time()
        wav = self._run_vgmplay(vgm_path, ym2612_mask=0, psg_mask=0)
        if wav:
            shutil.move(str(wav), str(out_wav))
            print(f"{time.time()-t0:.1f}s")
            return out_wav
        print("FAILED")
        return None

    def render_channel(self, vgm_path: Path, ch: Channel, out_dir: Path) -> Optional[Path]:
        """Render one channel in isolation."""
        out_dir.mkdir(parents=True, exist_ok=True)
        out_wav = out_dir / f"{vgm_path.stem}_{ch.file_stem}.wav"
        if out_wav.exists():
            print(f"  [cached] {out_wav.name}")
            return out_wav

        # Build masks: isolate target chip/channel, silence the sibling chip
        if ch.ini_section == "YM2612":
            sibling_n = _SIBLING_SECTIONS["YM2612"][1]   # SN76496 has 4 channels
            ym_mask_val  = ch.isolate_mask                 # keep only this FM ch
            psg_mask_val = (1 << sibling_n) - 1           # mute all PSG
        else:
            ym_n = _SIBLING_SECTIONS["SN76496"][1]        # YM2612 has 7 channels
            ym_mask_val  = (1 << ym_n) - 1                # mute all FM + DAC
            psg_mask_val = ch.isolate_mask                 # keep only this PSG ch

        print(f"  Rendering {ch.name} …", end=" ", flush=True)
        t0 = time.time()
        wav = self._run_vgmplay(vgm_path, ym2612_mask=ym_mask_val, psg_mask=psg_mask_val)
        if wav:
            shutil.move(str(wav), str(out_wav))
            print(f"{time.time()-t0:.1f}s")
            return out_wav
        print("FAILED (channel may be unused)")
        return None


# ── WAV loading ───────────────────────────────────────────────────────────────

def load_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    """Load a WAV file and return (float32 mono array, sample_rate)."""
    with wave.open(str(path), "rb") as wf:
        n_ch = wf.getnchannels()
        swidth = wf.getsampwidth()
        rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if swidth == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif swidth == 3:
        # 24-bit: unpack manually
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        samples = (b[:, 0].astype(np.int32) |
                   (b[:, 1].astype(np.int32) << 8) |
                   (b[:, 2].astype(np.int32) << 16))
        samples[samples >= 2**23] -= 2**24
        samples = samples.astype(np.float32) / (2**23)
    else:
        raise ValueError(f"Unsupported sample width {swidth} in {path}")

    if n_ch == 2:
        samples = samples.reshape(-1, 2).mean(axis=1)
    elif n_ch > 2:
        samples = samples.reshape(-1, n_ch).mean(axis=1)

    return samples, rate


def is_active(audio: np.ndarray, threshold_rms: float = 1e-4) -> bool:
    """True if the channel has significant audio content."""
    return float(np.sqrt(np.mean(audio ** 2))) > threshold_rms


# ── detect active channels + build dynamic state timelines ──────────────────

def build_vgm_info(
    vgm_path: Path,
) -> tuple[set[int], list[tuple[int, int]], list[tuple[int, int]]]:
    """
    Decode VGM once and return:
      active_ids        — set of channel IDs that have notes
      dac_ranges        — sorted [(on_vgm_sample, off_vgm_sample)] for DAC
      ch3_special_ranges — sorted [(on, off)] for FM3 special-mode notes
    All sample times are in VGM ticks @ 44100 Hz.
    """
    try:
        vgm = load_vgm(vgm_path)
        notes, _ = decode_vgm(vgm)
    except Exception as e:
        print(f"  Warning: could not parse VGM: {e}")
        return set(range(11)), [], []

    active_ids = set(n.channel for n in notes)

    dac_ranges = sorted(
        [(n.sample_on, n.sample_off) for n in notes if n.channel == CH_DAC],
        key=lambda r: r[0],
    )
    ch3_special_ranges = sorted(
        [
            (n.sample_on, n.sample_off)
            for n in notes
            if n.channel == 2 and n.patch and n.patch.ch3_mode != 0
        ],
        key=lambda r: r[0],
    )
    return active_ids, dac_ranges, ch3_special_ranges


def _make_is_active(ranges: list[tuple[int, int]]):
    """Return a fast O(log n) callable: vgm_sample -> bool."""
    if not ranges:
        return lambda s: False
    ons = [r[0] for r in ranges]

    def check(sample: int) -> bool:
        idx = bisect.bisect_right(ons, sample) - 1
        return idx >= 0 and ranges[idx][0] <= sample < ranges[idx][1]

    return check


def _dynamic_title(ch: Channel, vgm_pos: int, dac_is_active, ch3_special_is_active) -> str:
    """Return the live subplot title for a channel at a given VGM sample position."""
    if ch.ch_id == 5 and dac_is_active(vgm_pos):        # FM6 silenced by DAC
        return f"{ch.name}  ● DAC"
    if ch.ch_id == 2 and ch3_special_is_active(vgm_pos):  # FM3 special mode
        return f"{ch.name}  ● SPECIAL"
    return ch.name


# keep backward-compat alias
def detect_active_ch_ids(vgm_path: Path) -> set[int]:
    active_ids, _, _ = build_vgm_info(vgm_path)
    return active_ids


# ── oscilloscope trigger ──────────────────────────────────────────────────────

def find_trigger_offset(audio: np.ndarray, center: int, window: int) -> int:
    """
    Find the nearest rising zero-crossing around `center` for a stable
    oscilloscope display.  Returns a start index such that
    audio[start : start+window] is centred on a zero crossing.
    """
    half = window // 2
    search_start = max(0, center - half)
    search_end = min(len(audio) - window, center + half)
    if search_start >= search_end:
        return max(0, min(center - half, len(audio) - window))

    segment = audio[search_start:search_end + 1]
    # Find rising zero crossings
    crossings = np.where((segment[:-1] < 0) & (segment[1:] >= 0))[0]
    if len(crossings) == 0:
        return max(0, center - half)

    # Pick the crossing closest to center
    best = crossings[np.argmin(np.abs(crossings - half))]
    return search_start + best


def get_oscilloscope_window(
    audio: np.ndarray, pos: int, window_samples: int, triggered: bool = True
) -> np.ndarray:
    """Return a window of `window_samples` centered near `pos`.

    When triggered=True (default) the window is snapped to the nearest
    rising zero-crossing for a stable trace.  Pass triggered=False for
    percussive / noise channels (DAC, PSG Noise) so the raw audio scrolls
    through without the trigger hunting in silence.
    """
    if triggered:
        start = find_trigger_offset(audio, pos, window_samples)
    else:
        half = window_samples // 2
        start = max(0, min(pos - half, len(audio) - window_samples))
    end = start + window_samples
    if end <= len(audio):
        return audio[start:end]
    # Pad if near end of file
    seg = audio[start:]
    return np.pad(seg, (0, window_samples - len(seg)))


# ── adaptive windowing ────────────────────────────────────────────────────────

def detect_pitch_hz(
    segment: np.ndarray,
    sample_rate: int,
    min_hz: float = 40.0,
    max_hz: float = 8000.0,
) -> Optional[float]:
    """Estimate dominant pitch via autocorrelation.  Returns Hz or None."""
    if len(segment) < 8:
        return None
    seg = segment - np.mean(segment)
    if float(np.sqrt(np.mean(seg ** 2))) < 1e-4:
        return None

    min_lag = max(1, int(sample_rate / max_hz))
    max_lag = min(int(sample_rate / min_hz), len(seg) // 2)
    if min_lag >= max_lag:
        return None

    n = len(seg)
    n_fft = 1 << (2 * n - 1).bit_length()  # next power-of-2 >= 2*n
    fft_out = np.fft.rfft(seg, n=n_fft)
    acf = np.fft.irfft(fft_out * np.conj(fft_out))[:n]
    if acf[0] < 1e-10:
        return None
    acf /= acf[0]

    search = acf[min_lag : max_lag + 1]
    if len(search) == 0:
        return None
    peak_idx = int(np.argmax(search))
    if search[peak_idx] < 0.3:   # weak periodicity → treat as unpitched
        return None

    return float(sample_rate) / (min_lag + peak_idx)


def adaptive_window_samples(
    audio: np.ndarray,
    pos: int,
    sample_rate: int,
    target_cycles: float = 2.5,
    default_samples: int = None,
) -> int:
    """
    Return window size in samples showing ~target_cycles of the channel's
    current pitch.  Falls back to default_samples when pitch is undetectable
    (noise, silence, percussive content).
    """
    if default_samples is None:
        default_samples = int(0.08 * sample_rate)

    detect_n = int(0.10 * sample_rate)  # 100 ms for pitch analysis
    start = max(0, pos - detect_n // 2)
    end = min(len(audio), start + detect_n)
    if end - start < 8:
        return default_samples

    hz = detect_pitch_hz(audio[start:end], sample_rate)
    if hz is None:
        return default_samples

    window_s = target_cycles / hz
    window_s = max(0.008, min(0.250, window_s))  # clamp 8 ms – 250 ms
    return int(window_s * sample_rate)


# ── figure layout ─────────────────────────────────────────────────────────────

def build_layout(n_channels: int) -> tuple[int, int]:
    """Return (n_cols, n_rows) for a grid of n_channels subplots."""
    if n_channels <= 4:
        return 2, (n_channels + 1) // 2
    if n_channels <= 6:
        return 3, 2
    return 3, (n_channels + 2) // 3


# ── static preview ────────────────────────────────────────────────────────────

def show_preview(
    channels: list[Channel],
    sample_rate: int,
    window_s: float = 0.08,
    start_s: float = 1.0,
    title: str = "",
    dac_is_active=None,
    ch3_special_is_active=None,
    autoscale: bool = True,
):
    """Show a static matplotlib oscilloscope figure."""
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    if dac_is_active is None:
        dac_is_active = lambda s: False
    if ch3_special_is_active is None:
        ch3_special_is_active = lambda s: False

    window = int(window_s * sample_rate)
    pos = int(start_s * sample_rate)
    vgm_pos = int(pos * VGM_RATE / sample_rate)
    n = len(channels)
    n_cols, n_rows = build_layout(n)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 2.5 * n_rows))
    fig.patch.set_facecolor("#0a0a0a")
    sub = f"  ·  {title}" if title else ""
    fig.suptitle(f"{APP_NAME}{sub}", color="#cccccc", fontsize=11)
    axes_flat = np.array(axes).flatten()

    for i, (ax, ch) in enumerate(zip(axes_flat, channels)):
        if autoscale and ch.audio is not None and len(ch.audio) > 0:
            this_window = adaptive_window_samples(ch.audio, pos, sample_rate,
                                                  default_samples=window)
        else:
            this_window = window
        t = np.linspace(0, this_window / sample_rate * 1000, this_window)

        ax.set_facecolor("#0d0d0d")
        ax.tick_params(colors="#444444", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#222222")
        if autoscale:
            ax.set_yticks([])
        ax.set_title(_dynamic_title(ch, vgm_pos, dac_is_active, ch3_special_is_active),
                     color=ch.color, fontsize=9, pad=3)
        ax.set_ylim(-1.1, 1.1)
        ax.axhline(0, color="#222222", linewidth=0.5)
        ax.set_xlabel("ms", color="#444444", fontsize=7)

        if ch.audio is not None and len(ch.audio) > 0:
            wnd = get_oscilloscope_window(ch.audio, pos, this_window, triggered=ch.use_trigger)
            ax.plot(t, wnd, color=ch.color, linewidth=0.9, antialiased=True)
        else:
            ax.plot(t, np.zeros(this_window), color=ch.color, linewidth=0.9)

    # Hide unused axes
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()


# ── live player ───────────────────────────────────────────────────────────────

def play_live(
    channels: list[Channel],
    mix_wav: Path,
    sample_rate: int,
    window_s: float = 0.08,
    dac_is_active=None,
    ch3_special_is_active=None,
    autoscale: bool = True,
):
    """Play audio with real-time oscilloscope using pygame + matplotlib."""
    try:
        import pygame
    except ImportError:
        print("pygame not installed.  Run: pip install pygame")
        return

    if dac_is_active is None:
        dac_is_active = lambda s: False
    if ch3_special_is_active is None:
        ch3_special_is_active = lambda s: False

    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    window = int(window_s * sample_rate)
    n = len(channels)
    n_cols, n_rows = build_layout(n)

    pygame.mixer.init(frequency=sample_rate, size=-16, channels=2, buffer=2048)
    pygame.mixer.music.load(str(mix_wav))
    pygame.mixer.music.play()

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 2.5 * n_rows))
    fig.patch.set_facecolor("#0a0a0a")
    fig.suptitle(f"{APP_NAME}  ·  {mix_wav.stem}", color="#cccccc", fontsize=11)
    axes_flat = np.array(axes).flatten()
    t = np.linspace(0, window_s * 1000, window)

    lines = []
    title_texts = []
    for i, (ax, ch) in enumerate(zip(axes_flat, channels)):
        ax.set_facecolor("#0d0d0d")
        ax.tick_params(colors="#444444", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#222222")
        if autoscale:
            ax.set_yticks([])
        tt = ax.set_title(ch.name, color=ch.color, fontsize=9, pad=3)
        title_texts.append(tt)
        ax.set_ylim(-1.1, 1.1)
        ax.axhline(0, color="#222222", linewidth=0.5)
        ax.set_xlabel("ms", color="#444444", fontsize=7)
        (ln,) = ax.plot(t, np.zeros(window), color=ch.color, linewidth=0.9)
        lines.append(ln)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.ion()
    plt.show(block=False)

    # Timer label — top-left of figure, outside subplot area
    timer_text = fig.text(0.01, 0.99, "0:00.000",
                          color="#888888", fontsize=9,
                          ha="left", va="top",
                          fontfamily="monospace")

    print("Playing … close window or press Ctrl+C to stop.")
    try:
        while pygame.mixer.music.get_busy():
            pos_ms = pygame.mixer.music.get_pos()
            pos_sample = int(pos_ms / 1000.0 * sample_rate)
            vgm_pos = int(pos_sample * VGM_RATE / sample_rate)
            mins, secs = divmod(pos_ms / 1000.0, 60)
            timer_text.set_text(f"{int(mins)}:{secs:06.3f}")

            for ln, ch in zip(lines, channels):
                if ch.audio is not None and len(ch.audio) > 0:
                    if autoscale:
                        this_win = adaptive_window_samples(ch.audio, pos_sample,
                                                           sample_rate, default_samples=window)
                    else:
                        this_win = window
                    wnd = get_oscilloscope_window(ch.audio, pos_sample, this_win,
                                                  triggered=ch.use_trigger)
                    t_ch = np.linspace(0, this_win / sample_rate * 1000, this_win)
                    ln.set_xdata(t_ch)
                    ln.set_ydata(wnd)
                    if autoscale:
                        ln.axes.set_xlim(0, t_ch[-1])

            for tt, ch in zip(title_texts, channels):
                new_title = _dynamic_title(ch, vgm_pos, dac_is_active, ch3_special_is_active)
                if tt.get_text() != new_title:
                    tt.set_text(new_title)

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.033)

    except KeyboardInterrupt:
        pass
    finally:
        pygame.mixer.music.stop()
        pygame.mixer.quit()

    plt.ioff()
    plt.show()


# ── MP4 export ────────────────────────────────────────────────────────────────

def export_mp4(
    channels: list[Channel],
    mix_wav: Path,
    out_mp4: Path,
    sample_rate: int,
    fps: int = 60,
    width: int = 1920,
    height: int = 1080,
    window_s: float = 0.08,
    dac_is_active=None,
    ch3_special_is_active=None,
    autoscale: bool = True,
):
    """Export oscilloscope video to MP4 using ffmpeg."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if dac_is_active is None:
        dac_is_active = lambda s: False
    if ch3_special_is_active is None:
        ch3_special_is_active = lambda s: False

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("ffmpeg not found in PATH.  Cannot export MP4.")
        return

    window = int(window_s * sample_rate)
    n = len(channels)
    n_cols, n_rows = build_layout(n)
    dpi = 96
    fig_w = width / dpi
    fig_h = height / dpi

    # Determine total duration from the longest channel audio
    max_samples = max((len(ch.audio) for ch in channels if ch.audio is not None), default=0)
    if max_samples == 0:
        print("No audio data to export.")
        return
    total_frames = int(max_samples / sample_rate * fps) + 1
    print(f"  Exporting {total_frames} frames @ {fps}fps -> {out_mp4.name}")

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor("#0a0a0a")
    fig.suptitle(f"{APP_NAME}  ·  {mix_wav.stem}", color="#cccccc", fontsize=14, y=0.99)
    axes_flat = np.array(axes).flatten()
    t = np.linspace(0, window_s * 1000, window)

    lines = []
    title_texts = []
    for i, (ax, ch) in enumerate(zip(axes_flat, channels)):
        ax.set_facecolor("#0d0d0d")
        ax.tick_params(colors="#444444", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#1a1a1a")
        if autoscale:
            ax.set_yticks([])
        tt = ax.set_title(ch.name, color=ch.color, fontsize=11, pad=4)
        title_texts.append(tt)
        ax.set_ylim(-1.1, 1.1)
        ax.axhline(0, color="#1e1e1e", linewidth=0.6)
        ax.set_xlabel("ms", color="#444444", fontsize=8)
        (ln,) = ax.plot(t, np.zeros(window), color=ch.color, linewidth=1.2, antialiased=True)
        lines.append(ln)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    plt.tight_layout()

    # Timer label — top-left of figure
    timer_text = fig.text(0.01, 0.99, "0:00.000",
                          color="#666666", fontsize=11,
                          ha="left", va="top",
                          fontfamily="monospace")

    # ffmpeg process: reads raw RGB frames from stdin, writes MP4
    cmd = [
        ffmpeg, "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{width}x{height}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "pipe:0",
        "-i", str(mix_wav),
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-preset", "fast",
        "-shortest",
        str(out_mp4),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    t0 = time.time()
    samples_per_frame = sample_rate / fps

    try:
        for frame_idx in range(total_frames):
            pos_sample = int(frame_idx * samples_per_frame)
            vgm_pos = int(pos_sample * VGM_RATE / sample_rate)
            pos_ms = pos_sample / sample_rate * 1000.0
            mins, secs = divmod(pos_ms / 1000.0, 60)
            timer_text.set_text(f"{int(mins)}:{secs:06.3f}")

            for ln, ch in zip(lines, channels):
                if ch.audio is not None and len(ch.audio) > 0:
                    if autoscale:
                        this_win = adaptive_window_samples(ch.audio, pos_sample,
                                                           sample_rate, default_samples=window)
                    else:
                        this_win = window
                    wnd = get_oscilloscope_window(ch.audio, pos_sample, this_win,
                                                  triggered=ch.use_trigger)
                    t_ch = np.linspace(0, this_win / sample_rate * 1000, this_win)
                    ln.set_xdata(t_ch)
                    ln.set_ydata(wnd)
                    if autoscale:
                        ln.axes.set_xlim(0, t_ch[-1])

            for tt, ch in zip(title_texts, channels):
                new_title = _dynamic_title(ch, vgm_pos, dac_is_active, ch3_special_is_active)
                if tt.get_text() != new_title:
                    tt.set_text(new_title)

            fig.canvas.draw()
            # buffer_rgba() → drop alpha → send RGB bytes to ffmpeg
            rgba = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            proc.stdin.write(rgba.tobytes())

            if frame_idx % fps == 0:
                elapsed = time.time() - t0
                pct = frame_idx / total_frames * 100
                remaining = (elapsed / max(frame_idx, 1)) * (total_frames - frame_idx)
                print(f"  {pct:5.1f}%  frame {frame_idx}/{total_frames}  "
                      f"ETA {remaining:.0f}s", end="\r", flush=True)

    finally:
        proc.stdin.close()
        proc.wait()
        plt.close(fig)

    print(f"\n  Done -> {out_mp4}  ({time.time()-t0:.1f}s total)")


# ── corrscope YAML generator ──────────────────────────────────────────────────

_CORRSCOPE_TEMPLATE = """\
# corrscope config generated by vgm_oscilloscope.py
# Install: pip install corrscope
# Run:     corrscope {yaml_path}
master_audio: {mix_wav}
fps: 60
width: 1920
height: 1080
amplification: 1.0
channels:
{channel_entries}
"""

_CH_ENTRY_TEMPLATE = """\
- name: "{name}"
  wav_path: {wav_path}
  color: "{color}"
"""


def write_corrscope_yaml(channels: list[Channel], mix_wav: Path, out_yaml: Path):
    """Write a corrscope YAML config pointing at the per-channel WAVs."""
    entries = ""
    for ch in channels:
        if ch.wav_path and ch.wav_path.exists():
            entries += _CH_ENTRY_TEMPLATE.format(
                name=ch.name,
                wav_path=ch.wav_path.as_posix(),
                color=ch.color,
            )
    text = _CORRSCOPE_TEMPLATE.format(
        yaml_path=out_yaml.name,
        mix_wav=mix_wav.as_posix(),
        channel_entries=entries,
    )
    out_yaml.write_text(text, encoding="utf-8")
    print(f"  corrscope config -> {out_yaml}")
    print(f"  Run: corrscope \"{out_yaml}\"")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multi-channel oscilloscope visualizer for VGM files."
    )
    parser.add_argument("vgm", type=Path, help="Input .vgm or .vgz file")
    parser.add_argument("--play",       action="store_true", help="Live oscilloscope player")
    parser.add_argument("--mp4",        action="store_true", help="Export MP4 video")
    parser.add_argument("--corrscope",  action="store_true", help="Write corrscope YAML")
    parser.add_argument("--preview-at", type=float, default=2.0, metavar="SEC",
                        help="Static preview: show oscilloscope at this timestamp (default 2.0)")
    parser.add_argument("--fps",    type=int,   default=60,   help="MP4 framerate (default 60)")
    parser.add_argument("--width",  type=int,   default=1920, help="MP4 width (default 1920)")
    parser.add_argument("--height", type=int,   default=1080, help="MP4 height (default 1080)")
    parser.add_argument("--window", type=float, default=0.08, metavar="SEC",
                        help="Oscilloscope window width in seconds (default 0.08)")
    parser.add_argument("--wav-dir", type=Path, default=None,
                        help="Directory to store per-channel WAVs (default: <vgm_dir>/<stem>_wavs/)")
    parser.add_argument("--vgmplay-dir", type=Path, default=_DEFAULT_VGMPLAY_DIR,
                        help=f"VGMPlay directory (default: {_DEFAULT_VGMPLAY_DIR})")
    parser.add_argument("--all-channels", action="store_true",
                        help="Render all channels, even apparently inactive ones")
    parser.add_argument("--no-autoscale", action="store_true",
                        help="Disable per-channel amplitude normalization (default: autoscale on)")
    args = parser.parse_args()

    vgm_path = args.vgm.resolve()
    if not vgm_path.exists():
        print(f"Error: VGM file not found: {vgm_path}")
        sys.exit(1)

    wav_dir = args.wav_dir or (vgm_path.parent / f"{vgm_path.stem}_wavs")
    renderer = VGMPlayRenderer(args.vgmplay_dir)
    sample_rate = 48000  # must match VGMPlay.ini SampleRate

    print(f"\n=== VGM Oscilloscope: {vgm_path.name} ===\n")

    # ── Step 1: detect active channels + build state timelines ─────────────
    print("Detecting active channels …")
    if not args.all_channels:
        active_ids, dac_ranges, ch3_special_ranges = build_vgm_info(vgm_path)
    else:
        active_ids = set(range(11))
        _, dac_ranges, ch3_special_ranges = build_vgm_info(vgm_path)
    dac_is_active = _make_is_active(dac_ranges)
    ch3_special_is_active = _make_is_active(ch3_special_ranges)
    print(f"  Active channel IDs: {sorted(active_ids)}")
    if dac_ranges:
        print(f"  DAC active: {len(dac_ranges)} note(s) — will badge FM6 live")
    if ch3_special_ranges:
        print(f"  CH3 special mode: {len(ch3_special_ranges)} note(s) — will badge FM3 live")

    all_ch = _make_channel_table()
    target_channels = [ch for ch in all_ch if ch.ch_id in active_ids]

    if not target_channels:
        print("No active channels found.  Use --all-channels to force rendering.")
        sys.exit(0)

    print(f"  Channels to render: {[ch.name for ch in target_channels]}\n")

    # ── Step 2: render mix WAV ────────────────────────────────────────────────
    print("Rendering full mix …")
    mix_wav = renderer.render_mix(vgm_path, wav_dir)
    if mix_wav is None:
        print("Error: mix render failed.  Check VGMPlay.exe path.")
        sys.exit(1)

    # ── Step 3: render per-channel WAVs ───────────────────────────────────────
    print(f"\nRendering {len(target_channels)} channel WAVs …")
    rendered: list[Channel] = []
    for ch in target_channels:
        wav = renderer.render_channel(vgm_path, ch, wav_dir)
        if wav:
            ch.wav_path = wav
            try:
                ch.audio, _ = load_wav_mono(wav)
                if not args.no_autoscale:
                    peak = float(np.max(np.abs(ch.audio)))
                    if peak > 1e-6:
                        ch.audio = ch.audio / peak
                if is_active(ch.audio):
                    rendered.append(ch)
                else:
                    print(f"  {ch.name}: rendered but silent — skipping")
            except Exception as e:
                print(f"  {ch.name}: WAV load error: {e}")

    if not rendered:
        print("No renderable channel audio found.")
        sys.exit(1)

    print(f"\n  {len(rendered)} active channels: {[ch.name for ch in rendered]}")

    # ── Step 4: output mode ───────────────────────────────────────────────────
    if args.corrscope:
        out_yaml = vgm_path.parent / f"{vgm_path.stem}_corrscope.yaml"
        print(f"\nWriting corrscope config …")
        write_corrscope_yaml(rendered, mix_wav, out_yaml)

    if args.mp4:
        out_mp4 = vgm_path.parent / f"{vgm_path.stem}_oscilloscope.mp4"
        print(f"\nExporting MP4 …")
        export_mp4(rendered, mix_wav, out_mp4,
                   sample_rate=sample_rate,
                   fps=args.fps,
                   width=args.width,
                   height=args.height,
                   window_s=args.window,
                   dac_is_active=dac_is_active,
                   ch3_special_is_active=ch3_special_is_active,
                   autoscale=not args.no_autoscale)

    elif args.play:
        print(f"\nStarting live player …")
        play_live(rendered, mix_wav, sample_rate, window_s=args.window,
                  dac_is_active=dac_is_active,
                  ch3_special_is_active=ch3_special_is_active,
                  autoscale=not args.no_autoscale)

    else:
        print(f"\nShowing preview at t={args.preview_at}s …")
        show_preview(rendered, sample_rate,
                     window_s=args.window,
                     start_s=args.preview_at,
                     title=vgm_path.stem,
                     dac_is_active=dac_is_active,
                     ch3_special_is_active=ch3_special_is_active,
                     autoscale=not args.no_autoscale)


if __name__ == "__main__":
    main()
