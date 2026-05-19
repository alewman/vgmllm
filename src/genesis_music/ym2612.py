"""YM2612 (OPN2) and SN76489 (PSG) state decoders.

Converts raw VGM register-write events into musical NoteEvents with MIDI
pitch, velocity, channel index, and FM patch information.

Reference:
  YM2612 Application Manual (Yamaha, 1996)
  https://vgmrips.net/wiki/YM2612
  SN76489 data sheet (Texas Instruments)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterator

from .vgm_parser import EventType, VgmEvent, VgmFile

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# NTSC Genesis master clock = 53,693,175 Hz; YM2612 runs at master/7
YM2612_CLOCK: int = 7_670_454  # Hz

# SN76489 on Genesis: master clock / 15
SN76489_CLOCK: int = 3_579_545  # Hz

# Channel index constants
CH_FM_0 = 0   # YM2612 FM channel 1
CH_FM_1 = 1
CH_FM_2 = 2
CH_FM_3 = 3   # YM2612 FM channel 4
CH_FM_4 = 4
CH_FM_5 = 5   # Also used for DAC when DAC mode is enabled
CH_DAC  = 6   # Logical DAC channel (YM2612 DAC, replaces FM6 when enabled)
CH_PSG_0 = 7  # SN76489 tone channel 0
CH_PSG_1 = 8  # SN76489 tone channel 1
CH_PSG_2 = 9  # SN76489 tone channel 2
CH_PSG_NOISE = 10  # SN76489 noise channel

# ---------------------------------------------------------------------------
# DAC onset detection
# ---------------------------------------------------------------------------

# Minimum silence gap (in audio samples @ 44100 Hz) between DAC_WRITE events
# that constitutes a new drum onset.  At 16kHz streaming, consecutive sample
# bytes arrive ~2-3 samples apart; anything larger signals the DAC stopped.
# 64 samples ≈ 1.5 ms.  Increase to 128 (≈ 3 ms) if you still see duplicates.
ONSET_GAP_SAMPLES: int = 64

# TL value at which a carrier operator is considered perceptually silent.
# Real hardware: TL 127 = -96 dB; 120 = ~-91 dB.  Both are inaudible.
# Using 120 catches composers who sweep TL to near-max to fade a voice
# without issuing a Key Off command.
TL_SILENCE_THRESHOLD: int = 120

# ---------------------------------------------------------------------------
# Unified perceptual velocity scale
# ---------------------------------------------------------------------------
# Both YM2612 and SN76489 are mapped to the same 0–15 scale by normalising
# their native attenuation values to a common −60 dB range.
#
#   FM (YM2612):  TL is in ~0.75 dB steps  (0 = 0 dB, 127 ≈ −95 dB)
#   PSG (SN76489): attenuation is in 2 dB steps (0 = 0 dB, 14 = −28 dB,
#                  15 = hardware silence)
#
# Unified mapping:
#   vel 15 = 0 dB  (loudest)
#   vel  0 = ≤ −60 dB  (inaudible / silent)
#   each step = 4 dB

_VEL_DB_RANGE: float = 60.0   # dB range covered by vel 0–15


def _db_to_vel(db_attenuation: float) -> int:
    """Convert a positive dB attenuation to a 0–15 velocity.

    db_attenuation: 0.0 = loudest, positive = quieter.
    """
    if db_attenuation >= _VEL_DB_RANGE:
        return 0
    return max(0, min(15, int(15.0 * (1.0 - db_attenuation / _VEL_DB_RANGE) + 0.5)))


# ---------------------------------------------------------------------------
# Operator slot order in registers: physical slot 0=OP1, 1=OP3, 2=OP2, 3=OP4
# We reorder to logical OP1-OP4 for storage
_SLOT_TO_OP = (0, 2, 1, 3)  # phys_slot → logical_op_index

# Carrier operators per algorithm (logical indices 0=OP1 … 3=OP4)
# ALG 0-3: OP4 only; ALG 4: OP2+OP4; ALG 5-6: OP2+OP3+OP4; ALG 7: all
_CARRIERS: tuple[tuple[int, ...], ...] = (
    (3,),           # ALG 0
    (3,),           # ALG 1
    (3,),           # ALG 2
    (3,),           # ALG 3
    (1, 3),         # ALG 4
    (1, 2, 3),      # ALG 5
    (1, 2, 3),      # ALG 6
    (0, 1, 2, 3),   # ALG 7
)


def _carrier_ops(algorithm: int) -> tuple[int, ...]:
    """Return logical OP indices (0-3) that drive the audio output."""
    return _CARRIERS[algorithm & 0x07]


@dataclass(frozen=True)
class Ym2612Patch:
    """All operator parameters defining a YM2612 FM timbre."""
    algorithm: int          # 0–7: operator connection topology
    feedback: int           # 0–7: OP1 self-feedback amount

    # Per-operator parameters (index 0=OP1, 1=OP2, 2=OP3, 3=OP4)
    tl:  tuple[int, ...]    # Total Level 0–127 (0=loudest)
    ar:  tuple[int, ...]    # Attack Rate 0–31
    dr:  tuple[int, ...]    # Decay Rate 0–31
    sr:  tuple[int, ...]    # Sustain Rate 0–31
    rr:  tuple[int, ...]    # Release Rate 0–15
    sl:  tuple[int, ...]    # Sustain Level 0–15
    mul: tuple[int, ...]    # Multiple 0–15
    dt:  tuple[int, ...]    # Detune 0–7

    # Channel-level modulation
    ams: int = 0            # Amplitude modulation sensitivity 0–3
    fms: int = 0            # Frequency modulation sensitivity 0–7
    pan: int = 3            # Stereo output bits (bits 7:6 of reg 0xB4): 3=both, 2=R, 1=L

    # Per-operator AM enable (bit 7 of register 0x60)
    am_en: tuple[bool, ...] = (False, False, False, False)
    # Per-operator Key Scale / Rate Scaling (register 0x50, bits 7:6)
    # Controls how fast envelopes decay at higher pitches — critical for percussion.
    ks: tuple[int, ...] = (0, 0, 0, 0)
    # Per-operator SSG-EG (register 0x90, bits 3:0)
    ssg_eg: tuple[int, ...] = (0, 0, 0, 0)

    # Global LFO state (register 0x22) snapshotted at key-on time
    lfo_en:   bool = False
    lfo_rate: int  = 0     # 0–7

    # CH3 special-mode per-operator F-numbers (regs 0xA8–0xAE, port 0 only).
    # Indexed 0–2: slot 0=0xA8/0xAC, slot 1=0xA9/0xAD, slot 2=0xAA/0xAE.
    # Slot 3 (op4) uses the normal channel F-number (fnum_lo/fnum_hi/block).
    # Only meaningful for FM channel 3 (index 2); all zeros on other channels.
    ch3_mode:        int   = 0               # 0=normal, 2=special (reg 0x27 bits 7:6)
    ch3_op_fnum_lo:  tuple = (0, 0, 0)       # raw F-number low bytes
    ch3_op_fnum_hi:  tuple = (0, 0, 0)       # F-number high nibbles (bits 10:8)
    ch3_op_block:    tuple = (0, 0, 0)       # block values (3 bits each)

    def output_tl(self) -> int:
        """Total level of the output carrier operator(s).

        The output operator depends on the algorithm:
          ALG 0-3: OP4 only
          ALG 4:   OP2 and OP4
          ALG 5-6: OP2, OP3, OP4
          ALG 7:   all four operators
        """
        # Return minimum TL among carriers (loudest output)
        return min(self.tl[i] for i in _carrier_ops(self.algorithm))

    def to_fingerprint(self) -> tuple:
        """Compact tuple for hashing/equality checks.

        Includes all perceptually significant parameters so that patches
        differing only in envelope shape (DR/SR/RR/SL), multiplier or
        detune are treated as distinct entries rather than collapsed.
        """
        return (
            self.algorithm, self.feedback,
            self.tl, self.ar, self.dr, self.sr, self.rr, self.sl,
            self.mul, self.dt, self.ks,
            self.ams, self.fms,
            self.am_en, self.ssg_eg,
        )


@dataclass
class NoteEvent:
    """A single note on/off event decoded from VGM register writes."""
    channel: int            # CH_FM_0 … CH_PSG_NOISE (see constants above)
    pitch: int              # MIDI note number 0–127; -1 for noise/DAC
    velocity: int           # 0–15 (derived from output operator TL)
    sample_on: int          # absolute sample position of Key On
    sample_off: int = -1    # absolute sample position of Key Off (-1 = still on)
    patch: Ym2612Patch | None = None  # FM patch at time of Key On; None for PSG/DAC
    # For DAC channel events: the pcm_offset at which this onset was detected.
    # Used by the tokenizer to assign drum identity slots (kick/snare/etc.).
    # -1 for all non-DAC events.
    dac_sample_id: int = -1

    # For PSG tone channels: list of (absolute_sample, midi_pitch) waypoints
    # capturing mid-note period changes (vibrato, glides, pitch-bend).
    # Empty for most notes; non-empty when pitch changes while the note is held.
    pitch_envelope: list = field(default_factory=list)

    # For FM channels: list of (absolute_sample, [tl0, tl1, tl2, tl3]) snapshots
    # capturing mid-note TL (Total Level) register changes.  These are used by
    # game music composers to shape note decay/articulation without using key-off
    # (the "TL-fade envelope" technique).  Empty for most notes.
    tl_envelope: list = field(default_factory=list)

    @property
    def duration_samples(self) -> int:
        if self.sample_off < 0:
            return -1
        return max(0, self.sample_off - self.sample_on)

    @property
    def is_closed(self) -> bool:
        return self.sample_off >= 0


# ---------------------------------------------------------------------------
# YM2612 channel state
# ---------------------------------------------------------------------------

@dataclass
class _FmChannelState:
    """Register state for one YM2612 FM channel."""
    # Pitch
    fnum_lo: int = 0        # F-Number bits 7:0 (register 0xA0-0xA2)
    fnum_hi: int = 0        # F-Number bits 10:8 (register 0xA4-0xA6, bits 2:0)
    block: int = 0          # Block 0–7 (register 0xA4-0xA6, bits 5:3)

    # Patch data — operator arrays indexed by logical OP (0=OP1 … 3=OP4)
    algorithm: int = 0
    feedback: int = 0
    tl:  list = field(default_factory=lambda: [0] * 4)
    ar:  list = field(default_factory=lambda: [0] * 4)
    dr:  list = field(default_factory=lambda: [0] * 4)
    sr:  list = field(default_factory=lambda: [0] * 4)
    rr:  list = field(default_factory=lambda: [0] * 4)
    sl:  list = field(default_factory=lambda: [0] * 4)
    mul: list = field(default_factory=lambda: [0] * 4)
    dt:  list = field(default_factory=lambda: [0] * 4)
    ams: int = 0
    fms: int = 0
    lr:  int = 3            # L/R output enable bits 0b11=both, 0b10=R, 0b01=L
    am_en: list = field(default_factory=lambda: [False] * 4)
    ssg_eg: list = field(default_factory=lambda: [0] * 4)
    ks: list = field(default_factory=lambda: [0] * 4)

    # CH3 special-mode per-operator F-numbers (slot indices 0-2)
    # Written via regs 0xA8/0xA9/0xAA (lo) and 0xAC/0xAD/0xAE (block+hi)
    ch3_op_fnum_lo: list = field(default_factory=lambda: [0, 0, 0])
    ch3_op_fnum_hi: list = field(default_factory=lambda: [0, 0, 0])
    ch3_op_block:   list = field(default_factory=lambda: [0, 0, 0])

    # Key state
    key_on: bool = False
    key_on_sample: int = 0  # sample position when key turned on

    # Open note event (closed when key off is received)
    open_note: NoteEvent | None = None

    @property
    def fnum(self) -> int:
        return ((self.fnum_hi & 0x07) << 8) | (self.fnum_lo & 0xFF)

    def midi_pitch(self) -> int:
        """Convert current F-Number + Block to MIDI note number."""
        return fnumber_to_midi(self.fnum, self.block)

    def make_patch(self, lfo_en: bool = False, lfo_rate: int = 0, ch3_mode: int = 0) -> Ym2612Patch:
        return Ym2612Patch(
            algorithm=self.algorithm,
            feedback=self.feedback,
            tl=tuple(self.tl),
            ar=tuple(self.ar),
            dr=tuple(self.dr),
            sr=tuple(self.sr),
            rr=tuple(self.rr),
            sl=tuple(self.sl),
            mul=tuple(self.mul),
            dt=tuple(self.dt),
            ams=self.ams,
            fms=self.fms,
            pan=self.lr,
            am_en=tuple(self.am_en),
            ssg_eg=tuple(self.ssg_eg),
            ks=tuple(self.ks),
            lfo_en=lfo_en,
            lfo_rate=lfo_rate,
            ch3_mode=ch3_mode,
            ch3_op_fnum_lo=tuple(self.ch3_op_fnum_lo),
            ch3_op_fnum_hi=tuple(self.ch3_op_fnum_hi),
            ch3_op_block=tuple(self.ch3_op_block),
        )

    def velocity_from_patch(self) -> int:
        """Derive 0–15 velocity from the output carrier's Total Level.

        Uses the unified dB scale: TL step = 0.75 dB, range capped at 60 dB.
        """
        patch = self.make_patch()
        tl = patch.output_tl()
        return _db_to_vel(tl * 0.75)


# ---------------------------------------------------------------------------
# SN76489 channel state
# ---------------------------------------------------------------------------

@dataclass
class _PsgChannelState:
    """Register state for one SN76489 tone or noise channel."""
    period: int = 0     # 10-bit period register (tone channels) or noise control
    volume: int = 15    # 4-bit attenuation (15 = silence, 0 = max)
    key_on: bool = False
    key_on_sample: int = 0
    open_note: NoteEvent | None = None

    def psg_pitch(self, clock: int = SN76489_CLOCK) -> int:
        """Convert period register to MIDI note (tone channels only)."""
        if self.period == 0:
            return -1
        freq = clock / (32.0 * self.period)
        return freq_to_midi(freq)

    def psg_velocity(self) -> int:
        """Derive 0–15 velocity from PSG volume attenuation.

        PSG attenuation step = 2 dB; volume 15 = hardware silence.
        Uses the unified dB scale so PSG and FM velocities are comparable.
        """
        if self.volume >= 15:
            return 0
        return _db_to_vel(self.volume * 2.0)


# ---------------------------------------------------------------------------
# Helper math
# ---------------------------------------------------------------------------

def fnumber_to_midi(fnum: int, block: int) -> int:
    """Convert YM2612 F-Number + Block to MIDI note number.

    Formula:  freq_Hz = F_Number × YM2612_CLOCK / (144 × 2^(20 - block))
    MIDI:     note = 69 + 12 × log2(freq / 440)
    """
    if fnum == 0:
        return -1
    try:
        freq = fnum * YM2612_CLOCK / (144.0 * (1 << (20 - block)))
        if freq <= 0:
            return -1
        midi = 69 + 12 * math.log2(freq / 440.0)
        return round(midi)
    except (ValueError, OverflowError):
        return -1


def midi_to_fnumber(midi_note: int, block: int = 4) -> int:
    """Convert a MIDI note + block to the nearest YM2612 F-Number.

    Inverse of fnumber_to_midi.  Block 4 is a sensible default covering
    roughly the full piano range.
    """
    freq = 440.0 * (2.0 ** ((midi_note - 69) / 12.0))
    fnum = freq * 144.0 * (1 << (20 - block)) / YM2612_CLOCK
    return max(0, min(0x7FF, round(fnum)))


def freq_to_midi(freq: float) -> int:
    """Convert a frequency in Hz to the nearest MIDI note number."""
    if freq <= 0:
        return -1
    midi = 69 + 12 * math.log2(freq / 440.0)
    return round(midi)


def optimal_block(midi_note: int) -> int:
    """Choose the best YM2612 block for a given MIDI note.

    Selects the block that maximises the F-Number while keeping it within
    the 11-bit hardware limit (0–2047).  A larger F-Number gives finer pitch
    resolution.
    """
    freq = 440.0 * (2.0 ** ((midi_note - 69) / 12.0))
    best_block = 4
    best_fnum  = 0.0
    for block in range(8):
        fnum = freq * 144.0 * (1 << (20 - block)) / YM2612_CLOCK
        if 0 < fnum <= 2047 and fnum > best_fnum:
            best_fnum  = fnum
            best_block = block
    return best_block


# ---------------------------------------------------------------------------
# YM2612 state machine
# ---------------------------------------------------------------------------

class Ym2612State:
    """Stateful decoder for YM2612 register writes → NoteEvents.

    Usage::

        decoder = Ym2612State()
        note_events = list(decoder.process_vgm(vgm_file))
        patches_by_channel = decoder.last_patches
    """

    def __init__(self) -> None:
        self.channels: list[_FmChannelState] = [_FmChannelState() for _ in range(6)]
        self.dac_enabled: bool = False
        self.lfo_enabled: bool = False
        self.lfo_rate: int = 0
        self.ch3_mode: int = 0   # CH3 mode: 0=normal, 2=special (reg 0x27 bits 7:6)
        self._current_sample: int = 0
        # DAC onset tracking
        self._last_dac_sample_end: int = -ONSET_GAP_SAMPLES  # force first event as onset
        self._last_dac_pcm_offset: int = -1

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def process_vgm(self, vgm: VgmFile) -> Iterator[NoteEvent]:
        """Process all VGM events and yield NoteEvents as they are completed.

        Both NOTE_ON and NOTE_OFF events are yielded.  NOTE_ON events have
        ``sample_off == -1``; they are later closed by a corresponding
        NOTE_OFF event (same channel).

        Note: open notes at end-of-file are closed at the final sample.
        """
        self._current_sample = 0

        for event in vgm.events:
            if event.type == EventType.WAIT:
                self._current_sample += event.value

            elif event.type == EventType.YM2612_PORT0:
                yield from self._write_fm(event.register, event.value, port=0)

            elif event.type == EventType.YM2612_PORT1:
                yield from self._write_fm(event.register, event.value, port=1)

            elif event.type == EventType.DAC_WRITE:
                if self.dac_enabled:
                    gap = self._current_sample - self._last_dac_sample_end
                    # A sequential advance (offset == last_seen + 1) means the
                    # same sample is still streaming byte-by-byte; not a new hit.
                    # Any non-sequential jump (seek, loop restart, first byte) is
                    # a new onset.  This prevents each PCM byte from becoming its
                    # own token — the original intent of the offset check, which
                    # was broken because _last_dac_pcm_offset was only updated
                    # inside the `if is_new_onset` block, so consecutive bytes
                    # always read N != N-1 and always triggered.
                    is_sequential = (event.pcm_offset == self._last_dac_pcm_offset + 1)
                    is_new_onset = gap > ONSET_GAP_SAMPLES or not is_sequential
                    if is_new_onset:
                        yield NoteEvent(
                            channel=CH_DAC,
                            pitch=-1,
                            velocity=15,
                            sample_on=self._current_sample,
                            sample_off=self._current_sample + 735,  # ~1 frame
                            dac_sample_id=event.pcm_offset,
                        )
                    # Always update both trailing pointers so sequential detection
                    # and silence-gap detection both work correctly on the next byte.
                    self._last_dac_pcm_offset = event.pcm_offset
                    self._last_dac_sample_end = self._current_sample

            elif event.type == EventType.END:
                break

        # Close any notes still open at end of stream
        final = self._current_sample
        for ch_idx, ch in enumerate(self.channels):
            if ch.open_note is not None:
                note = ch.open_note
                note.sample_off = final
                ch.open_note = None
                ch.key_on = False
                yield note

    @property
    def last_patches(self) -> dict[int, Ym2612Patch]:
        """Return the most recent FM patch for each channel (0-5)."""
        return {
            i: ch.make_patch(
                lfo_en=self.lfo_enabled,
                lfo_rate=self.lfo_rate,
                ch3_mode=self.ch3_mode if i == 2 else 0,
            )
            for i, ch in enumerate(self.channels)
        }

    # ------------------------------------------------------------------
    # Internal register processing
    # ------------------------------------------------------------------

    def _write_fm(self, reg: int, val: int, port: int) -> Iterator[NoteEvent]:
        """Process one YM2612 register write (port 0 or 1)."""

        ch_offset = port * 3  # channels 0-2 on port 0, 3-5 on port 1

        # ---- Global registers (port 0 only) ----
        if port == 0:
            if reg == 0x22:
                self.lfo_enabled = bool(val & 0x08)
                self.lfo_rate = val & 0x07
                return
            if reg == 0x27:
                # CH3 mode: bits 7:6 (0=normal, 2=special/CSM)
                self.ch3_mode = (val >> 6) & 0x03
                return
            if reg == 0x2B:
                new_dac_en = bool(val & 0x80)
                if new_dac_en and not self.dac_enabled:
                    # DAC just enabled: FM6 output is replaced by DAC.
                    # Close any open FM6 note so it is not left unclosed.
                    ch5 = self.channels[CH_FM_5]
                    if ch5.open_note is not None:
                        note = ch5.open_note
                        note.sample_off = self._current_sample
                        ch5.open_note = None
                        ch5.key_on = False
                        yield note
                self.dac_enabled = new_dac_en
                return
            if reg == 0x28:
                yield from self._handle_key_on_off(val)
                return

        # ---- CH3 special-mode per-operator F-Numbers (port 0 only) ----
        # Regs 0xA8/0xA9/0xAA: per-operator F-number low byte (slots 0/1/2)
        # Regs 0xAC/0xAD/0xAE: per-operator block + F-number high (slots 0/1/2)
        # Slot 3 (OP4) uses the normal CH3 F-number regs 0xA2/0xA6.
        if port == 0 and 0xA8 <= reg <= 0xAA:
            self.channels[2].ch3_op_fnum_lo[reg - 0xA8] = val
            return
        if port == 0 and 0xAC <= reg <= 0xAE:
            slot_idx = reg - 0xAC
            self.channels[2].ch3_op_fnum_hi[slot_idx] = val & 0x07
            self.channels[2].ch3_op_block[slot_idx]   = (val >> 3) & 0x07
            return

        # ---- Per-channel F-Number (frequency) ----
        if 0xA0 <= reg <= 0xA2:
            ch = reg - 0xA0 + ch_offset
            if 0 <= ch < 6:
                self.channels[ch].fnum_lo = val
                # Record mid-note pitch bend waypoint for FM channels.
                # Writing 0xA0/0xA2 COMMITS the frequency (shadow hi+new lo).
                # If a note is already ringing, this is a portamento/glide step.
                # Store (sample, fnum, block) — 3-element to distinguish from
                # PSG 2-element (sample, period) tuples in pitch_envelope.
                ch_state = self.channels[ch]
                if ch_state.key_on and ch_state.open_note is not None:
                    fnum  = ch_state.fnum   # updated fnum_lo already applied above
                    block = ch_state.block
                    pe = ch_state.open_note.pitch_envelope
                    if not pe or pe[-1][1] != fnum or pe[-1][2] != block:
                        pe.append((self._current_sample, fnum, block))
            return

        if 0xA4 <= reg <= 0xA6:
            ch = reg - 0xA4 + ch_offset
            if 0 <= ch < 6:
                self.channels[ch].fnum_hi = val & 0x07
                self.channels[ch].block   = (val >> 3) & 0x07
            return

        # ---- Per-channel algorithm / feedback ----
        if 0xB0 <= reg <= 0xB2:
            ch = reg - 0xB0 + ch_offset
            if 0 <= ch < 6:
                self.channels[ch].algorithm = val & 0x07
                self.channels[ch].feedback  = (val >> 3) & 0x07
            return

        # ---- Per-channel LR/AMS/FMS ----
        if 0xB4 <= reg <= 0xB6:
            ch = reg - 0xB4 + ch_offset
            if 0 <= ch < 6:
                self.channels[ch].lr  = (val >> 6) & 0x03
                self.channels[ch].ams = (val >> 4) & 0x03
                self.channels[ch].fms = val & 0x07
            return

        # ---- Per-operator registers (0x30 – 0x9F) ----
        if 0x30 <= reg <= 0x9F:
            ch_idx = (reg & 0x03)
            if ch_idx == 3:
                return  # invalid / CH3 special mode sub-register
            ch = ch_idx + ch_offset
            if ch >= 6:
                return
            phys_slot = (reg >> 2) & 0x03
            op = _SLOT_TO_OP[phys_slot]     # logical OP index 0-3
            base = reg & 0xF0

            if base == 0x30:
                self.channels[ch].mul[op] = val & 0x0F
                self.channels[ch].dt[op]  = (val >> 4) & 0x07
            elif base == 0x40:
                self.channels[ch].tl[op] = val & 0x7F
                ch_state = self.channels[ch]
                if ch_state.key_on and ch_state.open_note is not None:
                    # Record mid-note TL snapshot for tl_envelope replay.
                    # Append whenever TL changes while a note is open; the synth
                    # will re-emit these writes at the correct sample positions.
                    tl_snap = list(ch_state.tl)  # copy all 4 ops
                    tl_env = ch_state.open_note.tl_envelope
                    if not tl_env or tl_env[-1][1] != tl_snap:
                        tl_env.append((self._current_sample, tl_snap))
                    # Synthetic Key Off: if all carrier operators are now at or
                    # above TL_SILENCE_THRESHOLD the voice is perceptually silent
                    # even though no 0x28 Key Off was issued (TL-fade technique).
                    carriers = _carrier_ops(ch_state.algorithm)
                    if all(ch_state.tl[c] >= TL_SILENCE_THRESHOLD
                           for c in carriers):
                        ch_state.key_on = False
                        note = ch_state.open_note
                        note.sample_off = self._current_sample
                        ch_state.open_note = None
                        yield note
            elif base == 0x50:
                self.channels[ch].ar[op] = val & 0x1F
                self.channels[ch].ks[op] = (val >> 6) & 0x03
            elif base == 0x60:
                self.channels[ch].dr[op]    = val & 0x1F
                self.channels[ch].am_en[op] = bool(val & 0x80)
            elif base == 0x70:
                self.channels[ch].sr[op] = val & 0x1F
            elif base == 0x80:
                self.channels[ch].rr[op] = val & 0x0F
                self.channels[ch].sl[op] = (val >> 4) & 0x0F
            elif base == 0x90:
                self.channels[ch].ssg_eg[op] = val & 0x0F

    def _handle_key_on_off(self, val: int) -> Iterator[NoteEvent]:
        """Process register 0x28 (Key On / Key Off)."""
        ch_bits  = val & 0x07
        op_bits  = (val >> 4) & 0x0F   # bit 0=OP1, 1=OP2, 2=OP3, 3=OP4
        key_on   = op_bits != 0

        # Map 0x28 channel encoding to channel index 0-5
        # 0→0, 1→1, 2→2, 3=invalid, 4→3, 5→4, 6→5, 7=invalid
        if ch_bits == 3 or ch_bits == 7:
            return
        ch = ch_bits if ch_bits < 3 else ch_bits - 1  # 4→3, 5→4, 6→5

        # FM6 (channel index 5 = CH_FM_5) is muted when DAC is enabled.
        # Key-on events produce no audio while dac_enabled is True.
        if key_on and ch == CH_FM_5 and self.dac_enabled:
            return

        ch_state = self.channels[ch]

        if key_on:
            # RE-TRIGGER: a Key On while the channel is already active closes
            # the current note immediately (arpeggio / drum-roll behaviour).
            # The YM2612 hardware resets the operator envelopes on every Key
            # On regardless of whether the channel was already keyed-on, so
            # each re-trigger is a genuinely new note event.
            if ch_state.key_on and ch_state.open_note is not None:
                old_note = ch_state.open_note
                old_note.sample_off = self._current_sample
                ch_state.open_note  = None
                ch_state.key_on     = False
                yield old_note

            # KEY ON: open a new note
            pitch = ch_state.midi_pitch()
            vel   = ch_state.velocity_from_patch()
            note  = NoteEvent(
                channel   = ch,
                pitch     = pitch,
                velocity  = vel,
                sample_on = self._current_sample,
                patch     = ch_state.make_patch(
                    lfo_en=self.lfo_enabled,
                    lfo_rate=self.lfo_rate,
                    ch3_mode=self.ch3_mode if ch == 2 else 0,
                ),
            )
            ch_state.key_on        = True
            ch_state.key_on_sample = self._current_sample
            ch_state.open_note     = note
            # Do NOT yield here — yield only when note is closed (key-off/retrigger/EOF)

        elif not key_on and ch_state.key_on:
            # KEY OFF: close the open note
            ch_state.key_on = False
            if ch_state.open_note is not None:
                note = ch_state.open_note
                note.sample_off = self._current_sample
                ch_state.open_note = None
                yield note


# ---------------------------------------------------------------------------
# SN76489 state machine
# ---------------------------------------------------------------------------

class Sn76489State:
    """Stateful decoder for SN76489 PSG register writes → NoteEvents.

    The SN76489 uses a latched write protocol:
      - Byte with bit 7 set: LATCH/DATA byte → sets channel + register type
      - Byte with bit 7 clear: DATA byte → extends the latched register value

    Tone channels produce pitched notes; the noise channel produces hits.
    """

    def __init__(self) -> None:
        # Channels 0-2: tone, channel 3: noise
        self.channels = [_PsgChannelState() for _ in range(4)]
        self._latch_channel: int = 0
        self._latch_type: int = 0    # 0=tone/noise, 1=volume
        self._current_sample: int = 0

    def process_vgm(self, vgm: VgmFile) -> Iterator[NoteEvent]:
        """Process VGM events and yield PSG NoteEvents."""
        self._current_sample = 0

        for event in vgm.events:
            if event.type == EventType.WAIT:
                self._current_sample += event.value
            elif event.type == EventType.SN76489:
                yield from self._write(event.value)
            elif event.type == EventType.END:
                break

        # Close open notes
        final = self._current_sample
        for ch_idx, ch in enumerate(self.channels):
            if ch.open_note is not None:
                ch.open_note.sample_off = final
                yield ch.open_note
                ch.open_note = None

    def _write(self, byte: int) -> Iterator[NoteEvent]:
        if byte & 0x80:
            # LATCH/DATA byte
            self._latch_channel = (byte >> 5) & 0x03
            self._latch_type    = (byte >> 4) & 0x01
            data4               = byte & 0x0F

            ch = self.channels[self._latch_channel]
            if self._latch_type == 1:
                # Volume write
                new_vol = data4 & 0x0F
                yield from self._update_volume(self._latch_channel, ch, new_vol)
            else:
                # Tone/noise frequency write (low 4 bits)
                if self._latch_channel < 3:
                    ch.period = (ch.period & 0x3F0) | data4
                    # Pitch waypoint captured on DATA byte (high bits) arrival,
                    # where frequency is fully determined. Skip here since
                    # the high byte always follows and will record the waypoint.
                else:
                    # Noise channel: bits [1:0] = shift rate, bit 2 = type
                    ch.period = data4 & 0x0F
        else:
            # DATA byte — only applies to tone channels
            if self._latch_type == 0 and self._latch_channel < 3:
                ch = self.channels[self._latch_channel]
                ch.period = ((byte & 0x3F) << 4) | (ch.period & 0x0F)
                # If a note is open, record a raw-period waypoint.
                # We store the raw SN76489 period register value (not MIDI) so
                # that sub-semitone vibrato (which rounds to the same MIDI note)
                # is faithfully preserved through the round-trip.
                if ch.open_note is not None:
                    new_pitch = ch.psg_pitch()
                    if new_pitch >= 0 and new_pitch != ch.open_note.pitch:
                        # Pitch changed while note is held → close old note, open new one
                        ch.open_note.sample_off = self._current_sample
                        yield ch.open_note
                        ch_id = CH_PSG_0 + self._latch_channel
                        ch.open_note = NoteEvent(
                            channel   = ch_id,
                            pitch     = new_pitch,
                            velocity  = ch.psg_velocity(),
                            sample_on = self._current_sample,
                        )
                    else:
                        new_period = ch.period
                        if not ch.open_note.pitch_envelope or ch.open_note.pitch_envelope[-1][1] != new_period:
                            ch.open_note.pitch_envelope.append((self._current_sample, new_period))

    def _update_volume(
        self, ch_idx: int, ch: _PsgChannelState, new_vol: int
    ) -> Iterator[NoteEvent]:
        """Handle PSG volume change — treat as note on/off events."""
        was_silent = ch.volume == 15
        now_silent = new_vol == 15
        ch.volume  = new_vol

        if was_silent and not now_silent:
            # Note ON
            if ch_idx < 3:
                pitch = ch.psg_pitch()
            else:
                pitch = -1  # noise channel

            ch_id = CH_PSG_0 + ch_idx if ch_idx < 3 else CH_PSG_NOISE
            note = NoteEvent(
                channel   = ch_id,
                pitch     = pitch,
                velocity  = ch.psg_velocity(),
                sample_on = self._current_sample,
            )
            ch.key_on       = True
            ch.open_note    = note
            # Do NOT yield here — yield only when note is closed

        elif not was_silent and now_silent:
            # Note OFF
            ch.key_on = False
            if ch.open_note is not None:
                ch.open_note.sample_off = self._current_sample
                yield ch.open_note
                ch.open_note = None


# ---------------------------------------------------------------------------
# Unified decoder
# ---------------------------------------------------------------------------

def decode_vgm(
    vgm: VgmFile,
) -> tuple[list[NoteEvent], dict[int, Ym2612Patch]]:
    """Decode a VGM file into NoteEvents and the FM patch map.

    Returns
    -------
    note_events : list[NoteEvent]
        All note on/off events sorted by sample_on, then channel.
    patch_map : dict[int, Ym2612Patch]
        Most recent patch observed for each FM channel index (0–5).
    """
    fm_decoder  = Ym2612State()
    psg_decoder = Sn76489State()

    fm_events  = list(fm_decoder.process_vgm(vgm))
    psg_events = list(psg_decoder.process_vgm(vgm))

    all_events = sorted(
        fm_events + psg_events,
        key=lambda e: (e.sample_on, e.channel),
    )

    return all_events, fm_decoder.last_patches
