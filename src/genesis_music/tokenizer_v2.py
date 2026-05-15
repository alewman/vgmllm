"""v2 Tokenizer for VGM event sequences.

Key improvement over v1: YM2612 frequency register writes and Key On/Off
events are abstracted to musical note tokens (e.g. CH1:ON:C4, CH3:OFF)
while all other register writes stay raw (preserving mid-note timbral
manipulation, volume envelopes, LFO, etc.).

This lets the model learn musical patterns (scales, intervals) from note
names rather than opaque frequency register bytes, without sacrificing
the YM2612's expressive per-frame register control like GEMS did.

Design:
    - Freq MSB writes (0xA4-0xA6): update state, suppress token
    - Freq LSB writes (0xA0-0xA2): update state, suppress if note off;
      emit PITCH token if note is on and pitch changed
    - Key On (reg 0x28, ops != 0): emit CH{n}:ON:{note}
    - Key Off (reg 0x28, ops == 0): emit CH{n}:OFF
    - All other register writes: raw tokens (FM0/FM1/PSG prefix)
    - Wait tokens: same log-spaced bins as v1
    - SN76489: raw passthrough
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections import Counter
from dataclasses import dataclass, field
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Sequence

import numpy as np

from .vgm_parser import EventType, VgmEvent, VgmFile, load_vgm

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Special token IDs
# ---------------------------------------------------------------------------
PAD = 0
BOS = 1
EOS = 2
UNK = 3

SPECIAL_TOKENS = {
    "<PAD>": PAD,
    "<BOS>": BOS,
    "<EOS>": EOS,
    "<UNK>": UNK,
}

# ---------------------------------------------------------------------------
# Note names
# ---------------------------------------------------------------------------
NOTE_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

# Reverse lookup: name -> semitone (0-11)
_NAME_TO_SEMI = {name: i for i, name in enumerate(NOTE_NAMES)}


def midi_to_name(midi_note: int) -> str:
    """Convert MIDI note number to name like 'C4', 'Db5'."""
    octave = (midi_note // 12) - 1
    semi = midi_note % 12
    return f"{NOTE_NAMES[semi]}{octave}"


def name_to_midi(name: str) -> int:
    """Convert note name like 'C4', 'Db5' to MIDI note number."""
    # Parse: note part is letters, octave is trailing digits (possibly negative)
    i = 0
    while i < len(name) and not (name[i].isdigit() or name[i] == '-'):
        i += 1
    note_part = name[:i]
    octave = int(name[i:])
    semi = _NAME_TO_SEMI[note_part]
    return (octave + 1) * 12 + semi


# ---------------------------------------------------------------------------
# YM2612 frequency <-> MIDI note conversion
# ---------------------------------------------------------------------------
# YM2612 frequency formula:
#   freq = (f_number * clock) / (144 * 2^(21 - block))
# Standard NTSC clock: 7670453 Hz
#
# We precompute a lookup from (block, f_number) -> nearest MIDI note,
# and the reverse: MIDI note -> (block, f_number).

YM2612_CLOCK = 7670453


def _fnum_block_to_freq(f_number: int, block: int) -> float:
    """Convert YM2612 F-number + block to frequency in Hz."""
    if f_number == 0:
        return 0.0
    return (f_number * YM2612_CLOCK) / (144 * (1 << (21 - block)))


def _freq_to_midi(freq: float) -> int:
    """Convert frequency in Hz to nearest MIDI note number."""
    if freq <= 0:
        return 0
    return round(69 + 12 * math.log2(freq / 440.0))


def _midi_to_fnum_block(midi_note: int) -> tuple[int, int]:
    """Convert MIDI note to best (f_number, block) pair for YM2612."""
    freq = 440.0 * (2.0 ** ((midi_note - 69) / 12.0))
    # Try each block, pick the one that gives f_number closest to center range
    best_block = 0
    best_fnum = 0
    best_err = float("inf")
    for block in range(8):
        # f_number = freq * 144 * 2^(21-block) / clock
        fnum = freq * 144 * (1 << (21 - block)) / YM2612_CLOCK
        fnum_rounded = round(fnum)
        if fnum_rounded < 1 or fnum_rounded > 2047:  # 11-bit max
            continue
        # Check reconstruction error
        reconstructed = _fnum_block_to_freq(fnum_rounded, block)
        err = abs(reconstructed - freq)
        # Prefer f_numbers in the middle range (better resolution)
        if err < best_err:
            best_err = err
            best_block = block
            best_fnum = fnum_rounded
    return best_fnum, best_block


def fnum_block_to_midi(f_number: int, block: int) -> int:
    """Convert YM2612 F-number + block to nearest MIDI note."""
    freq = _fnum_block_to_freq(f_number, block)
    return _freq_to_midi(freq)


def fnum_block_to_note(f_number: int, block: int) -> str:
    """Convert YM2612 F-number + block to note name like 'C4'."""
    midi = fnum_block_to_midi(f_number, block)
    return midi_to_name(midi)


# ---------------------------------------------------------------------------
# YM2612 register helpers
# ---------------------------------------------------------------------------
# Key On register (port 0 only, reg 0x28):
#   Bits 0-2: channel (0-2 = port0 ch1-3, 4-6 = port1 ch4-6)
#   Bits 4-7: operator enable (0 = all off = key off)

def parse_key_on(value: int) -> tuple[int, bool]:
    """Parse Key On register value -> (channel_1based, is_key_on).

    Returns channel number 1-6 and whether any operators are keyed on.
    """
    raw_ch = value & 0x07
    ops = (value >> 4) & 0x0F
    # Map: 0->1, 1->2, 2->3, 4->4, 5->5, 6->6
    if raw_ch <= 2:
        ch = raw_ch + 1
    elif 4 <= raw_ch <= 6:
        ch = raw_ch
    else:
        ch = 0  # invalid
    return ch, ops != 0


def freq_reg_to_channel(register: int, port: int) -> int:
    """Map frequency register + port to channel number 1-6."""
    # 0xA0/0xA4 -> ch offset 0
    # 0xA1/0xA5 -> ch offset 1
    # 0xA2/0xA6 -> ch offset 2
    ch_offset = register & 0x03
    if ch_offset > 2:
        return 0  # invalid (0xA3/0xA7 are Ch3 special mode)
    base = 1 if port == 0 else 4
    return base + ch_offset


# ---------------------------------------------------------------------------
# Channel state machine for v2 encoding
# ---------------------------------------------------------------------------

@dataclass
class _ChannelState:
    """Per-channel state for the frequency -> note abstraction."""
    freq_msb: int = 0       # Last written value to 0xA4+offset
    freq_lsb: int = 0       # Last written value to 0xA0+offset
    is_on: bool = False      # Whether key is currently on
    current_midi: int = 0    # Current MIDI note (from latest freq writes)

    @property
    def f_number(self) -> int:
        """11-bit F-number from MSB/LSB registers."""
        return ((self.freq_msb & 0x07) << 8) | self.freq_lsb

    @property
    def block(self) -> int:
        """3-bit block (octave) from MSB register."""
        return (self.freq_msb >> 3) & 0x07

    def update_midi(self) -> int:
        """Recompute MIDI note from current freq registers. Returns it."""
        if self.f_number == 0:
            self.current_midi = 0
        else:
            self.current_midi = fnum_block_to_midi(self.f_number, self.block)
        return self.current_midi


# ---------------------------------------------------------------------------
# Wait-time quantization (same as v1)
# ---------------------------------------------------------------------------

def _build_wait_bins(n_bins: int = 64, max_samples: int = 3_000_000) -> np.ndarray:
    bins = np.unique(
        np.round(np.geomspace(1, max_samples, n_bins)).astype(np.int64)
    )
    return bins


def quantize_wait(samples: int, bins: np.ndarray) -> int:
    idx = int(np.searchsorted(bins, samples))
    if idx >= len(bins):
        return len(bins) - 1
    if idx > 0:
        if abs(int(bins[idx]) - samples) > abs(int(bins[idx - 1]) - samples):
            return idx - 1
    return idx


def dequantize_wait(bin_idx: int, bins: np.ndarray) -> int:
    return int(bins[min(bin_idx, len(bins) - 1)])


# ---------------------------------------------------------------------------
# v2 token string formats
# ---------------------------------------------------------------------------
# Note events:
#   CH{1-6}:ON:{note}     e.g. CH1:ON:C4
#   CH{1-6}:OFF           e.g. CH3:OFF
#   CH{1-6}:PITCH:{note}  e.g. CH2:PITCH:D4 (mid-note pitch change >= 1 semi)
#
# Raw register writes (everything except freq and key on/off):
#   FM0:{reg:02X}:{val:02X}   (YM2612 port 0)
#   FM1:{reg:02X}:{val:02X}   (YM2612 port 1)
#   PSG:{val:02X}              (SN76489)
#
# Wait tokens:
#   <WAIT:{bin_idx}>
#
# Special:
#   <PAD>, <BOS>, <EOS>, <UNK>

# Registers to suppress (absorbed into note tokens):
_FREQ_MSB_REGS = {0xA4, 0xA5, 0xA6}
_FREQ_LSB_REGS = {0xA0, 0xA1, 0xA2}
_KEY_ON_REG = 0x28


def _is_freq_reg(register: int) -> bool:
    return register in _FREQ_MSB_REGS or register in _FREQ_LSB_REGS


# ---------------------------------------------------------------------------
# Encode: VGM events -> v2 token strings
# ---------------------------------------------------------------------------

def encode_events_v2(
    events: list[VgmEvent],
    *,
    include_dac: bool = False,
) -> list[str]:
    """Convert VGM events to v2 token strings.

    This is the string-level encoding (before vocab ID mapping).
    Returns a list of token strings like ['CH1:ON:C4', 'FM0:30:71', '<WAIT:5>'].
    """
    channels: dict[int, _ChannelState] = {ch: _ChannelState() for ch in range(1, 7)}
    tokens: list[str] = []
    wait_bins = _build_wait_bins()

    pending_wait = 0

    def flush_wait():
        nonlocal pending_wait
        if pending_wait > 0:
            idx = quantize_wait(pending_wait, wait_bins)
            tokens.append(f"<WAIT:{idx}>")
            pending_wait = 0

    for ev in events:
        if ev.type == EventType.END:
            break

        if ev.type == EventType.DAC_WRITE and not include_dac:
            continue

        if ev.type == EventType.WAIT:
            pending_wait += ev.value
            continue

        # --- Flush wait before any non-wait event ---
        flush_wait()

        if ev.type == EventType.SN76489:
            tokens.append(f"PSG:{ev.register:02X}")
            continue

        if ev.type == EventType.DAC_WRITE:
            tokens.append(f"FM0:2A:{ev.value:02X}")
            continue

        # YM2612 register write
        port = 0 if ev.type == EventType.YM2612_PORT0 else 1
        reg = ev.register
        val = ev.value

        # --- Key On/Off (always port 0, reg 0x28) ---
        if port == 0 and reg == _KEY_ON_REG:
            ch, is_on = parse_key_on(val)
            if ch == 0:
                # Invalid channel, emit raw
                tokens.append(f"FM0:{reg:02X}:{val:02X}")
                continue

            state = channels[ch]
            if is_on:
                state.is_on = True
                state.update_midi()
                note = midi_to_name(state.current_midi) if state.current_midi > 0 else "X"
                tokens.append(f"CH{ch}:ON:{note}")
            else:
                state.is_on = False
                tokens.append(f"CH{ch}:OFF")
            continue

        # --- Frequency MSB (0xA4-0xA6) ---
        if reg in _FREQ_MSB_REGS:
            ch = freq_reg_to_channel(reg, port)
            if ch == 0:
                tokens.append(f"FM{port}:{reg:02X}:{val:02X}")
                continue
            channels[ch].freq_msb = val
            # MSB write is always suppressed (absorbed into next ON or PITCH)
            continue

        # --- Frequency LSB (0xA0-0xA2) ---
        if reg in _FREQ_LSB_REGS:
            ch = freq_reg_to_channel(reg, port)
            if ch == 0:
                tokens.append(f"FM{port}:{reg:02X}:{val:02X}")
                continue
            state = channels[ch]
            state.freq_lsb = val
            if state.is_on:
                old_midi = state.current_midi
                new_midi = state.update_midi()
                if new_midi != old_midi and new_midi > 0:
                    tokens.append(f"CH{ch}:PITCH:{midi_to_name(new_midi)}")
            else:
                state.update_midi()
            continue

        # --- All other registers: pass through raw ---
        tokens.append(f"FM{port}:{reg:02X}:{val:02X}")

    # Flush trailing wait
    flush_wait()

    return tokens


# ---------------------------------------------------------------------------
# Decode: v2 token strings -> VGM events
# ---------------------------------------------------------------------------

def decode_token_str_v2(token: str) -> list[VgmEvent]:
    """Decode a single v2 token string back to VGM event(s).

    Note tokens expand to multiple register writes (freq + key on/off).
    Raw register tokens map 1:1.
    Returns empty list for special tokens.
    """
    if token.startswith("<") and token.endswith(">"):
        # Special or wait token
        if token.startswith("<WAIT:"):
            bins = _build_wait_bins()
            idx = int(token[6:-1])
            samples = dequantize_wait(idx, bins)
            return [VgmEvent(type=EventType.WAIT, value=samples)]
        return []  # PAD, BOS, EOS, UNK

    # PSG token: PSG:XX
    if token.startswith("PSG:"):
        val = int(token[4:], 16)
        return [VgmEvent(type=EventType.SN76489, register=val, value=0)]

    # Note ON: CH{n}:ON:{note}
    if ":ON:" in token:
        parts = token.split(":")
        ch = int(parts[0][2:])  # CH1 -> 1
        note_name = parts[2]
        if note_name == "X":
            # Unknown pitch, just do key on with whatever
            midi = 60  # default to C4
        else:
            midi = name_to_midi(note_name)
        return _note_on_events(ch, midi)

    # Note OFF: CH{n}:OFF
    if ":OFF" in token and ":PITCH:" not in token:
        parts = token.split(":")
        ch = int(parts[0][2:])
        return _note_off_events(ch)

    # Pitch change: CH{n}:PITCH:{note}
    if ":PITCH:" in token:
        parts = token.split(":")
        ch = int(parts[0][2:])
        note_name = parts[2]
        midi = name_to_midi(note_name)
        return _pitch_change_events(ch, midi)

    # Raw FM register: FM{0|1}:{reg}:{val}
    if token.startswith("FM"):
        parts = token.split(":")
        port = int(parts[0][2:])
        reg = int(parts[1], 16)
        val = int(parts[2], 16)
        etype = EventType.YM2612_PORT0 if port == 0 else EventType.YM2612_PORT1
        return [VgmEvent(type=etype, register=reg, value=val)]

    return []


def _note_on_events(ch: int, midi_note: int) -> list[VgmEvent]:
    """Generate VGM events for a note-on: freq MSB, freq LSB, key on."""
    fnum, block = _midi_to_fnum_block(midi_note)
    freq_msb = ((block & 0x07) << 3) | ((fnum >> 8) & 0x07)
    freq_lsb = fnum & 0xFF

    # Determine port and register offset
    if ch <= 3:
        port = 0
        ch_offset = ch - 1
    else:
        port = 1
        ch_offset = ch - 4
    etype = EventType.YM2612_PORT0 if port == 0 else EventType.YM2612_PORT1

    # Key on channel mapping for reg 0x28
    raw_ch = ch_offset if ch <= 3 else ch_offset + 4

    events = [
        # Freq MSB first (latches)
        VgmEvent(type=etype, register=0xA4 + ch_offset, value=freq_msb),
        # Freq LSB
        VgmEvent(type=etype, register=0xA0 + ch_offset, value=freq_lsb),
        # Key On (all 4 operators)
        VgmEvent(type=EventType.YM2612_PORT0, register=0x28, value=0xF0 | raw_ch),
    ]
    return events


def _note_off_events(ch: int) -> list[VgmEvent]:
    """Generate VGM event for key off."""
    if ch <= 3:
        raw_ch = ch - 1
    else:
        raw_ch = ch
    return [VgmEvent(type=EventType.YM2612_PORT0, register=0x28, value=raw_ch)]


def _pitch_change_events(ch: int, midi_note: int) -> list[VgmEvent]:
    """Generate VGM events for mid-note pitch change (no key on/off)."""
    fnum, block = _midi_to_fnum_block(midi_note)
    freq_msb = ((block & 0x07) << 3) | ((fnum >> 8) & 0x07)
    freq_lsb = fnum & 0xFF

    if ch <= 3:
        port = 0
        ch_offset = ch - 1
    else:
        port = 1
        ch_offset = ch - 4
    etype = EventType.YM2612_PORT0 if port == 0 else EventType.YM2612_PORT1

    return [
        VgmEvent(type=etype, register=0xA4 + ch_offset, value=freq_msb),
        VgmEvent(type=etype, register=0xA0 + ch_offset, value=freq_lsb),
    ]


# ---------------------------------------------------------------------------
# Full sequence decode
# ---------------------------------------------------------------------------

def decode_tokens_v2(token_strs: list[str]) -> list[VgmEvent]:
    """Decode a list of v2 token strings back to VGM events."""
    events = []
    for tok in token_strs:
        events.extend(decode_token_str_v2(tok))
    events.append(VgmEvent(type=EventType.END))
    return events


# ---------------------------------------------------------------------------
# Vocabulary (v2)
# ---------------------------------------------------------------------------

@dataclass
class VocabV2:
    """v2 token vocabulary with note-level abstraction."""
    wait_bins: np.ndarray
    token_to_id: dict[str, int] = field(default_factory=dict)
    id_to_token: dict[int, str] = field(default_factory=dict)
    wait_offset: int = 0
    event_offset: int = 0

    # Cache for the pre-built wait bins used during decode
    _decode_wait_bins: np.ndarray | None = field(default=None, repr=False)

    @property
    def size(self) -> int:
        return len(self.token_to_id)

    @property
    def n_wait_tokens(self) -> int:
        return len(self.wait_bins)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        obj = {
            "version": 2,
            "wait_bins": self.wait_bins.tolist(),
            "token_to_id": self.token_to_id,
            "wait_offset": self.wait_offset,
            "event_offset": self.event_offset,
        }
        path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> VocabV2:
        path = Path(path)
        obj = json.loads(path.read_text(encoding="utf-8"))
        bins = np.array(obj["wait_bins"], dtype=np.int64)
        v = cls(wait_bins=bins)
        v.token_to_id = obj["token_to_id"]
        v.id_to_token = {int(i): t for t, i in v.token_to_id.items()}
        v.wait_offset = obj["wait_offset"]
        v.event_offset = obj["event_offset"]
        return v

    def encode(self, token_str: str) -> int:
        """Map a v2 token string to its integer ID. Returns UNK if unknown."""
        return self.token_to_id.get(token_str, UNK)

    def decode(self, token_id: int) -> str:
        """Map an integer ID back to its token string."""
        return self.id_to_token.get(token_id, "<UNK>")


# ---------------------------------------------------------------------------
# Multiprocessing worker for v2 vocab building
# ---------------------------------------------------------------------------

def _extract_v2_tokens_from_file(
    fpath: Path | str,
    *,
    include_dac: bool,
) -> dict[str, int] | None:
    """Worker: load one VGM and return counter of v2 token strings."""
    try:
        vgm = load_vgm(fpath)
    except Exception:
        return None

    token_strs = encode_events_v2(vgm.events, include_dac=include_dac)
    counts: dict[str, int] = {}
    for t in token_strs:
        if t.startswith("<WAIT:"):
            continue  # wait tokens are fixed, not data-driven
        counts[t] = counts.get(t, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Vocab builder
# ---------------------------------------------------------------------------

def build_vocab_v2(
    vgm_files: Sequence[Path | str],
    n_wait_bins: int = 64,
    min_count: int = 2,
    include_dac: bool = False,
    max_tokens: int | None = None,
) -> VocabV2:
    """Build v2 vocabulary from corpus.

    Scans files, encodes to v2 token strings, counts frequencies,
    and assigns integer IDs.
    """
    event_counts: Counter[str] = Counter()
    n_workers = max(1, min(os.cpu_count() or 1, 12))
    log.info("  Building v2 vocab with %d workers", n_workers)

    worker = partial(_extract_v2_tokens_from_file, include_dac=include_dac)
    done = 0

    with Pool(n_workers) as pool:
        for file_counts in pool.imap_unordered(worker, vgm_files, chunksize=64):
            done += 1
            if file_counts is not None:
                for tok_str, cnt in file_counts.items():
                    event_counts[tok_str] += cnt
            if done % 500 == 0:
                log.info("  Scanned %d/%d files, %d unique v2 tokens so far",
                         done, len(vgm_files), len(event_counts))

    # Filter by min_count
    filtered = {k: c for k, c in event_counts.items() if c >= min_count}
    log.info("v2 corpus tokens: %d total unique, %d after min_count=%d",
             len(event_counts), len(filtered), min_count)

    # Count note tokens vs raw tokens
    note_tokens = {k for k in filtered if ":ON:" in k or ":OFF" in k or ":PITCH:" in k}
    raw_tokens = {k for k in filtered if k not in note_tokens}
    log.info("  Note tokens: %d, Raw register tokens: %d",
             len(note_tokens), len(raw_tokens))

    sorted_tokens = sorted(filtered, key=lambda k: filtered[k], reverse=True)
    if max_tokens is not None and len(sorted_tokens) > max_tokens:
        sorted_tokens = sorted_tokens[:max_tokens]

    # Build wait bins
    wait_bins = _build_wait_bins(n_wait_bins)

    # Assign IDs
    token_to_id: dict[str, int] = {}
    id_to_token: dict[int, str] = {}

    # Special tokens
    for name, tid in SPECIAL_TOKENS.items():
        token_to_id[name] = tid
        id_to_token[tid] = name

    # Wait tokens
    wait_offset = len(SPECIAL_TOKENS)
    for i in range(len(wait_bins)):
        name = f"<WAIT:{i}>"
        tid = wait_offset + i
        token_to_id[name] = tid
        id_to_token[tid] = name

    # Event tokens (note + raw combined, sorted by frequency)
    event_offset = wait_offset + len(wait_bins)
    for i, tok_str in enumerate(sorted_tokens):
        tid = event_offset + i
        token_to_id[tok_str] = tid
        id_to_token[tid] = tok_str

    vocab = VocabV2(
        wait_bins=wait_bins,
        token_to_id=token_to_id,
        id_to_token=id_to_token,
        wait_offset=wait_offset,
        event_offset=event_offset,
    )

    log.info("v2 vocab: %d total (%d special + %d wait + %d event [%d note + %d raw])",
             vocab.size, len(SPECIAL_TOKENS), len(wait_bins), len(sorted_tokens),
             len(note_tokens), len(raw_tokens))

    return vocab


# ---------------------------------------------------------------------------
# Full encode/decode with vocab (integer IDs)
# ---------------------------------------------------------------------------

def encode_vgm_v2(
    vgm: VgmFile,
    vocab: VocabV2,
    *,
    include_dac: bool = False,
) -> list[int]:
    """Encode a VGM file to v2 integer token IDs."""
    token_strs = encode_events_v2(vgm.events, include_dac=include_dac)
    ids = [BOS]
    for t in token_strs:
        tid = vocab.encode(t)
        if tid != UNK:
            ids.append(tid)
    ids.append(EOS)
    return ids


def decode_ids_v2(token_ids: list[int], vocab: VocabV2) -> list[VgmEvent]:
    """Decode v2 integer token IDs back to VGM events."""
    token_strs = []
    for tid in token_ids:
        s = vocab.decode(tid)
        if s not in ("<PAD>", "<BOS>", "<EOS>", "<UNK>"):
            token_strs.append(s)
    return decode_tokens_v2(token_strs)
