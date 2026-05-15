"""VGM synthesis — convert decoded NoteEvents back into a VGM binary.

This is the inverse of ym2612.py: given a list of NoteEvent objects (with
MIDI pitches, velocities, FM patches) it emits the corresponding YM2612 /
SN76489 register writes and wait commands, then packages them into a valid
VGM 1.61 binary that can be played by any VGM-compatible player.

Typical usage::

    from genesis_music.ym2612 import decode_vgm
    from genesis_music.vgm_parser import load_vgm
    from genesis_music.vgm_synth import synthesise_vgm

    src = load_vgm("track.vgm")
    notes, patches = decode_vgm(src)
    vgm_bytes = synthesise_vgm(notes, src.header.total_samples, patches)
    with open("reconstructed.vgm", "wb") as f:
        f.write(vgm_bytes)
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Sequence

from .ym2612 import (
    CH_DAC, CH_FM_0, CH_PSG_0, CH_PSG_1, CH_PSG_2, CH_PSG_NOISE,
    NoteEvent, Ym2612Patch,
    YM2612_CLOCK, SN76489_CLOCK,
    midi_to_fnumber, optimal_block,
)

# ---------------------------------------------------------------------------
# VGM command bytes
# ---------------------------------------------------------------------------

_CMD_SN76489  = 0x50
_CMD_YM2612_0 = 0x52   # port 0  (channels 0-2)
_CMD_YM2612_1 = 0x53   # port 1  (channels 3-5)
_CMD_WAIT_N   = 0x61   # followed by uint16-LE N samples
_CMD_WAIT_735 = 0x62   # exactly 735 samples (1/60 s NTSC frame)
_CMD_WAIT_882 = 0x63   # exactly 882 samples (1/50 s PAL frame)
_CMD_WAIT_1   = 0x70   # 0x7n = wait (n+1) samples,  n in 0-15
_CMD_DAC_SEEK = 0xE0   # seek PCM data position (4-byte offset)
_CMD_END      = 0x66

# Logical OP index → physical slot (reverses ym2612.py _SLOT_TO_OP = (0,2,1,3))
_OP_TO_SLOT = (0, 2, 1, 3)  # op0→slot0, op1→slot2, op2→slot1, op3→slot3

# YM2612 register 0x28 channel-ID encoding
# ch index 0-2 → 0-2,  ch index 3-5 → 4-6  (skips 3)
_CH_TO_KEY_BITS = (0, 1, 2, 4, 5, 6)


# ---------------------------------------------------------------------------
# Internal event types used during synthesis planning
# ---------------------------------------------------------------------------

@dataclass(order=True)
class _Action:
    """A register write or DAC/PSG event at a specific sample position."""
    sample: int
    priority: int       # lower number = emit first (patch before note-on)
    kind: str = field(compare=False)
    data: object = field(compare=False)


# ---------------------------------------------------------------------------
# Low-level VGM command emitters
# ---------------------------------------------------------------------------

class _VgmStream:
    """Accumulates raw VGM command bytes with wait management."""

    def __init__(self) -> None:
        self._buf: list[bytes] = []
        self._pos: int = 0      # current sample position in stream

    # ---- waits --------------------------------------------------------

    def advance_to(self, target: int) -> None:
        """Emit wait commands to advance the sample counter to *target*."""
        remaining = max(0, target - self._pos)
        while remaining > 0:
            if remaining >= 882:
                self._buf.append(bytes([_CMD_WAIT_882]))
                remaining -= 882
                self._pos  += 882
            elif remaining >= 735:
                self._buf.append(bytes([_CMD_WAIT_735]))
                remaining -= 735
                self._pos  += 735
            elif remaining <= 16:
                self._buf.append(bytes([_CMD_WAIT_1 + remaining - 1]))
                self._pos += remaining
                remaining  = 0
            else:
                n = min(remaining, 0xFFFF)
                self._buf.append(struct.pack("<BH", _CMD_WAIT_N, n))
                self._pos  += n
                remaining  -= n

    # ---- register writes -----------------------------------------------

    def write_ym2612(self, port: int, reg: int, val: int) -> None:
        cmd = _CMD_YM2612_0 if port == 0 else _CMD_YM2612_1
        self._buf.append(bytes([cmd, reg & 0xFF, val & 0xFF]))

    def write_psg(self, data: int) -> None:
        self._buf.append(bytes([_CMD_SN76489, data & 0xFF]))

    def write_dac(self, pcm_byte: int = 0x80) -> None:
        """Emit a single DAC write + wait-0 (0x80 command)."""
        self._buf.append(bytes([0x80]))

    def write_dac_byte(self, n: int) -> None:
        """Emit a 0x8n DAC write command (write PCM bank byte + wait n extra samples)."""
        self._buf.append(bytes([0x80 | (n & 0x0F)]))
        self._pos += n  # account for the n extra samples

    def write_dac_seek(self, offset: int) -> None:
        """Emit 0xE0 seek to a specific PCM data bank offset."""
        self._buf.append(struct.pack("<BI", _CMD_DAC_SEEK, offset))

    def write_dac_stream(self, pcm_bytes: bytes) -> None:
        """Stream PCM bytes through the DAC using 0x80-0x8F commands.

        Each byte is emitted as a 0x80 command (write PCM byte + wait 0 extra
        samples).  This replicates how real Genesis VGMs play DAC samples.
        """
        for b in pcm_bytes:
            # 0x80 command writes next byte from PCM bank to DAC register 0x2A
            # and waits (cmd & 0x0F) = 0 additional samples.
            self._buf.append(bytes([0x80]))

    def write_pcm_data_block(self, pcm_data: bytes) -> None:
        """Emit a 0x67 PCM data block header for the full drum kit bank.

        Must be emitted before any 0xE0 seek or 0x80 DAC write commands.
        Format: 0x67 0x66 0x00 <size u32-LE> <data>
        """
        size = len(pcm_data)
        block = bytearray(7 + size)
        block[0] = 0x67
        block[1] = 0x66
        block[2] = 0x00          # type: YM2612 PCM data
        struct.pack_into("<I", block, 3, size)
        block[7:] = pcm_data
        self._buf.append(bytes(block))

    def end(self) -> None:
        self._buf.append(bytes([_CMD_END]))

    def getvalue(self) -> bytes:
        return b"".join(self._buf)

    @property
    def sample_pos(self) -> int:
        return self._pos


# ---------------------------------------------------------------------------
# FM register helpers
# ---------------------------------------------------------------------------

def _port_and_reg_offset(ch: int) -> tuple[int, int]:
    """Return (port, ch_offset) for channel index 0-5."""
    if ch < 3:
        return 0, ch
    return 1, ch - 3


def _write_patch(stream: _VgmStream, ch: int, patch: Ym2612Patch) -> None:
    """Emit all YM2612 register writes to program an FM patch on *ch*."""
    port, offset = _port_and_reg_offset(ch)

    # Algorithm and feedback
    stream.write_ym2612(port, 0xB0 + offset,
                        ((patch.feedback & 0x07) << 3) | (patch.algorithm & 0x07))

    # Stereo (LR) / AMS / FMS — preserve original panning, default to both if absent
    pan = getattr(patch, 'pan', 3)
    stream.write_ym2612(port, 0xB4 + offset,
                        ((pan & 0x03) << 6) | ((patch.ams & 0x03) << 4) | (patch.fms & 0x07))

    # Per-operator registers
    for op_idx in range(4):
        slot = _OP_TO_SLOT[op_idx]          # physical slot
        slot_offset = slot * 4              # distance between slots in reg space

        stream.write_ym2612(port, 0x30 + offset + slot_offset,
                            ((patch.dt[op_idx] & 0x07) << 4) | (patch.mul[op_idx] & 0x0F))
        stream.write_ym2612(port, 0x40 + offset + slot_offset,
                            patch.tl[op_idx] & 0x7F)
        stream.write_ym2612(port, 0x50 + offset + slot_offset,
                            patch.ar[op_idx] & 0x1F)
        stream.write_ym2612(port, 0x60 + offset + slot_offset,
                            (int(patch.am_en[op_idx]) << 7) | (patch.dr[op_idx] & 0x1F))
        stream.write_ym2612(port, 0x70 + offset + slot_offset,
                            patch.sr[op_idx] & 0x1F)
        stream.write_ym2612(port, 0x80 + offset + slot_offset,
                            ((patch.sl[op_idx] & 0x0F) << 4) | (patch.rr[op_idx] & 0x0F))
        stream.write_ym2612(port, 0x90 + offset + slot_offset,
                            patch.ssg_eg[op_idx] & 0x0F)


def _write_fnumber(stream: _VgmStream, ch: int, midi: int) -> None:
    """Emit F-Number + Block register writes for a given MIDI pitch.

    YM2612 shadow-register rule: writing 0xA4 stores hi+block in a shadow
    register; writing 0xA0 IMMEDIATELY commits (new_lo, shadow_hi, shadow_block)
    to the live frequency register.  So 0xA4 MUST be written first.
    """
    port, offset = _port_and_reg_offset(ch)
    block = optimal_block(midi)
    fnum  = midi_to_fnumber(midi, block)
    stream.write_ym2612(port, 0xA4 + offset, ((block & 0x07) << 3) | ((fnum >> 8) & 0x07))
    stream.write_ym2612(port, 0xA0 + offset, fnum & 0xFF)


def _write_key_on(stream: _VgmStream, ch: int) -> None:
    key_bits = _CH_TO_KEY_BITS[ch]
    # All four operators on (bits 7:4 = 0xF)
    stream.write_ym2612(0, 0x28, 0xF0 | key_bits)


def _write_key_off(stream: _VgmStream, ch: int) -> None:
    key_bits = _CH_TO_KEY_BITS[ch]
    stream.write_ym2612(0, 0x28, 0x00 | key_bits)


# ---------------------------------------------------------------------------
# PSG helpers
# ---------------------------------------------------------------------------

def _psg_tone_period(midi: int, sn_clock: int = SN76489_CLOCK) -> int:
    """Convert MIDI note to SN76489 tone period register value."""
    if midi < 0:
        return 0
    freq = 440.0 * (2.0 ** ((midi - 69) / 12.0))
    if freq <= 0:
        return 0
    period = round(sn_clock / (32.0 * freq))
    return max(1, min(0x3FF, period))


def _write_psg_tone(stream: _VgmStream, ch_psg: int, period: int, volume: int) -> None:
    """Emit SN76489 writes to set tone period and volume for PSG channel 0-2."""
    ch = ch_psg  # 0, 1, or 2

    # Latch + low data: 0b1_cc_0_dddd
    low4 = period & 0x0F
    stream.write_psg(0x80 | (ch << 5) | low4)

    # High data: 0b00_dddddd  (upper 6 bits of period)
    high6 = (period >> 4) & 0x3F
    stream.write_psg(high6)

    # Volume (attenuation): 0b1_cc_1_vvvv  (0=loudest, 15=silent)
    # Inverse of _db_to_vel(atten*2): vel = 15*(1 - atten*2/60)  →  atten = (15-vel)*2
    attenuation = max(0, min(15, (15 - volume) * 2))
    stream.write_psg(0x80 | (ch << 5) | 0x10 | attenuation)


def _write_psg_volume(stream: _VgmStream, ch_psg: int, volume: int) -> None:
    """Emit only the SN76489 volume byte for PSG channel 0-2."""
    attenuation = max(0, min(15, (15 - volume) * 2))
    stream.write_psg(0x80 | (ch_psg << 5) | 0x10 | attenuation)


def _write_psg_noise_hit(stream: _VgmStream) -> None:
    """Emit a short noise burst on PSG noise channel (channel 3)."""
    # Noise control: 0b1_11_0_ffbb  ff=white/periodic, bb=rate
    # 0b1_11_0_0111 = white noise, max rate
    stream.write_psg(0xE7)                      # noise on, white, high rate
    stream.write_psg(0x80 | (3 << 5) | 0x10 | 0)   # ch3 volume = 0 (loud)


def _write_psg_off(stream: _VgmStream, ch_psg: int) -> None:
    """Silence a PSG tone channel."""
    stream.write_psg(0x80 | (ch_psg << 5) | 0x10 | 0x0F)  # attenuation = 15


def _write_psg_noise_off(stream: _VgmStream) -> None:
    """Silence PSG noise channel."""
    stream.write_psg(0x80 | (3 << 5) | 0x10 | 0x0F)


# ---------------------------------------------------------------------------
# Main synthesis function
# ---------------------------------------------------------------------------

def synthesise_vgm(
    notes: list[NoteEvent],
    total_samples: int,
    patch_map: dict[int, Ym2612Patch] | None = None,
    ym2612_clock: int = YM2612_CLOCK,
    sn76489_clock: int = SN76489_CLOCK,
    drum_kit: dict[int, bytes] | None = None,
    pcm_data: bytes | None = None,
    dac_stream: list | None = None,
) -> bytes:
    """Synthesise a VGM binary from decoded NoteEvents.

    Args:
        notes:         List of NoteEvent objects (from ym2612.decode_vgm or
                       tokenizer_v4.TokenizerV4.decode).
        total_samples: Total playback length in samples at 44100 Hz.
        patch_map:     Optional dict of ch_index → Ym2612Patch to use as
                       fallback defaults per channel.  NoteEvent.patch takes
                       priority when present.
        ym2612_clock:  YM2612 clock frequency (default: 7,670,454 Hz).
        sn76489_clock: SN76489 clock frequency (default: 3,579,545 Hz).
        drum_kit:      Optional dict of slot_index (0-7) → raw PCM bytes.
                       Used when dac_sample_id is a slot index (tokenizer path).
        pcm_data:      Optional raw PCM bank bytes from the original VGM.
                       Used when dac_sample_id is a raw bank offset (direct path).
        sn76489_clock: SN76489 clock frequency (default: 3,579,545 Hz).
        drum_kit:      Optional dict of slot_index (0-7) → raw PCM bytes.
                       When provided, DAC hits seek to and stream the correct
                       sample.  When None, a single 0x80 (silence) byte is
                       emitted per hit (legacy behaviour).

    Returns:
        Raw VGM 1.61 file contents as bytes.
    """
    patch_map = patch_map or {}
    stream = _VgmStream()

    # ---- Build concatenated PCM bank from drum kit or raw pcm_data ----------
    pcm_bank = bytearray()
    slot_to_bank_offset: dict[int, int] = {}
    if drum_kit:
        # Tokenizer path: slot indices 0-7 mapped to contiguous bank
        for slot in sorted(drum_kit.keys()):
            slot_to_bank_offset[slot] = len(pcm_bank)
            pcm_bank.extend(drum_kit[slot])
    elif pcm_data:
        # Direct decode path: use original VGM's PCM bank verbatim
        pcm_bank.extend(pcm_data)

    # ---- Initial silence on all PSG channels ----
    for psg_ch in range(3):
        _write_psg_off(stream, psg_ch)
    _write_psg_noise_off(stream)

    # ---- Emit PCM data block before any DAC commands -------------------
    if pcm_bank:
        stream.write_pcm_data_block(bytes(pcm_bank))

    # ---- Build sorted action list ----
    actions: list[_Action] = []

    for note in notes:
        sample_on  = max(0, note.sample_on)
        sample_off = max(0, note.sample_off) if note.sample_off >= 0 else total_samples

        ch = note.channel

        if 0 <= ch <= 5:
            # FM channel
            patch = note.patch or patch_map.get(ch)
            if patch is not None:
                actions.append(_Action(sample_on,  0, "fm_patch", (ch, patch)))
            actions.append(_Action(sample_on,   1, "fm_on",    (ch, note.pitch)))
            actions.append(_Action(sample_off, -1, "fm_off",   (ch,)))

        elif ch == CH_DAC:
            if dac_stream is None:  # only use NoteEvent DAC if no verbatim stream provided
                actions.append(_Action(sample_on, 1, "dac_hit", (note.dac_sample_id,)))

        elif ch in (CH_PSG_0, CH_PSG_1, CH_PSG_2):
            psg_ch = ch - CH_PSG_0
            period = _psg_tone_period(note.pitch, sn76489_clock)
            actions.append(_Action(sample_on,   1, "psg_on",  (psg_ch, period, note.velocity)))
            actions.append(_Action(sample_off, -1, "psg_off", (psg_ch,)))
            # Emit mid-note pitch bends from the pitch envelope
            for wp_sample, wp_period in note.pitch_envelope:
                # wp_period is the raw SN76489 period register value stored by
                # the decoder — use it directly (no MIDI round-trip conversion).
                if wp_sample > sample_on and wp_sample < sample_off:
                    actions.append(_Action(wp_sample, 1, "psg_bend", (psg_ch, wp_period)))

        elif ch == CH_PSG_NOISE:
            actions.append(_Action(sample_on,   1, "noise_on",  ()))
            actions.append(_Action(sample_off, -1, "noise_off", ()))

    # ---- Build verbatim DAC stream actions (if provided) ----
    if dac_stream:
        for (s, cmd, arg) in dac_stream:
            if cmd == 'seek':
                actions.append(_Action(s, 0, 'dac_seek', arg))
            elif cmd == 'write':
                actions.append(_Action(s, 1, 'dac_write', arg))

    # Sort by (sample, priority)
    actions.sort()

    # ---- Track FM DAC mode ----
    dac_enabled = any(a.kind in ("dac_hit", "dac_seek", "dac_write") for a in actions)

    # ---- LFO: enable if any patch uses it, using that patch's rate ----
    lfo_patch = next(
        (a.data[1] for a in actions if a.kind == "fm_patch" and a.data[1].lfo_en),
        None,
    )
    if lfo_patch is not None:
        stream.write_ym2612(0, 0x22, 0x08 | (lfo_patch.lfo_rate & 0x07))
    else:
        stream.write_ym2612(0, 0x22, 0x00)  # LFO off

    # ---- Enable DAC if needed ----
    if dac_enabled:
        stream.write_ym2612(0, 0x2B, 0x80)  # DAC enable

    # ---- Track which patches have been written per channel ----
    last_patch_fp: dict[int, tuple] = {}

    # ---- Emit actions ----
    for action in actions:
        stream.advance_to(action.sample)

        if action.kind == "fm_patch":
            ch, patch = action.data
            fp = patch.to_fingerprint()
            if last_patch_fp.get(ch) != fp:
                _write_patch(stream, ch, patch)
                last_patch_fp[ch] = fp

        elif action.kind == "fm_on":
            ch, midi = action.data
            if midi >= 0:
                _write_fnumber(stream, ch, midi)
                _write_key_on(stream, ch)

        elif action.kind == "fm_off":
            (ch,) = action.data
            _write_key_off(stream, ch)

        elif action.kind == "dac_hit":
            (slot,) = action.data
            if slot_to_bank_offset and slot in slot_to_bank_offset:
                # Tokenizer path: slot index → offset in our reconstructed bank
                bank_offset = slot_to_bank_offset[slot]
                sample_bytes = drum_kit[slot]
                stream.write_dac_seek(bank_offset)
                stream.write_dac_stream(sample_bytes)
            elif pcm_data and 0 <= slot < len(pcm_data):
                # Direct decode path: dac_sample_id IS the raw bank offset
                stream.write_dac_seek(slot)
                # Determine length by finding the next hit offset, capped at 4096 bytes
                stream.write_dac_stream(pcm_data[slot:slot + 4096])
            else:
                # No PCM data available — emit single silence byte (legacy)
                stream.write_dac()

        elif action.kind == "psg_on":
            psg_ch, period, velocity = action.data
            _write_psg_tone(stream, psg_ch, period, velocity)

        elif action.kind == "psg_off":
            (psg_ch,) = action.data
            _write_psg_off(stream, psg_ch)

        elif action.kind == "psg_bend":
            # Mid-note period change — write period bytes only, no volume re-set
            psg_ch, period = action.data
            ch = psg_ch
            low4  = period & 0x0F
            high6 = (period >> 4) & 0x3F
            stream.write_psg(0x80 | (ch << 5) | low4)
            stream.write_psg(high6)

        elif action.kind == "dac_seek":
            stream.write_dac_seek(action.data)

        elif action.kind == "dac_write":
            stream.write_dac_byte(action.data)  # 0x8n write + advance n extra samples

        elif action.kind == "noise_on":
            _write_psg_noise_hit(stream)

        elif action.kind == "noise_off":
            _write_psg_noise_off(stream)

    # ---- Advance to full length and end ----
    stream.advance_to(total_samples)
    stream.end()

    data_bytes = stream.getvalue()
    return _build_vgm_header(data_bytes, total_samples, ym2612_clock, sn76489_clock)


# ---------------------------------------------------------------------------
# VGM header builder
# ---------------------------------------------------------------------------

def _build_vgm_header(
    data: bytes,
    total_samples: int,
    ym2612_clock: int,
    sn76489_clock: int,
) -> bytes:
    """Prepend a valid VGM 1.61 header to *data* and return the full file."""

    # VGM 1.61 header is 0x40 bytes; data starts at 0x40.
    # data_offset field (at 0x34) is relative to 0x34; value = 0x40 - 0x34 = 0x0C.
    HEADER_SIZE  = 0x40
    DATA_OFFSET_FIELD_POS = 0x34
    data_offset_value = HEADER_SIZE - DATA_OFFSET_FIELD_POS   # = 0x0C

    total_size = HEADER_SIZE + len(data)
    eof_offset = total_size - 4   # relative to offset 0x04

    header = bytearray(HEADER_SIZE)

    def u32(offset: int, value: int) -> None:
        struct.pack_into("<I", header, offset, value & 0xFFFFFFFF)

    def u16(offset: int, value: int) -> None:
        struct.pack_into("<H", header, offset, value & 0xFFFF)

    # Magic
    header[0:4] = b"Vgm "

    u32(0x04, eof_offset)
    u32(0x08, 0x00000161)           # VGM version 1.61
    u32(0x0C, sn76489_clock)        # SN76489 clock
    u32(0x10, 0)                    # YM2413 not present
    u32(0x14, 0)                    # GD3 tag offset = 0 (none)
    u32(0x18, total_samples)
    u32(0x1C, 0)                    # no loop
    u32(0x20, 0)                    # no loop
    u32(0x24, 60)                   # 60 Hz rate

    # SN76489 feedback / shift-register width (Genesis values)
    u16(0x28, 0x0009)               # feedback mask
    header[0x2A] = 16               # shift register width
    header[0x2B] = 0x00             # flags

    u32(0x2C, ym2612_clock)         # YM2612 clock
    u32(0x30, 0)                    # YM2151 not present
    u32(DATA_OFFSET_FIELD_POS, data_offset_value)

    # Remaining bytes (0x38–0x3F) are zero (no extra chips)

    return bytes(header) + data
