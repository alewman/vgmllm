"""VGM binary format parser for Sega Genesis (YM2612 + SN76489).

Parses .vgm and .vgz files into structured event sequences suitable for
tokenization and ML training.

Reference: https://vgmrips.net/wiki/VGM_Specification
"""

from __future__ import annotations

import gzip
import struct
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import BinaryIO


# ---------------------------------------------------------------------------
# Data classes for parsed VGM content
# ---------------------------------------------------------------------------

class EventType(IntEnum):
    """Types of events we extract from VGM data streams."""
    YM2612_PORT0 = 0  # YM2612 port 0 register write
    YM2612_PORT1 = 1  # YM2612 port 1 register write
    SN76489 = 2       # SN76489 PSG write
    WAIT = 3          # Wait N samples
    END = 4           # End of data
    DAC_WRITE = 5     # YM2612 DAC sample write (from data bank)


@dataclass(slots=True)
class VgmEvent:
    """A single event from the VGM data stream."""
    type: EventType
    # For register writes: register address (YM2612) or data byte (SN76489)
    register: int = 0
    # For register writes: data value; for WAIT: number of samples
    value: int = 0
    # Absolute sample position in the stream (accumulated from waits)
    sample_pos: int = 0
    # For DAC_WRITE: byte offset in the PCM data bank where this sample started
    # (set by the most recent 0xE0 seek command). Used for drum identity detection.
    pcm_offset: int = 0


@dataclass
class VgmHeader:
    """Parsed VGM file header."""
    version: int = 0          # BCD version (e.g. 0x171 = v1.71)
    eof_offset: int = 0
    sn76489_clock: int = 0    # 0 if chip not used
    ym2413_clock: int = 0     # 0 if chip not used (offset 0x10)
    ym2612_clock: int = 0     # 0 if chip not used (offset 0x2C)
    gd3_offset: int = 0
    total_samples: int = 0
    loop_offset: int = 0      # 0 if no loop
    loop_samples: int = 0
    data_offset: int = 0      # Absolute offset where VGM data begins

    @property
    def version_string(self) -> str:
        major = (self.version >> 8) & 0xFF
        minor = self.version & 0xFF
        return f"{major}.{minor:02x}"

    @property
    def duration_seconds(self) -> float:
        return self.total_samples / 44100.0

    @property
    def has_ym2612(self) -> bool:
        return self.ym2612_clock != 0

    @property
    def has_sn76489(self) -> bool:
        return self.sn76489_clock != 0


@dataclass
class VgmFile:
    """A fully parsed VGM file."""
    header: VgmHeader
    events: list[VgmEvent] = field(default_factory=list)
    source_path: str | None = None

    # PCM data bank for DAC streaming (may be empty)
    pcm_data: bytes = b""

    @property
    def ym2612_events(self) -> list[VgmEvent]:
        """Return only YM2612 register write events."""
        return [
            e for e in self.events
            if e.type in (EventType.YM2612_PORT0, EventType.YM2612_PORT1)
        ]

    @property
    def wait_events(self) -> list[VgmEvent]:
        return [e for e in self.events if e.type == EventType.WAIT]

    @property
    def sn76489_events(self) -> list[VgmEvent]:
        return [e for e in self.events if e.type == EventType.SN76489]


# ---------------------------------------------------------------------------
# GD3 tag parser (metadata: game name, track name, etc.)
# ---------------------------------------------------------------------------

@dataclass
class Gd3Tag:
    """GD3 metadata tag."""
    track_name_en: str = ""
    track_name_jp: str = ""
    game_name_en: str = ""
    game_name_jp: str = ""
    system_name_en: str = ""
    system_name_jp: str = ""
    author_en: str = ""
    author_jp: str = ""
    date: str = ""
    ripper: str = ""
    notes: str = ""


def _read_gd3_string(data: bytes, offset: int) -> tuple[str, int]:
    """Read a null-terminated UTF-16LE string from GD3 data."""
    end = offset
    while end + 1 < len(data):
        char = int.from_bytes(data[end:end + 2], "little")
        if char == 0:
            break
        end += 2
    text = data[offset:end].decode("utf-16-le", errors="replace")
    return text, end + 2  # skip past null terminator


def parse_gd3(data: bytes, offset: int) -> Gd3Tag | None:
    """Parse a GD3 tag at the given offset in raw file data."""
    if offset <= 0 or offset + 12 > len(data):
        return None

    magic = data[offset:offset + 4]
    if magic != b"Gd3 ":
        return None

    tag = Gd3Tag()
    # Skip magic (4) + version (4) + data size (4) = 12 bytes
    pos = offset + 12
    fields = [
        "track_name_en", "track_name_jp",
        "game_name_en", "game_name_jp",
        "system_name_en", "system_name_jp",
        "author_en", "author_jp",
        "date", "ripper", "notes",
    ]
    for field_name in fields:
        if pos >= len(data):
            break
        text, pos = _read_gd3_string(data, pos)
        setattr(tag, field_name, text)
    return tag


# ---------------------------------------------------------------------------
# VGM Parser
# ---------------------------------------------------------------------------

def _read_u32(data: bytes, offset: int) -> int:
    """Read unsigned 32-bit little-endian value."""
    return struct.unpack_from("<I", data, offset)[0]


def _read_u16(data: bytes, offset: int) -> int:
    """Read unsigned 16-bit little-endian value."""
    return struct.unpack_from("<H", data, offset)[0]


def parse_header(data: bytes) -> VgmHeader:
    """Parse a VGM header from raw file data."""
    if len(data) < 64:
        raise ValueError(f"File too small for VGM header: {len(data)} bytes")

    magic = data[0:4]
    if magic != b"Vgm ":
        raise ValueError(f"Invalid VGM magic: {magic!r} (expected b'Vgm ')")

    h = VgmHeader()
    h.eof_offset = _read_u32(data, 0x04) + 0x04
    h.version = _read_u32(data, 0x08)
    h.sn76489_clock = _read_u32(data, 0x0C)
    h.ym2413_clock = _read_u32(data, 0x10)
    h.gd3_offset = _read_u32(data, 0x14)
    if h.gd3_offset != 0:
        h.gd3_offset += 0x14  # make absolute
    h.total_samples = _read_u32(data, 0x18)
    h.loop_offset = _read_u32(data, 0x1C)
    if h.loop_offset != 0:
        h.loop_offset += 0x1C  # make absolute
    h.loop_samples = _read_u32(data, 0x20)

    # YM2612 clock is at offset 0x2C (version >= 1.10)
    if len(data) > 0x30:
        h.ym2612_clock = _read_u32(data, 0x2C)

    # VGM data offset: at 0x34 in version >= 1.50, otherwise data starts at 0x40
    if h.version >= 0x150 and len(data) > 0x38:
        rel = _read_u32(data, 0x34)
        h.data_offset = (0x34 + rel) if rel != 0 else 0x40
    else:
        h.data_offset = 0x40

    return h


def _extract_pcm_data_block(data: bytes, offset: int) -> tuple[bytes, int]:
    """Extract a PCM data block (command 0x67) from the stream.

    Format: 0x67 0x66 tt ss ss ss ss [data...]
    tt = data type, ss = size (32-bit LE)
    Returns (pcm_bytes, new_offset_after_block).
    """
    if offset + 6 > len(data):
        return b"", offset + 1
    # data[offset] == 0x67, data[offset+1] should be 0x66
    data_type = data[offset + 2]
    size = _read_u32(data, offset + 3)
    pcm_start = offset + 7
    pcm_end = pcm_start + size
    if pcm_end > len(data):
        pcm_end = len(data)
    return data[pcm_start:pcm_end], pcm_end


def parse_events(data: bytes, header: VgmHeader) -> tuple[list[VgmEvent], bytes]:
    """Parse VGM data stream into a list of events.

    Returns (events, pcm_data).
    """
    events: list[VgmEvent] = []
    pcm_data = b""
    pcm_offset = 0  # current read position in PCM data bank

    pos = header.data_offset
    end = min(header.eof_offset, len(data))
    sample_pos = 0

    def _append_wait(samples: int):
        nonlocal sample_pos
        if samples <= 0:
            return
        # Merge consecutive waits
        if events and events[-1].type == EventType.WAIT:
            events[-1].value += samples
            sample_pos += samples
        else:
            events.append(VgmEvent(
                type=EventType.WAIT,
                value=samples,
                sample_pos=sample_pos,
            ))
            sample_pos += samples

    while pos < end:
        cmd = data[pos]

        # YM2612 port 0 write
        if cmd == 0x52:
            if pos + 2 >= end:
                break
            reg = data[pos + 1]
            val = data[pos + 2]
            events.append(VgmEvent(
                type=EventType.YM2612_PORT0,
                register=reg,
                value=val,
                sample_pos=sample_pos,
            ))
            pos += 3

        # YM2612 port 1 write
        elif cmd == 0x53:
            if pos + 2 >= end:
                break
            reg = data[pos + 1]
            val = data[pos + 2]
            events.append(VgmEvent(
                type=EventType.YM2612_PORT1,
                register=reg,
                value=val,
                sample_pos=sample_pos,
            ))
            pos += 3

        # SN76489 write
        elif cmd == 0x50:
            if pos + 1 >= end:
                break
            val = data[pos + 1]
            events.append(VgmEvent(
                type=EventType.SN76489,
                value=val,
                sample_pos=sample_pos,
            ))
            pos += 2

        # Wait N samples
        elif cmd == 0x61:
            if pos + 2 >= end:
                break
            samples = _read_u16(data, pos + 1)
            _append_wait(samples)
            pos += 3

        # Wait 735 samples (NTSC frame)
        elif cmd == 0x62:
            _append_wait(735)
            pos += 1

        # Wait 882 samples (PAL frame)
        elif cmd == 0x63:
            _append_wait(882)
            pos += 1

        # End of sound data
        elif cmd == 0x66:
            events.append(VgmEvent(
                type=EventType.END,
                sample_pos=sample_pos,
            ))
            break

        # Short waits: 0x70-0x7F = wait (n+1) samples where n = cmd & 0x0F
        elif 0x70 <= cmd <= 0x7F:
            samples = (cmd & 0x0F) + 1
            _append_wait(samples)
            pos += 1

        # YM2612 DAC write + wait: 0x80-0x8F
        # Write byte from PCM data bank to YM2612 port 0 register 0x2A,
        # then wait (cmd & 0x0F) samples
        elif 0x80 <= cmd <= 0x8F:
            wait_n = cmd & 0x0F
            if pcm_data and pcm_offset < len(pcm_data):
                events.append(VgmEvent(
                    type=EventType.DAC_WRITE,
                    register=0x2A,
                    value=pcm_data[pcm_offset],
                    sample_pos=sample_pos,
                    pcm_offset=pcm_offset,
                ))
                pcm_offset += 1
            if wait_n > 0:
                _append_wait(wait_n)
            pos += 1

        # PCM data block
        elif cmd == 0x67:
            pcm_data, pos = _extract_pcm_data_block(data, pos)

        # PCM data bank seek — tracks which sample is being played
        elif cmd == 0xE0:
            if pos + 4 >= end:
                break
            pcm_offset = _read_u32(data, pos + 1)
            pos += 5

        # --- Commands we skip but need to advance past ---
        # Two-byte commands (0x30-0x3F range, other chip writes)
        elif 0x30 <= cmd <= 0x3F:
            pos += 2  # 1 byte cmd + 1 byte data

        # Three-byte commands (0x40-0x4E range, other chip writes)
        elif 0x40 <= cmd <= 0x4E:
            pos += 3

        # 0x4F: Game Gear stereo
        elif cmd == 0x4F:
            pos += 2

        # 0x51: YM2413
        elif cmd == 0x51:
            pos += 3

        # 0x54-0x5F: Various other chip writes (3 bytes each)
        elif 0x54 <= cmd <= 0x5F:
            pos += 3

        # 0xA0-0xBF: Various chip writes (3 bytes each)
        elif 0xA0 <= cmd <= 0xBF:
            pos += 3

        # 0xC0-0xDF: Various chip writes (4 bytes each)
        elif 0xC0 <= cmd <= 0xDF:
            pos += 4

        # 0xE1-0xFF: Various chip writes (5 bytes each)
        elif 0xE1 <= cmd <= 0xFF:
            pos += 5

        # Unknown command — skip one byte and hope for the best
        else:
            pos += 1

    return events, pcm_data


def parse_vgm(data: bytes, source_path: str | None = None) -> VgmFile:
    """Parse raw VGM data (already decompressed) into a VgmFile."""
    header = parse_header(data)
    events, pcm_data = parse_events(data, header)
    gd3 = parse_gd3(data, header.gd3_offset) if header.gd3_offset else None

    vgm = VgmFile(
        header=header,
        events=events,
        source_path=source_path,
        pcm_data=pcm_data,
    )
    # Attach GD3 metadata as an attribute (optional, not in __init__)
    vgm.gd3 = gd3  # type: ignore[attr-defined]
    return vgm


def load_vgm(path: str | Path) -> VgmFile:
    """Load a VGM or VGZ file from disk."""
    path = Path(path)
    raw = path.read_bytes()

    # Try gzip decompression (VGZ files, or .vgm that are actually gzipped)
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)

    return parse_vgm(raw, source_path=str(path))


# ---------------------------------------------------------------------------
# YM2612 register analysis helpers
# ---------------------------------------------------------------------------

# Register ranges on each port (port 0 = ch1-3, port 1 = ch4-6)
YM2612_REGISTERS = {
    # Global registers (port 0 only)
    0x22: "LFO",
    0x24: "Timer A MSB",
    0x25: "Timer A LSB",
    0x26: "Timer B",
    0x27: "Ch3 Mode / Timers",
    0x28: "Key On/Off",
    0x2A: "DAC Data",
    0x2B: "DAC Enable",
}


def ym2612_register_name(register: int, port: int = 0) -> str:
    """Get a human-readable name for a YM2612 register."""
    # Global registers
    if register in YM2612_REGISTERS and port == 0:
        return YM2612_REGISTERS[register]

    # Per-channel / per-operator registers
    if 0x30 <= register <= 0x9E:
        op = (register - 0x30) // 0x10  # operator 0-3 (but interleaved)
        # Map to actual operator: register layout is op1, op3, op2, op4
        op_map = {0: 1, 1: 3, 2: 2, 3: 4}
        op_num = op_map.get(op & 0x03, op & 0x03)
        ch = (register & 0x03)  # channel within port group (0-2)
        if port == 1:
            ch += 3

        reg_type = (register - 0x30) & 0xF0
        type_names = {
            0x00: "DT/MUL", 0x10: "TL", 0x20: "RS/AR",
            0x30: "AM/DR", 0x40: "SR", 0x50: "SL/RR",
            0x60: "SSG-EG",
        }
        name = type_names.get(reg_type, f"0x{register:02X}")
        return f"Ch{ch + 1} Op{op_num} {name}"

    if 0xA0 <= register <= 0xA2:
        ch = register - 0xA0 + (3 if port == 1 else 0)
        return f"Ch{ch + 1} Freq LSB"

    if 0xA4 <= register <= 0xA6:
        ch = register - 0xA4 + (3 if port == 1 else 0)
        return f"Ch{ch + 1} Freq MSB/Block"

    if 0xA8 <= register <= 0xAE:
        return f"Ch3 Special Freq"

    if 0xB0 <= register <= 0xB2:
        ch = register - 0xB0 + (3 if port == 1 else 0)
        return f"Ch{ch + 1} Algorithm/Feedback"

    if 0xB4 <= register <= 0xB6:
        ch = register - 0xB4 + (3 if port == 1 else 0)
        return f"Ch{ch + 1} Stereo/LFO"

    return f"0x{register:02X}"


def summarize_vgm(vgm: VgmFile) -> dict:
    """Generate a summary of a parsed VGM file for debugging / analysis."""
    ym_p0 = sum(1 for e in vgm.events if e.type == EventType.YM2612_PORT0)
    ym_p1 = sum(1 for e in vgm.events if e.type == EventType.YM2612_PORT1)
    psg = sum(1 for e in vgm.events if e.type == EventType.SN76489)
    waits = sum(1 for e in vgm.events if e.type == EventType.WAIT)
    dac = sum(1 for e in vgm.events if e.type == EventType.DAC_WRITE)

    return {
        "version": vgm.header.version_string,
        "duration_seconds": round(vgm.header.duration_seconds, 2),
        "total_samples": vgm.header.total_samples,
        "has_ym2612": vgm.header.has_ym2612,
        "has_sn76489": vgm.header.has_sn76489,
        "ym2612_clock": vgm.header.ym2612_clock,
        "total_events": len(vgm.events),
        "ym2612_port0_writes": ym_p0,
        "ym2612_port1_writes": ym_p1,
        "sn76489_writes": psg,
        "wait_events": waits,
        "dac_writes": dac,
        "pcm_data_size": len(vgm.pcm_data),
        "has_loop": vgm.header.loop_offset != 0,
        "source": vgm.source_path,
    }
