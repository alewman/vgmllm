"""VGM file writer — converts event sequences back to valid VGM binary files.

This is the inverse of vgm_parser: takes a list of VgmEvents and writes a
playable .vgm file that any VGM player can render through a YM2612 emulator.
"""

from __future__ import annotations

import struct
from pathlib import Path

from .vgm_parser import EventType, VgmEvent


def _write_u32(value: int) -> bytes:
    return struct.pack("<I", value & 0xFFFFFFFF)


def _write_u16(value: int) -> bytes:
    return struct.pack("<H", value & 0xFFFF)


def _encode_wait(samples: int) -> bytearray:
    """Encode a wait of N samples using the most efficient VGM commands."""
    out = bytearray()
    remaining = samples

    while remaining > 0:
        if remaining == 735:
            out.append(0x62)  # Wait 1 NTSC frame
            remaining = 0
        elif remaining == 882:
            out.append(0x63)  # Wait 1 PAL frame
            remaining = 0
        elif remaining <= 16:
            # Short wait: 0x7n = wait n+1 samples
            out.append(0x70 | (remaining - 1))
            remaining = 0
        elif remaining >= 735 * 2 and remaining % 735 == 0:
            # Multiple NTSC frames
            frames = remaining // 735
            for _ in range(frames):
                out.append(0x62)
            remaining = 0
        elif remaining > 65535:
            # Max 16-bit wait
            out.append(0x61)
            out.extend(_write_u16(65535))
            remaining -= 65535
        else:
            # Generic wait
            out.append(0x61)
            out.extend(_write_u16(remaining))
            remaining = 0

    return out


def events_to_vgm(
    events: list[VgmEvent],
    ym2612_clock: int = 7670453,  # NTSC Genesis
    sn76489_clock: int = 3579545,
    loop_offset: int | None = None,
) -> bytes:
    """Convert a list of VgmEvents into a complete VGM file.

    Args:
        events: Sequence of VGM events to encode.
        ym2612_clock: YM2612 clock rate. 0 to disable.
        sn76489_clock: SN76489 clock rate. 0 to disable.
        loop_offset: If not None, byte offset into the data section where
            the loop point is. Set to 0 to loop from the beginning.

    Returns:
        Complete VGM file as bytes.
    """
    # Build the data section
    data = bytearray()
    total_samples = 0

    loop_data_offset = None
    if loop_offset == 0:
        loop_data_offset = 0

    for event in events:
        if loop_offset is not None and event.sample_pos == loop_offset and loop_data_offset is None:
            loop_data_offset = len(data)

        if event.type == EventType.YM2612_PORT0:
            data.append(0x52)
            data.append(event.register & 0xFF)
            data.append(event.value & 0xFF)

        elif event.type == EventType.YM2612_PORT1:
            data.append(0x53)
            data.append(event.register & 0xFF)
            data.append(event.value & 0xFF)

        elif event.type == EventType.SN76489:
            data.append(0x50)
            data.append(event.value & 0xFF)

        elif event.type == EventType.DAC_WRITE:
            # Write DAC data as a normal port 0 register 0x2A write
            data.append(0x52)
            data.append(0x2A)
            data.append(event.value & 0xFF)

        elif event.type == EventType.WAIT:
            total_samples += event.value
            data.extend(_encode_wait(event.value))

        elif event.type == EventType.END:
            break

    # End of data marker
    data.append(0x66)

    # Build header (VGM v1.71 — 256 byte header)
    header_size = 0x100  # 256 bytes, standard for v1.71
    header = bytearray(header_size)

    # Magic
    header[0:4] = b"Vgm "

    # EOF offset (relative to 0x04)
    eof = header_size + len(data)
    struct.pack_into("<I", header, 0x04, eof - 0x04)

    # Version
    struct.pack_into("<I", header, 0x08, 0x00000171)

    # Chip clocks
    struct.pack_into("<I", header, 0x0C, sn76489_clock)
    # 0x10 is YM2413 clock (not used for Genesis)
    struct.pack_into("<I", header, 0x10, 0)
    # YM2612 clock at 0x2C
    struct.pack_into("<I", header, 0x2C, ym2612_clock)

    # GD3 offset (none)
    struct.pack_into("<I", header, 0x14, 0)

    # Total samples
    struct.pack_into("<I", header, 0x18, total_samples)

    # Loop offset (relative to 0x1C)
    if loop_data_offset is not None:
        abs_loop = header_size + loop_data_offset
        struct.pack_into("<I", header, 0x1C, abs_loop - 0x1C)
        # Loop samples = total_samples (simple looping for now)
        struct.pack_into("<I", header, 0x20, total_samples)
    else:
        struct.pack_into("<I", header, 0x1C, 0)
        struct.pack_into("<I", header, 0x20, 0)

    # VGM data offset (relative to 0x34)
    struct.pack_into("<I", header, 0x34, header_size - 0x34)

    # SN76489 flags (standard Genesis values)
    if sn76489_clock:
        struct.pack_into("<H", header, 0x28, 0x0009)  # feedback
        struct.pack_into("B", header, 0x2A, 16)        # shift register width
        struct.pack_into("B", header, 0x2B, 0)         # flags

    return bytes(header) + bytes(data)


def save_vgm(
    events: list[VgmEvent],
    path: str | Path,
    ym2612_clock: int = 7670453,
    sn76489_clock: int = 3579545,
    loop_offset: int | None = None,
) -> None:
    """Write events to a .vgm file on disk."""
    vgm_data = events_to_vgm(
        events,
        ym2612_clock=ym2612_clock,
        sn76489_clock=sn76489_clock,
        loop_offset=loop_offset,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(vgm_data)


# ---------------------------------------------------------------------------
# Test sound generation helpers
# ---------------------------------------------------------------------------

def make_test_note(
    channel: int = 0,
    frequency_block: int = 4,
    frequency_num: int = 0x269,  # ~440 Hz (A4)
    algorithm: int = 7,  # All operators output directly (simple)
    total_level: int = 0x20,  # Volume (lower = louder, 0x00 = max)
    duration_frames: int = 60,  # Duration in NTSC frames (60 = 1 second)
    attack_rate: int = 31,  # Fastest attack
    decay_rate: int = 0,
    sustain_rate: int = 0,
    sustain_level: int = 0,
    release_rate: int = 7,
) -> list[VgmEvent]:
    """Generate events for a single test note on a YM2612 channel.

    This is useful for validating the writer produces playable output.
    Defaults produce a simple sine-like tone at ~440 Hz for 1 second.
    """
    events: list[VgmEvent] = []
    sample_pos = 0

    # Determine port and channel offset
    if channel < 3:
        port = EventType.YM2612_PORT0
        ch_offset = channel
    else:
        port = EventType.YM2612_PORT1
        ch_offset = channel - 3

    def wr(reg: int, val: int):
        events.append(VgmEvent(type=port, register=reg, value=val, sample_pos=sample_pos))

    def wr0(reg: int, val: int):
        """Write to port 0 (global registers)."""
        events.append(VgmEvent(
            type=EventType.YM2612_PORT0, register=reg, value=val, sample_pos=sample_pos
        ))

    # --- Configure the instrument patch ---
    # Set algorithm + feedback
    wr(0xB0 + ch_offset, (algorithm & 0x07))

    # Stereo: both L+R
    wr(0xB4 + ch_offset, 0xC0)

    # Configure all 4 operators
    for op_offset in [0x00, 0x08, 0x04, 0x0C]:  # Op1, Op2, Op3, Op4
        reg_base = ch_offset + op_offset

        # DT1/MUL: detune=0, multiply=1
        wr(0x30 + reg_base, 0x01)

        # Total Level (volume)
        wr(0x40 + reg_base, total_level)

        # Rate Scaling / Attack Rate
        wr(0x50 + reg_base, attack_rate & 0x1F)

        # AM enable / Decay Rate
        wr(0x60 + reg_base, decay_rate & 0x1F)

        # Sustain Rate
        wr(0x70 + reg_base, sustain_rate & 0x1F)

        # Sustain Level / Release Rate
        wr(0x80 + reg_base, ((sustain_level & 0x0F) << 4) | (release_rate & 0x0F))

    # --- Set frequency ---
    freq_msb = ((frequency_block & 0x07) << 3) | ((frequency_num >> 8) & 0x07)
    freq_lsb = frequency_num & 0xFF

    # Write MSB first (latches), then LSB (triggers update)
    wr(0xA4 + ch_offset, freq_msb)
    wr(0xA0 + ch_offset, freq_lsb)

    # --- Key On (all 4 operators) ---
    # Register 0x28 format: [Op4][Op3][Op2][Op1] 0 [Ch2][Ch1][Ch0]
    key_ch = channel  # channel number 0-5
    wr0(0x28, 0xF0 | key_ch)

    # --- Wait for note duration ---
    wait_samples = duration_frames * 735  # NTSC frames to samples
    sample_pos += wait_samples
    events.append(VgmEvent(type=EventType.WAIT, value=wait_samples, sample_pos=sample_pos))

    # --- Key Off ---
    wr0(0x28, 0x00 | key_ch)
    sample_pos += 0

    # Small tail for release
    tail_samples = 15 * 735  # ~0.25s release tail
    sample_pos += tail_samples
    events.append(VgmEvent(type=EventType.WAIT, value=tail_samples, sample_pos=sample_pos))

    # End
    events.append(VgmEvent(type=EventType.END, sample_pos=sample_pos))

    return events
