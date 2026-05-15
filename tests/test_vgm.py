"""Tests for VGM parser and writer — including round-trip validation."""

import struct
from genesis_music.vgm_parser import (
    EventType,
    VgmEvent,
    VgmFile,
    VgmHeader,
    load_vgm,
    parse_events,
    parse_header,
    parse_vgm,
    summarize_vgm,
    ym2612_register_name,
)
from genesis_music.vgm_writer import (
    events_to_vgm,
    make_test_note,
    save_vgm,
)


# ---------------------------------------------------------------------------
# Helpers to build minimal VGM binary data for testing
# ---------------------------------------------------------------------------

def _make_minimal_vgm(
    data_commands: bytes = b"\x66",  # just end-of-data
    version: int = 0x171,
    ym2612_clock: int = 7670453,
    sn76489_clock: int = 3579545,
    total_samples: int = 0,
) -> bytes:
    """Build a minimal valid VGM binary for testing."""
    header_size = 0x100  # 256 bytes
    header = bytearray(header_size)

    header[0:4] = b"Vgm "
    eof = header_size + len(data_commands)
    struct.pack_into("<I", header, 0x04, eof - 0x04)
    struct.pack_into("<I", header, 0x08, version)
    struct.pack_into("<I", header, 0x0C, sn76489_clock)
    struct.pack_into("<I", header, 0x10, 0)  # YM2413 clock (not used)
    struct.pack_into("<I", header, 0x18, total_samples)
    struct.pack_into("<I", header, 0x2C, ym2612_clock)
    struct.pack_into("<I", header, 0x34, header_size - 0x34)

    return bytes(header) + data_commands


# ---------------------------------------------------------------------------
# Header parsing tests
# ---------------------------------------------------------------------------

class TestHeaderParsing:
    def test_parse_valid_header(self):
        data = _make_minimal_vgm()
        h = parse_header(data)
        assert h.version == 0x171
        assert h.ym2612_clock == 7670453
        assert h.sn76489_clock == 3579545
        assert h.has_ym2612 is True
        assert h.has_sn76489 is True
        assert h.data_offset == 0x100

    def test_version_string(self):
        data = _make_minimal_vgm(version=0x171)
        h = parse_header(data)
        assert h.version_string == "1.71"

    def test_duration_seconds(self):
        data = _make_minimal_vgm(total_samples=44100)
        h = parse_header(data)
        assert h.duration_seconds == 1.0

    def test_no_ym2612(self):
        data = _make_minimal_vgm(ym2612_clock=0)
        h = parse_header(data)
        assert h.has_ym2612 is False

    def test_invalid_magic_raises(self):
        data = bytearray(_make_minimal_vgm())
        data[0:4] = b"XXXX"
        try:
            parse_header(bytes(data))
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Invalid VGM magic" in str(e)

    def test_too_small_raises(self):
        try:
            parse_header(b"Vgm ")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "too small" in str(e)

    def test_old_version_data_offset(self):
        """Versions before 1.50 should default data offset to 0x40."""
        data = _make_minimal_vgm(version=0x110)
        h = parse_header(data)
        assert h.data_offset == 0x40


# ---------------------------------------------------------------------------
# Event parsing tests
# ---------------------------------------------------------------------------

class TestEventParsing:
    def test_ym2612_port0_write(self):
        cmd = bytes([0x52, 0x28, 0xF0, 0x66])  # key-on + end
        data = _make_minimal_vgm(cmd)
        vgm = parse_vgm(data)
        ym_events = [e for e in vgm.events if e.type == EventType.YM2612_PORT0]
        assert len(ym_events) == 1
        assert ym_events[0].register == 0x28
        assert ym_events[0].value == 0xF0

    def test_ym2612_port1_write(self):
        cmd = bytes([0x53, 0xB0, 0x07, 0x66])
        data = _make_minimal_vgm(cmd)
        vgm = parse_vgm(data)
        ym_events = [e for e in vgm.events if e.type == EventType.YM2612_PORT1]
        assert len(ym_events) == 1
        assert ym_events[0].register == 0xB0
        assert ym_events[0].value == 0x07

    def test_sn76489_write(self):
        cmd = bytes([0x50, 0x9F, 0x66])
        data = _make_minimal_vgm(cmd)
        vgm = parse_vgm(data)
        psg = [e for e in vgm.events if e.type == EventType.SN76489]
        assert len(psg) == 1
        assert psg[0].value == 0x9F

    def test_wait_generic(self):
        # Wait 1000 samples
        cmd = bytes([0x61, 0xE8, 0x03, 0x66])  # 0x03E8 = 1000
        data = _make_minimal_vgm(cmd)
        vgm = parse_vgm(data)
        waits = [e for e in vgm.events if e.type == EventType.WAIT]
        assert len(waits) == 1
        assert waits[0].value == 1000

    def test_wait_ntsc_frame(self):
        cmd = bytes([0x62, 0x66])
        data = _make_minimal_vgm(cmd)
        vgm = parse_vgm(data)
        waits = [e for e in vgm.events if e.type == EventType.WAIT]
        assert len(waits) == 1
        assert waits[0].value == 735

    def test_wait_pal_frame(self):
        cmd = bytes([0x63, 0x66])
        data = _make_minimal_vgm(cmd)
        vgm = parse_vgm(data)
        waits = [e for e in vgm.events if e.type == EventType.WAIT]
        assert len(waits) == 1
        assert waits[0].value == 882

    def test_short_wait(self):
        cmd = bytes([0x75, 0x66])  # wait 6 samples
        data = _make_minimal_vgm(cmd)
        vgm = parse_vgm(data)
        waits = [e for e in vgm.events if e.type == EventType.WAIT]
        assert len(waits) == 1
        assert waits[0].value == 6  # 0x75 & 0x0F = 5, + 1 = 6

    def test_consecutive_waits_merged(self):
        """Consecutive wait commands should be merged into a single event."""
        cmd = bytes([0x62, 0x62, 0x62, 0x66])  # three NTSC frames
        data = _make_minimal_vgm(cmd)
        vgm = parse_vgm(data)
        waits = [e for e in vgm.events if e.type == EventType.WAIT]
        assert len(waits) == 1
        assert waits[0].value == 735 * 3

    def test_end_event(self):
        cmd = bytes([0x66])
        data = _make_minimal_vgm(cmd)
        vgm = parse_vgm(data)
        assert any(e.type == EventType.END for e in vgm.events)

    def test_sample_positions_accumulate(self):
        """sample_pos should track cumulative time."""
        cmd = bytes([
            0x52, 0x28, 0xF0,  # key-on at t=0
            0x62,               # wait 735
            0x52, 0x28, 0x00,  # key-off at t=735
            0x66,
        ])
        data = _make_minimal_vgm(cmd)
        vgm = parse_vgm(data)

        key_on = vgm.events[0]
        assert key_on.type == EventType.YM2612_PORT0
        assert key_on.sample_pos == 0

        key_off = [e for e in vgm.events if e.type == EventType.YM2612_PORT0 and e.value == 0x00]
        assert len(key_off) == 1
        assert key_off[0].sample_pos == 735

    def test_mixed_event_sequence(self):
        """Parse a realistic sequence with multiple event types."""
        cmd = bytes([
            0x52, 0xB0, 0x04,  # algorithm
            0x52, 0x40, 0x20,  # total level
            0x52, 0xA4, 0x22,  # freq MSB
            0x52, 0xA0, 0x69,  # freq LSB
            0x52, 0x28, 0xF0,  # key-on ch0
            0x50, 0x9F,        # PSG mute
            0x62,              # wait frame
            0x52, 0x28, 0x00,  # key-off
            0x66,
        ])
        data = _make_minimal_vgm(cmd)
        vgm = parse_vgm(data)

        types = [e.type for e in vgm.events]
        assert types.count(EventType.YM2612_PORT0) == 6
        assert types.count(EventType.SN76489) == 1
        assert types.count(EventType.WAIT) == 1
        assert types.count(EventType.END) == 1


# ---------------------------------------------------------------------------
# PCM data block tests
# ---------------------------------------------------------------------------

class TestPCMDataBlock:
    def test_pcm_data_block_extraction(self):
        """Command 0x67 should extract PCM data for DAC streaming."""
        pcm_samples = bytes([0x80, 0x90, 0xA0, 0xB0])
        # 0x67 0x66 type(00) size(4 bytes LE) data...
        block = bytes([0x67, 0x66, 0x00]) + struct.pack("<I", len(pcm_samples)) + pcm_samples
        cmd = block + bytes([0x66])
        data = _make_minimal_vgm(cmd)
        vgm = parse_vgm(data)
        assert vgm.pcm_data == pcm_samples

    def test_dac_write_events(self):
        """0x80-0x8F commands should produce DAC_WRITE events."""
        pcm_samples = bytes([0x80, 0x90])
        block = bytes([0x67, 0x66, 0x00]) + struct.pack("<I", len(pcm_samples)) + pcm_samples
        cmd = block + bytes([
            0xE0, 0x00, 0x00, 0x00, 0x00,  # seek to 0
            0x80,  # DAC write + wait 0
            0x81,  # DAC write + wait 1
            0x66,
        ])
        data = _make_minimal_vgm(cmd)
        vgm = parse_vgm(data)

        dac_events = [e for e in vgm.events if e.type == EventType.DAC_WRITE]
        assert len(dac_events) == 2
        assert dac_events[0].value == 0x80
        assert dac_events[1].value == 0x90


# ---------------------------------------------------------------------------
# Writer tests
# ---------------------------------------------------------------------------

class TestVgmWriter:
    def test_make_test_note_produces_events(self):
        events = make_test_note(channel=0, duration_frames=60)
        assert len(events) > 0
        types = {e.type for e in events}
        assert EventType.YM2612_PORT0 in types
        assert EventType.WAIT in types
        assert EventType.END in types

    def test_events_to_vgm_produces_valid_header(self):
        events = make_test_note()
        vgm_bytes = events_to_vgm(events)
        assert vgm_bytes[:4] == b"Vgm "
        # Parse it back
        vgm = parse_vgm(vgm_bytes)
        assert vgm.header.has_ym2612
        assert vgm.header.version == 0x171

    def test_roundtrip_preserves_event_types(self):
        """Write events → VGM binary → parse back → same event types."""
        original = make_test_note(channel=0)
        vgm_bytes = events_to_vgm(original)
        parsed = parse_vgm(vgm_bytes)

        orig_types = [e.type for e in original if e.type != EventType.END]
        parsed_types = [e.type for e in parsed.events if e.type != EventType.END]

        # Types should match (DAC_WRITE gets encoded as YM2612_PORT0)
        for ot, pt in zip(orig_types, parsed_types):
            if ot == EventType.DAC_WRITE:
                assert pt == EventType.YM2612_PORT0
            else:
                assert ot == pt

    def test_roundtrip_preserves_register_writes(self):
        """Register addresses and values survive the round-trip."""
        original = make_test_note(channel=0, algorithm=5)
        vgm_bytes = events_to_vgm(original)
        parsed = parse_vgm(vgm_bytes)

        orig_regs = [
            (e.register, e.value) for e in original
            if e.type in (EventType.YM2612_PORT0, EventType.YM2612_PORT1)
        ]
        parsed_regs = [
            (e.register, e.value) for e in parsed.events
            if e.type in (EventType.YM2612_PORT0, EventType.YM2612_PORT1)
        ]

        assert orig_regs == parsed_regs

    def test_roundtrip_preserves_total_wait(self):
        """Total wait time (in samples) should be preserved."""
        original = make_test_note(duration_frames=120)
        vgm_bytes = events_to_vgm(original)
        parsed = parse_vgm(vgm_bytes)

        orig_wait = sum(e.value for e in original if e.type == EventType.WAIT)
        parsed_wait = sum(e.value for e in parsed.events if e.type == EventType.WAIT)
        assert orig_wait == parsed_wait

    def test_channel_4_uses_port1(self):
        """Channel 4 should use port 1 (channels 4-6)."""
        events = make_test_note(channel=4)
        port1_events = [e for e in events if e.type == EventType.YM2612_PORT1]
        assert len(port1_events) > 0

    def test_save_and_load(self, tmp_path):
        """Full save → load round-trip via filesystem."""
        events = make_test_note(channel=2, duration_frames=30)
        vgm_path = tmp_path / "test.vgm"
        save_vgm(events, vgm_path)

        loaded = load_vgm(vgm_path)
        assert loaded.header.has_ym2612
        assert len(loaded.events) > 0
        assert loaded.source_path == str(vgm_path)

    def test_multiple_notes_sequence(self):
        """Multiple notes in sequence should produce valid VGM."""
        events = []
        sample_pos = 0

        # C4, D4, E4 on channel 0
        freqs = [(4, 0x269), (4, 0x2AE), (4, 0x2F7)]
        for block, fnum in freqs:
            note = make_test_note(
                channel=0,
                frequency_block=block,
                frequency_num=fnum,
                duration_frames=30,
            )
            # Adjust sample positions
            for e in note:
                if e.type != EventType.END:
                    e.sample_pos += sample_pos
                    events.append(e)
            sample_pos = events[-1].sample_pos

        events.append(VgmEvent(type=EventType.END, sample_pos=sample_pos))

        vgm_bytes = events_to_vgm(events)
        parsed = parse_vgm(vgm_bytes)
        assert len(parsed.events) > 10  # should have many events


# ---------------------------------------------------------------------------
# Register name helper tests
# ---------------------------------------------------------------------------

class TestRegisterNames:
    def test_key_on_off(self):
        assert "Key" in ym2612_register_name(0x28, 0)

    def test_dac_data(self):
        assert "DAC" in ym2612_register_name(0x2A, 0)

    def test_algorithm(self):
        name = ym2612_register_name(0xB0, 0)
        assert "Algorithm" in name

    def test_frequency(self):
        name = ym2612_register_name(0xA0, 0)
        assert "Freq" in name

    def test_operator_param(self):
        name = ym2612_register_name(0x40, 0)  # TL for op1 ch1
        assert "TL" in name

    def test_port1_channel_offset(self):
        name = ym2612_register_name(0xB0, 1)
        assert "Ch4" in name


# ---------------------------------------------------------------------------
# Summary tests
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summarize_basic(self):
        events = make_test_note()
        vgm_bytes = events_to_vgm(events)
        vgm = parse_vgm(vgm_bytes)
        summary = summarize_vgm(vgm)

        assert summary["has_ym2612"] is True
        assert summary["duration_seconds"] > 0
        assert summary["ym2612_port0_writes"] > 0
        assert summary["total_events"] > 0
