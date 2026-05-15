"""Tests for the v2 tokenizer (note-level abstraction)."""

import math
import numpy as np
import pytest

from genesis_music.vgm_parser import EventType, VgmEvent, VgmFile, VgmHeader
from genesis_music.tokenizer_v2 import (
    PAD, BOS, EOS, UNK,
    NOTE_NAMES,
    midi_to_name,
    name_to_midi,
    fnum_block_to_midi,
    fnum_block_to_note,
    _fnum_block_to_freq,
    _midi_to_fnum_block,
    parse_key_on,
    freq_reg_to_channel,
    encode_events_v2,
    decode_token_str_v2,
    decode_tokens_v2,
    VocabV2,
    build_vocab_v2,
    encode_vgm_v2,
    decode_ids_v2,
    _build_wait_bins,
    quantize_wait,
    dequantize_wait,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_vgm_file(events: list[VgmEvent]) -> VgmFile:
    """Build a minimal VgmFile for testing."""
    total_samples = sum(e.value for e in events if e.type == EventType.WAIT)
    header = VgmHeader(
        version=0x171,
        ym2612_clock=7670453,
        sn76489_clock=3579545,
        total_samples=total_samples,
        data_offset=0x100,
    )
    return VgmFile(header=header, events=events, pcm_data=b"")


def _a4_freq_events(port: int = 0, ch_offset: int = 0) -> list[VgmEvent]:
    """Generate freq register writes for A4 (440 Hz) on specified channel.

    A4 at block=4: F-number ≈ 1082 -> MSB = 0x24 (block=4, fnum_hi=4),
    LSB = 0x3A (fnum_lo=58 -> 4*256+58=1082)
    """
    # F-number = 1082, block = 4
    # MSB: block=4 (bits 5-3 = 100), fnum_hi = 1082>>8 = 4 (bits 2-0 = 100)
    # So MSB = (4 << 3) | 4 = 0x24
    # LSB: 1082 & 0xFF = 0x3A
    fnum = 1082
    block = 4
    msb = ((block & 0x07) << 3) | ((fnum >> 8) & 0x07)
    lsb = fnum & 0xFF

    etype = EventType.YM2612_PORT0 if port == 0 else EventType.YM2612_PORT1
    return [
        VgmEvent(type=etype, register=0xA4 + ch_offset, value=msb),
        VgmEvent(type=etype, register=0xA0 + ch_offset, value=lsb),
    ]


# ---------------------------------------------------------------------------
# Note name conversion tests
# ---------------------------------------------------------------------------

class TestNoteNames:
    def test_midi_to_name_c4(self):
        assert midi_to_name(60) == "C4"

    def test_midi_to_name_a4(self):
        assert midi_to_name(69) == "A4"

    def test_midi_to_name_db5(self):
        assert midi_to_name(73) == "Db5"

    def test_name_to_midi_roundtrip(self):
        for midi in range(12, 120):
            name = midi_to_name(midi)
            assert name_to_midi(name) == midi

    def test_note_names_12(self):
        assert len(NOTE_NAMES) == 12


# ---------------------------------------------------------------------------
# YM2612 frequency conversion tests
# ---------------------------------------------------------------------------

class TestFreqConversion:
    def test_a4_fnum_to_freq(self):
        """F-number 1082, block 4 should be close to 440 Hz."""
        freq = _fnum_block_to_freq(1082, 4)
        assert abs(freq - 440.0) < 2.0  # Within 2 Hz

    def test_a4_fnum_to_midi(self):
        midi = fnum_block_to_midi(1082, 4)
        assert midi == 69  # A4

    def test_a4_fnum_to_note(self):
        note = fnum_block_to_note(1082, 4)
        assert note == "A4"

    def test_c4_fnum(self):
        """F-number ~644, block 4 should give C4."""
        midi = fnum_block_to_midi(644, 4)
        assert midi == 60  # C4

    def test_midi_to_fnum_roundtrip(self):
        """MIDI -> fnum+block -> freq -> MIDI should roundtrip."""
        for midi in range(24, 108):
            fnum, block = _midi_to_fnum_block(midi)
            assert 1 <= fnum <= 2047
            assert 0 <= block <= 7
            result_midi = fnum_block_to_midi(fnum, block)
            assert abs(result_midi - midi) <= 1, \
                f"MIDI {midi} -> fnum={fnum},block={block} -> MIDI {result_midi}"

    def test_zero_fnum(self):
        freq = _fnum_block_to_freq(0, 4)
        assert freq == 0.0

    def test_block_shifts_octave(self):
        """Same F-number at different blocks should differ by ~octave."""
        freq_b3 = _fnum_block_to_freq(1082, 3)
        freq_b4 = _fnum_block_to_freq(1082, 4)
        freq_b5 = _fnum_block_to_freq(1082, 5)
        assert abs(freq_b4 / freq_b3 - 2.0) < 0.01
        assert abs(freq_b5 / freq_b4 - 2.0) < 0.01


# ---------------------------------------------------------------------------
# Register helper tests
# ---------------------------------------------------------------------------

class TestRegisterHelpers:
    def test_parse_key_on_ch1(self):
        ch, is_on = parse_key_on(0xF0)  # All ops on, channel 0 -> ch1
        assert ch == 1
        assert is_on is True

    def test_parse_key_on_ch4(self):
        ch, is_on = parse_key_on(0xF4)  # All ops on, raw_ch=4 -> ch4
        assert ch == 4
        assert is_on is True

    def test_parse_key_off_ch1(self):
        ch, is_on = parse_key_on(0x00)
        assert ch == 1
        assert is_on is False

    def test_parse_key_off_ch6(self):
        ch, is_on = parse_key_on(0x06)
        assert ch == 6
        assert is_on is False

    def test_freq_reg_channel_port0(self):
        assert freq_reg_to_channel(0xA4, 0) == 1
        assert freq_reg_to_channel(0xA5, 0) == 2
        assert freq_reg_to_channel(0xA6, 0) == 3
        assert freq_reg_to_channel(0xA0, 0) == 1
        assert freq_reg_to_channel(0xA1, 0) == 2
        assert freq_reg_to_channel(0xA2, 0) == 3

    def test_freq_reg_channel_port1(self):
        assert freq_reg_to_channel(0xA4, 1) == 4
        assert freq_reg_to_channel(0xA5, 1) == 5
        assert freq_reg_to_channel(0xA6, 1) == 6


# ---------------------------------------------------------------------------
# Encode events v2 tests
# ---------------------------------------------------------------------------

class TestEncodeEventsV2:
    def test_simple_note_on_off(self):
        """Freq setup + key on -> CH:ON:note, key off -> CH:OFF."""
        events = [
            *_a4_freq_events(port=0, ch_offset=0),  # A4 on ch1
            VgmEvent(type=EventType.YM2612_PORT0, register=0x28, value=0xF0),
            VgmEvent(type=EventType.WAIT, value=735),
            VgmEvent(type=EventType.YM2612_PORT0, register=0x28, value=0x00),
            VgmEvent(type=EventType.END),
        ]
        tokens = encode_events_v2(events)
        assert "CH1:ON:A4" in tokens
        assert "CH1:OFF" in tokens
        # Frequency registers should NOT appear as raw tokens
        for t in tokens:
            assert not t.startswith("FM0:A4:"), f"Freq MSB leaked: {t}"
            assert not t.startswith("FM0:A0:"), f"Freq LSB leaked: {t}"

    def test_port1_channel(self):
        """Port 1 frequency writes should map to ch4-6."""
        events = [
            *_a4_freq_events(port=1, ch_offset=0),  # A4 on ch4
            VgmEvent(type=EventType.YM2612_PORT0, register=0x28, value=0xF4),
            VgmEvent(type=EventType.END),
        ]
        tokens = encode_events_v2(events)
        assert "CH4:ON:A4" in tokens

    def test_raw_register_passthrough(self):
        """Non-frequency, non-key-on registers should pass through raw."""
        events = [
            VgmEvent(type=EventType.YM2612_PORT0, register=0x30, value=0x71),
            VgmEvent(type=EventType.YM2612_PORT1, register=0xB0, value=0x3A),
            VgmEvent(type=EventType.END),
        ]
        tokens = encode_events_v2(events)
        assert "FM0:30:71" in tokens
        assert "FM1:B0:3A" in tokens

    def test_psg_passthrough(self):
        """SN76489 events should pass through as PSG tokens."""
        events = [
            VgmEvent(type=EventType.SN76489, register=0x80, value=0),
            VgmEvent(type=EventType.END),
        ]
        tokens = encode_events_v2(events)
        assert "PSG:80" in tokens

    def test_wait_encoding(self):
        """Waits should merge and quantize."""
        events = [
            VgmEvent(type=EventType.WAIT, value=735),
            VgmEvent(type=EventType.WAIT, value=735),
            VgmEvent(type=EventType.YM2612_PORT0, register=0x30, value=0x00),
            VgmEvent(type=EventType.END),
        ]
        tokens = encode_events_v2(events)
        wait_tokens = [t for t in tokens if t.startswith("<WAIT:")]
        # Two waits should merge into one
        assert len(wait_tokens) == 1

    def test_mid_note_pitch_change(self):
        """Frequency change while note is ON should emit PITCH token."""
        events = [
            *_a4_freq_events(port=0, ch_offset=0),  # A4 on ch1
            VgmEvent(type=EventType.YM2612_PORT0, register=0x28, value=0xF0),  # key on
            VgmEvent(type=EventType.WAIT, value=735),
            # Change to B4 (F-number ~1215, block 4)
            VgmEvent(type=EventType.YM2612_PORT0, register=0xA4, value=0x24),  # same block
            VgmEvent(type=EventType.YM2612_PORT0, register=0xA0, value=0xBF),  # fnum LSB for ~1215
            VgmEvent(type=EventType.WAIT, value=735),
            VgmEvent(type=EventType.YM2612_PORT0, register=0x28, value=0x00),
            VgmEvent(type=EventType.END),
        ]
        tokens = encode_events_v2(events)
        assert "CH1:ON:A4" in tokens
        # Should have a PITCH token for the mid-note change
        pitch_tokens = [t for t in tokens if t.startswith("CH1:PITCH:")]
        assert len(pitch_tokens) >= 1

    def test_no_pitch_when_note_off(self):
        """Frequency changes while note is OFF should NOT emit PITCH."""
        events = [
            # Set freq with note OFF
            *_a4_freq_events(port=0, ch_offset=0),
            VgmEvent(type=EventType.END),
        ]
        tokens = encode_events_v2(events)
        # No CH tokens at all since no key on
        ch_tokens = [t for t in tokens if t.startswith("CH")]
        assert len(ch_tokens) == 0

    def test_dac_filtered_by_default(self):
        """DAC writes should be filtered by default."""
        events = [
            VgmEvent(type=EventType.DAC_WRITE, register=0, value=0x80),
            VgmEvent(type=EventType.END),
        ]
        tokens = encode_events_v2(events)
        assert len(tokens) == 0

    def test_dac_included_when_requested(self):
        events = [
            VgmEvent(type=EventType.DAC_WRITE, register=0, value=0x80),
            VgmEvent(type=EventType.END),
        ]
        tokens = encode_events_v2(events, include_dac=True)
        assert "FM0:2A:80" in tokens

    def test_multi_channel(self):
        """Multiple channels playing simultaneously."""
        events = [
            # Setup ch1 on A4
            *_a4_freq_events(port=0, ch_offset=0),
            VgmEvent(type=EventType.YM2612_PORT0, register=0x28, value=0xF0),
            # Setup ch4 on A4
            *_a4_freq_events(port=1, ch_offset=0),
            VgmEvent(type=EventType.YM2612_PORT0, register=0x28, value=0xF4),
            VgmEvent(type=EventType.WAIT, value=735),
            VgmEvent(type=EventType.YM2612_PORT0, register=0x28, value=0x00),
            VgmEvent(type=EventType.YM2612_PORT0, register=0x28, value=0x04),
            VgmEvent(type=EventType.END),
        ]
        tokens = encode_events_v2(events)
        assert "CH1:ON:A4" in tokens
        assert "CH4:ON:A4" in tokens
        assert "CH1:OFF" in tokens
        assert "CH4:OFF" in tokens


# ---------------------------------------------------------------------------
# Decode tests
# ---------------------------------------------------------------------------

class TestDecodeV2:
    def test_decode_note_on(self):
        """CH1:ON:A4 should produce freq writes + key on."""
        events = decode_token_str_v2("CH1:ON:A4")
        assert len(events) == 3  # MSB, LSB, Key On
        # Last event should be key on
        assert events[2].type == EventType.YM2612_PORT0
        assert events[2].register == 0x28
        assert (events[2].value & 0x0F) == 0  # channel 0 (ch1)
        assert (events[2].value >> 4) == 0x0F  # all operators on

    def test_decode_note_off(self):
        events = decode_token_str_v2("CH1:OFF")
        assert len(events) == 1
        assert events[0].register == 0x28
        assert events[0].value == 0  # raw_ch = 0, ops = 0

    def test_decode_note_off_ch4(self):
        events = decode_token_str_v2("CH4:OFF")
        assert len(events) == 1
        assert events[0].value == 4  # raw_ch = 4

    def test_decode_pitch(self):
        events = decode_token_str_v2("CH2:PITCH:C5")
        assert len(events) == 2  # MSB, LSB
        assert events[0].register == 0xA5  # ch2 offset = 1

    def test_decode_raw_fm(self):
        events = decode_token_str_v2("FM0:30:71")
        assert len(events) == 1
        assert events[0].type == EventType.YM2612_PORT0
        assert events[0].register == 0x30
        assert events[0].value == 0x71

    def test_decode_raw_fm1(self):
        events = decode_token_str_v2("FM1:B0:3A")
        assert len(events) == 1
        assert events[0].type == EventType.YM2612_PORT1
        assert events[0].register == 0xB0
        assert events[0].value == 0x3A

    def test_decode_psg(self):
        events = decode_token_str_v2("PSG:80")
        assert len(events) == 1
        assert events[0].type == EventType.SN76489
        assert events[0].register == 0x80

    def test_decode_wait(self):
        events = decode_token_str_v2("<WAIT:10>")
        assert len(events) == 1
        assert events[0].type == EventType.WAIT

    def test_decode_special_empty(self):
        assert decode_token_str_v2("<BOS>") == []
        assert decode_token_str_v2("<EOS>") == []
        assert decode_token_str_v2("<PAD>") == []


# ---------------------------------------------------------------------------
# Roundtrip tests (critical!)
# ---------------------------------------------------------------------------

class TestRoundtrip:
    def test_note_on_off_roundtrip(self):
        """Encode a note on/off, decode, and verify the note is preserved."""
        events = [
            *_a4_freq_events(port=0, ch_offset=0),
            VgmEvent(type=EventType.YM2612_PORT0, register=0x28, value=0xF0),
            VgmEvent(type=EventType.WAIT, value=44100),
            VgmEvent(type=EventType.YM2612_PORT0, register=0x28, value=0x00),
            VgmEvent(type=EventType.END),
        ]
        token_strs = encode_events_v2(events)
        reconstructed = decode_tokens_v2(token_strs)

        # Find key on event
        key_ons = [e for e in reconstructed
                   if e.type == EventType.YM2612_PORT0 and e.register == 0x28
                   and (e.value >> 4) != 0]
        assert len(key_ons) == 1

        # Frequency should reconstruct close to A4 (440 Hz)
        freq_msb_events = [e for e in reconstructed
                           if e.register in (0xA4, 0xA5, 0xA6)]
        freq_lsb_events = [e for e in reconstructed
                           if e.register in (0xA0, 0xA1, 0xA2)]
        assert len(freq_msb_events) >= 1
        assert len(freq_lsb_events) >= 1

        msb = freq_msb_events[0].value
        lsb = freq_lsb_events[0].value
        block = (msb >> 3) & 0x07
        fnum = ((msb & 0x07) << 8) | lsb
        freq = _fnum_block_to_freq(fnum, block)
        # Should be within 1 semitone of A4
        midi = fnum_block_to_midi(fnum, block)
        assert abs(midi - 69) <= 1, f"Expected A4 (69), got {midi} (freq={freq:.1f})"

    def test_raw_register_roundtrip(self):
        """Raw registers should survive encode->decode perfectly."""
        events = [
            VgmEvent(type=EventType.YM2612_PORT0, register=0x30, value=0x71),
            VgmEvent(type=EventType.YM2612_PORT1, register=0xB0, value=0x3A),
            VgmEvent(type=EventType.SN76489, register=0x80, value=0),
            VgmEvent(type=EventType.END),
        ]
        token_strs = encode_events_v2(events)
        reconstructed = decode_tokens_v2(token_strs)

        # Filter out the END event
        recon = [e for e in reconstructed if e.type != EventType.END]
        assert len(recon) == 3
        assert recon[0].register == 0x30
        assert recon[0].value == 0x71
        assert recon[1].register == 0xB0
        assert recon[1].value == 0x3A
        assert recon[2].type == EventType.SN76489

    def test_port1_note_roundtrip(self):
        """Port 1 channel note on should roundtrip to correct port."""
        events = [
            *_a4_freq_events(port=1, ch_offset=2),  # ch6
            VgmEvent(type=EventType.YM2612_PORT0, register=0x28, value=0xF6),
            VgmEvent(type=EventType.END),
        ]
        token_strs = encode_events_v2(events)
        assert "CH6:ON:A4" in token_strs

        reconstructed = decode_tokens_v2(token_strs)
        # Freq writes should be on port 1
        freq_events = [e for e in reconstructed if e.register in (0xA4, 0xA5, 0xA6)]
        assert any(e.type == EventType.YM2612_PORT1 for e in freq_events)
        # Key on should reference raw_ch = 6
        key_on = [e for e in reconstructed if e.register == 0x28 and (e.value >> 4) != 0]
        assert key_on[0].value & 0x07 == 6


# ---------------------------------------------------------------------------
# VocabV2 tests
# ---------------------------------------------------------------------------

class TestVocabV2:
    def _make_temp_vocab(self, tmp_path) -> VocabV2:
        """Create a small manual vocab for testing."""
        bins = _build_wait_bins(8)  # Small for testing
        token_to_id = {
            "<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3,
        }
        id_to_token = {v: k for k, v in token_to_id.items()}

        wait_offset = 4
        for i in range(len(bins)):
            name = f"<WAIT:{i}>"
            tid = wait_offset + i
            token_to_id[name] = tid
            id_to_token[tid] = name

        event_offset = wait_offset + len(bins)
        event_tokens = ["CH1:ON:A4", "CH1:OFF", "FM0:30:71", "FM1:B0:3A"]
        for i, t in enumerate(event_tokens):
            tid = event_offset + i
            token_to_id[t] = tid
            id_to_token[tid] = t

        return VocabV2(
            wait_bins=bins,
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            wait_offset=wait_offset,
            event_offset=event_offset,
        )

    def test_encode_decode(self, tmp_path):
        vocab = self._make_temp_vocab(tmp_path)
        assert vocab.encode("CH1:ON:A4") == vocab.event_offset
        assert vocab.decode(vocab.event_offset) == "CH1:ON:A4"

    def test_unknown_returns_unk(self, tmp_path):
        vocab = self._make_temp_vocab(tmp_path)
        assert vocab.encode("CH5:ON:Z9") == UNK

    def test_save_load_roundtrip(self, tmp_path):
        vocab = self._make_temp_vocab(tmp_path)
        path = tmp_path / "vocab_v2.json"
        vocab.save(path)
        loaded = VocabV2.load(path)
        assert loaded.size == vocab.size
        assert loaded.token_to_id == vocab.token_to_id
        assert loaded.wait_offset == vocab.wait_offset
        assert loaded.event_offset == vocab.event_offset
        np.testing.assert_array_equal(loaded.wait_bins, vocab.wait_bins)

    def test_size(self, tmp_path):
        vocab = self._make_temp_vocab(tmp_path)
        # 4 special + 8 wait + 4 event = 16
        assert vocab.size == 16


# ---------------------------------------------------------------------------
# Full pipeline encode/decode with VocabV2
# ---------------------------------------------------------------------------

class TestFullPipelineV2:
    def _make_vocab_with_tokens(self, token_strs: list[str]) -> VocabV2:
        """Build a VocabV2 that contains the given token strings."""
        bins = _build_wait_bins()
        token_to_id = {k: v for k, v in {
            "<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3,
        }.items()}
        id_to_token = {v: k for k, v in token_to_id.items()}
        wait_offset = 4
        for i in range(len(bins)):
            name = f"<WAIT:{i}>"
            tid = wait_offset + i
            token_to_id[name] = tid
            id_to_token[tid] = name

        event_offset = wait_offset + len(bins)
        for i, t in enumerate(sorted(set(token_strs))):
            tid = event_offset + i
            token_to_id[t] = tid
            id_to_token[tid] = t

        return VocabV2(
            wait_bins=bins,
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            wait_offset=wait_offset,
            event_offset=event_offset,
        )

    def test_encode_vgm_v2(self):
        """Full encode through vocab: VGM -> IDs."""
        events = [
            *_a4_freq_events(port=0, ch_offset=0),
            VgmEvent(type=EventType.YM2612_PORT0, register=0x28, value=0xF0),
            VgmEvent(type=EventType.WAIT, value=735),
            VgmEvent(type=EventType.YM2612_PORT0, register=0x28, value=0x00),
            VgmEvent(type=EventType.END),
        ]
        vgm = _make_vgm_file(events)

        # First get token strings to build vocab
        strs = encode_events_v2(events)
        vocab = self._make_vocab_with_tokens(strs)

        ids = encode_vgm_v2(vgm, vocab)
        assert ids[0] == BOS
        assert ids[-1] == EOS
        assert UNK not in ids

    def test_decode_ids_v2(self):
        """Full decode: IDs -> VGM events."""
        events = [
            *_a4_freq_events(port=0, ch_offset=0),
            VgmEvent(type=EventType.YM2612_PORT0, register=0x28, value=0xF0),
            VgmEvent(type=EventType.WAIT, value=735),
            VgmEvent(type=EventType.YM2612_PORT0, register=0x28, value=0x00),
            VgmEvent(type=EventType.END),
        ]
        vgm = _make_vgm_file(events)
        strs = encode_events_v2(events)
        vocab = self._make_vocab_with_tokens(strs)

        ids = encode_vgm_v2(vgm, vocab)
        decoded = decode_ids_v2(ids, vocab)

        # Should have key on and key off
        key_events = [e for e in decoded if e.register == 0x28]
        assert len(key_events) == 2  # on + off

    def test_no_unk_leakage(self):
        """All tokens from encoding should be in vocab (no UNK)."""
        events = [
            VgmEvent(type=EventType.YM2612_PORT0, register=0x30, value=0x71),
            VgmEvent(type=EventType.YM2612_PORT0, register=0x40, value=0x23),
            *_a4_freq_events(port=0, ch_offset=0),
            VgmEvent(type=EventType.YM2612_PORT0, register=0x28, value=0xF0),
            VgmEvent(type=EventType.WAIT, value=44100),
            VgmEvent(type=EventType.YM2612_PORT0, register=0x28, value=0x00),
            VgmEvent(type=EventType.SN76489, register=0x80, value=0),
            VgmEvent(type=EventType.END),
        ]
        vgm = _make_vgm_file(events)
        strs = encode_events_v2(events)
        vocab = self._make_vocab_with_tokens(strs)
        ids = encode_vgm_v2(vgm, vocab)
        assert UNK not in ids


# ---------------------------------------------------------------------------
# Wait bin tests (sanity check same as v1)
# ---------------------------------------------------------------------------

class TestWaitBins:
    def test_bins_monotonic(self):
        bins = _build_wait_bins()
        assert all(bins[i] < bins[i + 1] for i in range(len(bins) - 1))

    def test_quantize_dequantize(self):
        bins = _build_wait_bins()
        for samples in [1, 100, 735, 44100, 1_000_000]:
            idx = quantize_wait(samples, bins)
            result = dequantize_wait(idx, bins)
            # Should be close-ish (within a bin)
            assert result > 0
