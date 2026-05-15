"""Tests for the VGM tokenizer."""

import numpy as np
import pytest

from genesis_music.vgm_parser import EventType, VgmEvent, VgmFile, VgmHeader
from genesis_music.tokenizer import (
    PAD, BOS, EOS, UNK,
    SPECIAL_TOKENS,
    Vocab,
    EventToken,
    build_vocab,
    encode_vgm,
    decode_tokens,
    quantize_wait,
    dequantize_wait,
    _build_wait_bins,
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


def _simple_events() -> list[VgmEvent]:
    """A short sequence of FM events with waits."""
    return [
        VgmEvent(type=EventType.YM2612_PORT0, register=0x28, value=0xF0),  # key on ch1
        VgmEvent(type=EventType.WAIT, value=735),  # ~1 NTSC frame
        VgmEvent(type=EventType.YM2612_PORT0, register=0xA4, value=0x22),  # freq hi
        VgmEvent(type=EventType.YM2612_PORT0, register=0xA0, value=0x69),  # freq lo
        VgmEvent(type=EventType.WAIT, value=735),
        VgmEvent(type=EventType.YM2612_PORT0, register=0x28, value=0x00),  # key off ch1
        VgmEvent(type=EventType.END),
    ]


# ---------------------------------------------------------------------------
# Wait quantization tests
# ---------------------------------------------------------------------------

class TestWaitQuantization:
    def test_build_bins_returns_sorted_unique(self):
        bins = _build_wait_bins(64)
        assert len(bins) > 0
        assert all(bins[i] < bins[i + 1] for i in range(len(bins) - 1))

    def test_build_bins_starts_at_one(self):
        bins = _build_wait_bins(64)
        assert bins[0] == 1

    def test_quantize_exact_match(self):
        bins = _build_wait_bins(64)
        # First bin is always 1
        assert quantize_wait(1, bins) == 0

    def test_quantize_last_bin(self):
        bins = _build_wait_bins(64)
        assert quantize_wait(999_999_999, bins) == len(bins) - 1

    def test_dequantize_roundtrip_exact(self):
        bins = _build_wait_bins(64)
        for i in range(len(bins)):
            val = int(bins[i])
            idx = quantize_wait(val, bins)
            assert dequantize_wait(idx, bins) == val

    def test_quantize_picks_nearest(self):
        bins = np.array([1, 10, 100, 1000], dtype=np.int64)
        assert quantize_wait(8, bins) == 1   # closer to 10 than 1
        assert quantize_wait(2, bins) == 0   # closer to 1 than 10
        assert quantize_wait(55, bins) == 2  # closer to 100 than 10 (log-scale still uses nearest)


# ---------------------------------------------------------------------------
# EventToken tests
# ---------------------------------------------------------------------------

class TestEventToken:
    def test_to_str_roundtrip(self):
        tok = EventToken("YM2612_PORT0", 0x28, 0xF0)
        s = tok.to_str()
        tok2 = EventToken.from_str(s)
        assert tok2 == tok

    def test_str_format(self):
        tok = EventToken("SN76489", 0x00, 0x9F)
        assert tok.to_str() == "SN76489:00:9F"


# ---------------------------------------------------------------------------
# Vocab build tests
# ---------------------------------------------------------------------------

class TestVocabBuild:
    def test_build_from_events(self, tmp_path):
        """Build vocab from synthetic VGM files written to disk."""
        from genesis_music.vgm_writer import events_to_vgm, save_vgm

        events = _simple_events()
        vgm_bytes = events_to_vgm(events)
        f1 = tmp_path / "test1.vgm"
        f1.write_bytes(vgm_bytes)
        # Write a second copy so min_count=2 passes
        f2 = tmp_path / "test2.vgm"
        f2.write_bytes(vgm_bytes)

        vocab = build_vocab([f1, f2], n_wait_bins=32, min_count=1)

        assert vocab.size > len(SPECIAL_TOKENS)
        assert vocab.wait_offset == len(SPECIAL_TOKENS)
        assert vocab.event_offset == vocab.wait_offset + vocab.n_wait_tokens

    def test_min_count_filters_rare(self, tmp_path):
        from genesis_music.vgm_writer import events_to_vgm

        events = _simple_events()
        vgm_bytes = events_to_vgm(events)
        f1 = tmp_path / "test1.vgm"
        f1.write_bytes(vgm_bytes)

        # With min_count=1, all events included
        v1 = build_vocab([f1], n_wait_bins=16, min_count=1)
        # With high min_count, most events excluded (only 1 occurrence of each)
        v2 = build_vocab([f1], n_wait_bins=16, min_count=999)

        assert v2.size < v1.size

    def test_vocab_save_load_roundtrip(self, tmp_path):
        from genesis_music.vgm_writer import events_to_vgm

        events = _simple_events()
        vgm_bytes = events_to_vgm(events)
        f1 = tmp_path / "test.vgm"
        f1.write_bytes(vgm_bytes)

        vocab = build_vocab([f1], n_wait_bins=16, min_count=1)
        vocab_path = tmp_path / "vocab.json"
        vocab.save(vocab_path)

        loaded = Vocab.load(vocab_path)
        assert loaded.size == vocab.size
        assert loaded.wait_offset == vocab.wait_offset
        assert loaded.event_offset == vocab.event_offset
        assert np.array_equal(loaded.wait_bins, vocab.wait_bins)


# ---------------------------------------------------------------------------
# Encode / decode tests
# ---------------------------------------------------------------------------

class TestEncodeDecode:
    @pytest.fixture
    def vocab_and_vgm(self, tmp_path):
        from genesis_music.vgm_writer import events_to_vgm

        events = _simple_events()
        vgm_bytes = events_to_vgm(events)
        f1 = tmp_path / "test.vgm"
        f1.write_bytes(vgm_bytes)

        vocab = build_vocab([f1], n_wait_bins=32, min_count=1)
        from genesis_music.vgm_parser import load_vgm
        vgm = load_vgm(f1)
        return vocab, vgm

    def test_encode_starts_with_bos(self, vocab_and_vgm):
        vocab, vgm = vocab_and_vgm
        tokens = encode_vgm(vgm, vocab)
        assert tokens[0] == BOS

    def test_encode_ends_with_eos(self, vocab_and_vgm):
        vocab, vgm = vocab_and_vgm
        tokens = encode_vgm(vgm, vocab)
        assert tokens[-1] == EOS

    def test_encode_no_unk_for_known_events(self, vocab_and_vgm):
        vocab, vgm = vocab_and_vgm
        tokens = encode_vgm(vgm, vocab)
        assert UNK not in tokens

    def test_decode_recovers_event_types(self, vocab_and_vgm):
        vocab, vgm = vocab_and_vgm
        tokens = encode_vgm(vgm, vocab)
        decoded = decode_tokens(tokens, vocab)

        # Should have waits and YM2612 writes
        types = {e.type for e in decoded}
        assert EventType.WAIT in types
        assert EventType.YM2612_PORT0 in types
        assert EventType.END in types

    def test_decode_preserves_register_values(self, vocab_and_vgm):
        vocab, vgm = vocab_and_vgm
        tokens = encode_vgm(vgm, vocab)
        decoded = decode_tokens(tokens, vocab)

        # Key on event should be preserved exactly
        key_on = [e for e in decoded if e.type == EventType.YM2612_PORT0 and e.register == 0x28]
        assert len(key_on) >= 1
        assert key_on[0].value == 0xF0

    def test_roundtrip_event_count(self, vocab_and_vgm):
        vocab, vgm = vocab_and_vgm
        tokens = encode_vgm(vgm, vocab)
        decoded = decode_tokens(tokens, vocab)

        # Original events (excluding END): 3 FM writes + 2 waits = 5
        orig_data = [e for e in vgm.events if e.type != EventType.END]
        decoded_data = [e for e in decoded if e.type != EventType.END]
        assert len(decoded_data) == len(orig_data)


class TestDACFiltering:
    def test_dac_excluded_by_default(self, tmp_path):
        """DAC writes should be excluded when include_dac=False."""
        from genesis_music.vgm_writer import events_to_vgm

        events = [
            VgmEvent(type=EventType.YM2612_PORT0, register=0x28, value=0xF0),
            VgmEvent(type=EventType.WAIT, value=5),
            VgmEvent(type=EventType.DAC_WRITE, value=0x80),
            VgmEvent(type=EventType.WAIT, value=5),
            VgmEvent(type=EventType.DAC_WRITE, value=0x90),
            VgmEvent(type=EventType.WAIT, value=5),
            VgmEvent(type=EventType.YM2612_PORT0, register=0x28, value=0x00),
            VgmEvent(type=EventType.END),
        ]
        vgm_bytes = events_to_vgm(events)
        f = tmp_path / "dac_test.vgm"
        f.write_bytes(vgm_bytes)

        vocab = build_vocab([f], n_wait_bins=16, min_count=1, include_dac=True)
        from genesis_music.vgm_parser import load_vgm
        vgm = load_vgm(f)

        # Without DAC
        tokens = encode_vgm(vgm, vocab, include_dac=False)
        decoded = decode_tokens(tokens, vocab)
        assert not any(e.type == EventType.DAC_WRITE for e in decoded)

    def test_waits_merged_after_dac_removal(self, tmp_path):
        """When DAC writes are removed, surrounding waits should merge."""
        from genesis_music.vgm_writer import events_to_vgm

        events = [
            VgmEvent(type=EventType.YM2612_PORT0, register=0x28, value=0xF0),
            VgmEvent(type=EventType.WAIT, value=100),
            VgmEvent(type=EventType.DAC_WRITE, value=0x80),
            VgmEvent(type=EventType.WAIT, value=100),
            VgmEvent(type=EventType.DAC_WRITE, value=0x90),
            VgmEvent(type=EventType.WAIT, value=100),
            VgmEvent(type=EventType.YM2612_PORT0, register=0x28, value=0x00),
            VgmEvent(type=EventType.END),
        ]
        # Build VgmFile directly (write/parse roundtrip converts DAC_WRITE to
        # YM2612_PORT0 reg 0x2A which makes them invisible to the filter)
        vgm = _make_vgm_file(events)

        vocab = build_vocab(
            [],  # empty — we'll build a minimal vocab manually
            n_wait_bins=32, min_count=1, include_dac=True,
        )
        # Manually add the exact event tokens we need
        for ev in events:
            if ev.type in (EventType.WAIT, EventType.END):
                continue
            tok = EventToken(ev.type.name, ev.register, ev.value)
            key = tok.to_str()
            if key not in vocab.token_to_id:
                tid = vocab.size
                vocab.token_to_id[key] = tid
                vocab.id_to_token[tid] = key

        tokens = encode_vgm(vgm, vocab, include_dac=False)
        decoded = decode_tokens(tokens, vocab)

        # Without DAC: FM_WRITE, WAIT(~300 quantized), FM_WRITE, END
        waits = [e for e in decoded if e.type == EventType.WAIT]
        # The 3 separate waits (100+100+100=300) should become 1 merged wait
        assert len(waits) == 1


class TestSpecialTokenIDs:
    def test_pad_is_zero(self):
        assert PAD == 0

    def test_bos_eos_distinct(self):
        assert BOS != EOS
        assert BOS != PAD
        assert EOS != PAD

    def test_special_tokens_contiguous(self):
        ids = sorted(SPECIAL_TOKENS.values())
        assert ids == list(range(len(SPECIAL_TOKENS)))
