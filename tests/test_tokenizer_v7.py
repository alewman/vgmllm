"""Tests for genesis_music.tokenizer_v7."""

import pytest

from genesis_music.tokenizer_v7 import (
    # Constants
    VOCAB_SIZE_V7, FM_PATCH_TOKENS_V7, RARE_TOKEN_IDS,
    PAD, BOS, EOS,
    PAN_OFF, PAN_LEFT, PAN_RIGHT, PAN_CENTER,
    LFO_OFF, LFO_ON_BASE,
    CH3_NORMAL_MODE, CH3_SPECIAL_MODE,
    DAC_DISABLE, DAC_ENABLE,
    LOOP_POINT, INSTRUMENT_CHANGE, DOWNBEAT, HALFBEAT, SEP,
    SSG_EG_BASE, TL_BUCKET_BASE, PITCH_WP_BASE,
    CH_FM_BASE, NOTE_ON, NOTE_OFF, PITCH_BASE, VEL_BASE,
    KEY_BASE, BAR, BEAT_BASE, TEMPO_BASE,
    # Functions
    encode_fm_patch_v7, decode_fm_patch_v7,
    _split_psg_note_by_vol_env, _split_fm_note_by_tl_env,
    _reclassify_fm6_notes, _collect_hw_events,
    TokenizerV7,
)
from genesis_music.vgm_parser import EventType, VgmEvent, VgmFile, VgmHeader
from genesis_music.ym2612 import (
    CH_DAC, CH_FM_0, CH_FM_5, CH_PSG_0,
    NoteEvent, Ym2612Patch,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _silence_patch(**kwargs) -> Ym2612Patch:
    defaults = dict(
        algorithm=0, feedback=0,
        tl=(80, 90, 100, 80),
        ar=(31, 31, 31, 31),
        dr=(5, 5, 5, 5),
        sr=(0, 0, 0, 0),
        rr=(7, 7, 7, 7),
        sl=(3, 3, 3, 3),
        mul=(1, 1, 1, 1),
        dt=(0, 0, 0, 0),
    )
    defaults.update(kwargs)
    return Ym2612Patch(**defaults)


def _make_vgm(events: list[VgmEvent], total_samples: int = 44100) -> VgmFile:
    header = VgmHeader(
        ym2612_clock=7_670_454,
        sn76489_clock=3_579_545,
        total_samples=total_samples,
    )
    return VgmFile(header=header, events=events)


def _fm0(reg: int, val: int, sample: int = 0) -> VgmEvent:
    return VgmEvent(type=EventType.YM2612_PORT0, register=reg, value=val, sample_pos=sample)


def _fm1(reg: int, val: int, sample: int = 0) -> VgmEvent:
    return VgmEvent(type=EventType.YM2612_PORT1, register=reg, value=val, sample_pos=sample)


# ---------------------------------------------------------------------------
# 1. Vocab constants
# ---------------------------------------------------------------------------

class TestVocabConstants:
    def test_vocab_size(self):
        assert VOCAB_SIZE_V7 == 1024

    def test_patch_tokens(self):
        assert FM_PATCH_TOKENS_V7 == 44

    def test_pan_range(self):
        assert PAN_OFF == 898
        assert PAN_CENTER == 901

    def test_lfo_range(self):
        assert LFO_OFF == 902
        assert LFO_ON_BASE == 903
        assert LFO_ON_BASE + 7 == 910

    def test_ch3_tokens(self):
        assert CH3_NORMAL_MODE == 911
        assert CH3_SPECIAL_MODE == 912

    def test_dac_tokens(self):
        assert DAC_DISABLE == 913
        assert DAC_ENABLE == 914

    def test_structural_tokens(self):
        assert LOOP_POINT == 915
        assert INSTRUMENT_CHANGE == 916
        assert DOWNBEAT == 917
        assert HALFBEAT == 918
        assert SEP == 919

    def test_ssg_eg_range(self):
        assert SSG_EG_BASE == 920
        assert SSG_EG_BASE + 15 == 935

    def test_reserved_ranges_non_overlapping(self):
        assert TL_BUCKET_BASE == 936
        assert TL_BUCKET_BASE + 7 == 943
        assert PITCH_WP_BASE == 944
        assert PITCH_WP_BASE + 15 == 959

    def test_rare_token_ids_coverage(self):
        # Hardware state tokens and structural tokens should be rare
        assert PAN_OFF in RARE_TOKEN_IDS
        assert INSTRUMENT_CHANGE in RARE_TOKEN_IDS
        assert DOWNBEAT in RARE_TOKEN_IDS
        assert HALFBEAT in RARE_TOKEN_IDS
        # Regular note tokens should NOT be rare
        assert NOTE_ON not in RARE_TOKEN_IDS
        assert PITCH_BASE not in RARE_TOKEN_IDS


# ---------------------------------------------------------------------------
# 2. FM patch round-trip  (44 tokens)
# ---------------------------------------------------------------------------

class TestFmPatchV7:
    def test_roundtrip_basic(self):
        patch = _silence_patch()
        toks = encode_fm_patch_v7(patch)
        assert len(toks) == 44
        decoded = decode_fm_patch_v7(toks, 0)
        assert decoded is not None
        assert decoded.algorithm == patch.algorithm
        assert decoded.feedback == patch.feedback
        assert decoded.tl == patch.tl
        assert decoded.ar == patch.ar

    def test_ssg_eg_preserved(self):
        patch = _silence_patch(ssg_eg=(8, 0, 12, 0))
        toks = encode_fm_patch_v7(patch)
        decoded = decode_fm_patch_v7(toks, 0)
        assert decoded is not None
        assert decoded.ssg_eg == (8, 0, 12, 0)

    def test_ssg_eg_zero_default(self):
        patch = _silence_patch()
        toks = encode_fm_patch_v7(patch)
        decoded = decode_fm_patch_v7(toks, 0)
        assert decoded is not None
        assert decoded.ssg_eg == (0, 0, 0, 0)

    def test_all_algorithms(self):
        for alg in range(8):
            patch = _silence_patch(algorithm=alg)
            toks = encode_fm_patch_v7(patch)
            decoded = decode_fm_patch_v7(toks, 0)
            assert decoded is not None
            assert decoded.algorithm == alg

    def test_ssg_eg_all_values(self):
        for val in range(16):
            patch = _silence_patch(ssg_eg=(val, val, val, val))
            toks = encode_fm_patch_v7(patch)
            decoded = decode_fm_patch_v7(toks, 0)
            assert decoded is not None
            assert decoded.ssg_eg[0] == val, f"SSG-EG value {val} not preserved"

    def test_decode_returns_none_on_bad_token(self):
        toks = [0] * 44  # PAD tokens are not a valid patch block
        result = decode_fm_patch_v7(toks, 0)
        assert result is None

    def test_decode_returns_none_on_truncated(self):
        patch = _silence_patch()
        toks = encode_fm_patch_v7(patch)[:20]  # truncated
        result = decode_fm_patch_v7(toks, 0)
        assert result is None

    def test_ssg_eg_token_range(self):
        """All SSG-EG tokens must fall within SSG_EG_BASE..SSG_EG_BASE+15."""
        patch = _silence_patch(ssg_eg=(15, 15, 15, 15))
        toks = encode_fm_patch_v7(patch)
        # SSG-EG tokens are at positions 13, 23, 33, 43 (every 10 starting at offset 13)
        ssg_positions = [13, 23, 33, 43]
        for pos in ssg_positions:
            assert SSG_EG_BASE <= toks[pos] < SSG_EG_BASE + 16, (
                f"Token at pos {pos} = {toks[pos]} out of SSG-EG range"
            )


# ---------------------------------------------------------------------------
# 3. PSG vol_envelope note splitting
# ---------------------------------------------------------------------------

class TestPsgNoteSplitting:
    def _make_psg_note(self, vel=15, vol_env=None, sample_on=0, sample_off=44100):
        n = NoteEvent(
            channel=CH_PSG_0, pitch=60, velocity=vel,
            sample_on=sample_on, sample_off=sample_off,
        )
        if vol_env:
            n.vol_envelope = vol_env
        return n

    def test_no_env_returns_single(self):
        note = self._make_psg_note()
        result = _split_psg_note_by_vol_env(note)
        assert len(result) == 1
        assert result[0] is note

    def test_single_bucket_change(self):
        # Start loud (att=0, bucket=7) → drop to quiet (att=8, bucket=3)
        note = self._make_psg_note(
            vel=15,
            vol_env=[(22050, 8)],   # halfway through, att=8 → bucket 4
            sample_off=44100,
        )
        result = _split_psg_note_by_vol_env(note)
        assert len(result) == 2
        assert result[0].sample_on == 0
        assert result[0].sample_off == 22050
        assert result[1].sample_on == 22050
        assert result[1].sample_off == 44100

    def test_no_change_same_bucket(self):
        # vel=14 (att=1) and vel=13 (att=2) both map to bucket 6 → no split
        # _vel_8bucket(14) = 14*7//15 = 6, _vel_8bucket(13) = 13*7//15 = 6
        note = self._make_psg_note(
            vel=14,
            vol_env=[(10000, 2)],  # att=2 → vel=13, still bucket 6
        )
        result = _split_psg_note_by_vol_env(note)
        assert len(result) == 1

    def test_max_8_segments(self):
        # Create 12 distinct bucket changes
        vol_env = []
        for i in range(1, 13):
            att = (i % 8) * 2   # cycles through attenuation 0-14
            vol_env.append((i * 3000, att))
        note = self._make_psg_note(vel=15, vol_env=vol_env, sample_off=44100)
        result = _split_psg_note_by_vol_env(note)
        assert len(result) <= 8

    def test_split_preserves_channel_and_pitch(self):
        note = self._make_psg_note(vel=15, vol_env=[(10000, 8)])
        result = _split_psg_note_by_vol_env(note)
        for seg in result:
            assert seg.channel == CH_PSG_0
            assert seg.pitch == 60

    def test_env_outside_note_window_ignored(self):
        # Envelope point before sample_on or after sample_off should be ignored
        note = self._make_psg_note(
            vel=15,
            vol_env=[(-100, 8), (50000, 8)],  # both outside window
            sample_off=44100,
        )
        result = _split_psg_note_by_vol_env(note)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# 4. FM TL envelope note splitting
# ---------------------------------------------------------------------------

class TestFmTlNoteSplitting:
    def _make_fm_note(self, patch=None, tl_env=None, vel=10):
        if patch is None:
            patch = _silence_patch(tl=(40, 40, 40, 40))  # alg=0, carrier=op3(idx3)
        n = NoteEvent(
            channel=0, pitch=60, velocity=vel,
            sample_on=0, sample_off=44100, patch=patch,
        )
        if tl_env:
            n.tl_envelope = tl_env
        return n

    def test_no_env_returns_single(self):
        note = self._make_fm_note()
        result = _split_fm_note_by_tl_env(note)
        assert len(result) == 1

    def test_no_patch_returns_single(self):
        n = NoteEvent(channel=0, pitch=60, velocity=10, sample_on=0, sample_off=44100)
        n.tl_envelope = [(10000, [40, 40, 40, 100])]
        result = _split_fm_note_by_tl_env(n)
        assert len(result) == 1

    def test_bucket_change_triggers_split(self):
        # alg=0: carrier is op3 (index 3)
        # Initial TL carrier=40 → bucket 2 (40*7//127)
        # New TL carrier=110 → bucket 6 (110*7//127=6)
        patch = _silence_patch(algorithm=0, tl=(40, 40, 40, 40))
        note = self._make_fm_note(
            patch=patch,
            tl_env=[(22050, [40, 40, 40, 110])],
        )
        result = _split_fm_note_by_tl_env(note)
        assert len(result) == 2

    def test_same_bucket_no_split(self):
        patch = _silence_patch(algorithm=0, tl=(40, 40, 40, 40))
        note = self._make_fm_note(
            patch=patch,
            tl_env=[(22050, [40, 40, 40, 42])],  # still bucket 2
        )
        result = _split_fm_note_by_tl_env(note)
        assert len(result) == 1

    def test_max_8_segments(self):
        patch = _silence_patch(algorithm=7, tl=(40, 40, 40, 40))
        # 12 alternating TL values: carriers = all 4 ops for alg 7
        tl_env = []
        for i in range(1, 13):
            tl = 20 + (i % 7) * 15
            tl_env.append((i * 3000, [tl, tl, tl, tl]))
        note = self._make_fm_note(patch=patch, tl_env=tl_env)
        result = _split_fm_note_by_tl_env(note)
        assert len(result) <= 8

    def test_split_preserves_patch(self):
        patch = _silence_patch()
        note = self._make_fm_note(patch=patch, tl_env=[(10000, [40, 40, 40, 110])])
        result = _split_fm_note_by_tl_env(note)
        for seg in result:
            assert seg.patch is patch


# ---------------------------------------------------------------------------
# 5. FM6 reclassification
# ---------------------------------------------------------------------------

class TestFm6Reclassification:
    def test_dac_with_fm_patch_reclassified(self):
        patch = _silence_patch()
        note = NoteEvent(
            channel=CH_DAC, pitch=60, velocity=10,
            sample_on=0, sample_off=44100,
            patch=patch, dac_sample_id=-1,
        )
        result = _reclassify_fm6_notes([note])
        assert len(result) == 1
        assert result[0].channel == CH_FM_5

    def test_real_dac_note_unchanged(self):
        note = NoteEvent(
            channel=CH_DAC, pitch=-1, velocity=15,
            sample_on=0, sample_off=1000,
            dac_sample_id=3,
        )
        result = _reclassify_fm6_notes([note])
        assert len(result) == 1
        assert result[0].channel == CH_DAC

    def test_dac_no_patch_unchanged(self):
        note = NoteEvent(
            channel=CH_DAC, pitch=60, velocity=10,
            sample_on=0, sample_off=44100,
            dac_sample_id=-1,  # no patch
        )
        result = _reclassify_fm6_notes([note])
        assert result[0].channel == CH_DAC

    def test_fm_channels_unchanged(self):
        for ch in range(6):
            note = NoteEvent(
                channel=ch, pitch=60, velocity=10,
                sample_on=0, sample_off=44100,
            )
            result = _reclassify_fm6_notes([note])
            assert result[0].channel == ch

    def test_reclassified_note_preserves_fields(self):
        patch = _silence_patch()
        note = NoteEvent(
            channel=CH_DAC, pitch=72, velocity=8,
            sample_on=100, sample_off=5000,
            patch=patch, dac_sample_id=-1,
        )
        result = _reclassify_fm6_notes([note])
        n = result[0]
        assert n.pitch == 72
        assert n.velocity == 8
        assert n.sample_on == 100
        assert n.sample_off == 5000
        assert n.patch is patch


# ---------------------------------------------------------------------------
# 6. Hardware event collection
# ---------------------------------------------------------------------------

class TestHwEventCollection:
    def test_pan_event_left(self):
        vgm = _make_vgm([
            _fm0(0xB4, 0b10_000000, sample=1000),  # port0, ch0: L-only
        ])
        events = _collect_hw_events(vgm)
        assert any(PAN_LEFT in e.tokens for e in events)

    def test_pan_event_right(self):
        vgm = _make_vgm([
            _fm0(0xB4, 0b01_000000, sample=500),   # ch0: R-only
        ])
        events = _collect_hw_events(vgm)
        assert any(PAN_RIGHT in e.tokens for e in events)

    def test_pan_event_center(self):
        vgm = _make_vgm([
            _fm0(0xB4, 0b11_000000, sample=500),   # ch0: both
        ])
        events = _collect_hw_events(vgm)
        assert any(PAN_CENTER in e.tokens for e in events)

    def test_pan_port1_maps_to_ch3(self):
        # port 1, reg 0xB4 → FM channel 3
        vgm = _make_vgm([
            _fm1(0xB4, 0b10_000000, sample=1000),  # port1, ch3: L-only
        ])
        events = _collect_hw_events(vgm)
        pan_events = [e for e in events if PAN_LEFT in e.tokens]
        assert pan_events
        # Channel token should be CH_FM_BASE + 3
        assert pan_events[0].tokens[0] == CH_FM_BASE + 3

    def test_lfo_on(self):
        vgm = _make_vgm([
            _fm0(0x22, 0b0000_1011, sample=100),   # LFO on, rate=3
        ])
        events = _collect_hw_events(vgm)
        assert any(LFO_ON_BASE + 3 in e.tokens for e in events)

    def test_lfo_off(self):
        vgm = _make_vgm([
            _fm0(0x22, 0b0000_0000, sample=100),   # LFO off
        ])
        events = _collect_hw_events(vgm)
        # LFO_OFF emitted when state differs from -1 (unknown initial)
        assert any(LFO_OFF in e.tokens for e in events)

    def test_ch3_special_mode(self):
        vgm = _make_vgm([
            _fm0(0x27, 0b0100_0000, sample=200),   # bit 6 = CH3 special
        ])
        events = _collect_hw_events(vgm)
        assert any(CH3_SPECIAL_MODE in e.tokens for e in events)

    def test_dac_enable(self):
        vgm = _make_vgm([
            _fm0(0x2B, 0b1000_0000, sample=300),   # bit 7 = DAC on
        ])
        events = _collect_hw_events(vgm)
        assert any(DAC_ENABLE in e.tokens for e in events)

    def test_loop_point_from_header(self):
        header = VgmHeader(
            ym2612_clock=7_670_454,
            total_samples=44100,
            loop_samples=22050,
        )
        vgm = VgmFile(header=header, events=[])
        events = _collect_hw_events(vgm)
        loop_events = [e for e in events if LOOP_POINT in e.tokens]
        assert len(loop_events) == 1
        assert loop_events[0].sample_pos == 44100 - 22050

    def test_no_duplicate_state_emits(self):
        # Same LFO rate emitted twice: should only produce 1 hw event
        vgm = _make_vgm([
            _fm0(0x22, 0b0000_1011, sample=100),
            _fm0(0x22, 0b0000_1011, sample=200),
        ])
        events = _collect_hw_events(vgm)
        lfo_events = [e for e in events if any(
            LFO_ON_BASE <= t <= LFO_ON_BASE + 7 for t in e.tokens
        )]
        assert len(lfo_events) == 1

    def test_sorted_by_sample_pos(self):
        vgm = _make_vgm([
            _fm0(0x2B, 0b1000_0000, sample=5000),
            _fm0(0x22, 0b0000_1011, sample=100),
        ])
        events = _collect_hw_events(vgm)
        positions = [e.sample_pos for e in events]
        assert positions == sorted(positions)


# ---------------------------------------------------------------------------
# 7. Transpose
# ---------------------------------------------------------------------------

class TestTranspose:
    def test_transpose_shifts_pitch(self):
        tok = TokenizerV7()
        tokens = [BOS, PITCH_BASE + 12, EOS]
        result = tok.transpose(tokens, 2)
        assert PITCH_BASE + 14 in result

    def test_transpose_shifts_key(self):
        tok = TokenizerV7()
        tokens = [KEY_BASE + 0]   # C major
        result = tok.transpose(tokens, 3)
        assert KEY_BASE + 3 in result

    def test_transpose_clamps_pitch(self):
        tok = TokenizerV7()
        tokens = [PITCH_BASE + 87]  # highest note
        result = tok.transpose(tokens, 10)
        assert result == [PITCH_BASE + 87]  # clamped

    def test_transpose_zero_identity(self):
        tok = TokenizerV7()
        tokens = [BOS, PITCH_BASE + 12, KEY_BASE + 0, EOS, DOWNBEAT, SSG_EG_BASE + 5]
        result = tok.transpose(tokens, 0)
        assert result == tokens

    def test_v7_tokens_not_shifted(self):
        """Non-pitch/key v7 tokens should pass through unchanged."""
        tok = TokenizerV7()
        v7_toks = [PAN_LEFT, LFO_ON_BASE + 3, CH3_SPECIAL_MODE, DOWNBEAT, SEP]
        result = tok.transpose(v7_toks, 7)
        assert result == v7_toks


# ---------------------------------------------------------------------------
# 8. Token stream  (encode / decode round-trip with SSG-EG)
# ---------------------------------------------------------------------------

class TestPatchRoundTrip:
    def test_encode_decode_fm_patch_v7_full(self):
        """All FM parameters survive a full encode→decode cycle."""
        patch = Ym2612Patch(
            algorithm=3, feedback=5,
            tl=(10, 30, 60, 90),
            ar=(31, 28, 20, 15),
            dr=(10, 8, 5, 3),
            sr=(2, 2, 1, 0),
            rr=(12, 10, 8, 6),
            sl=(3, 2, 1, 0),
            mul=(1, 2, 3, 4),
            dt=(0, 1, 2, 3),
            ks=(0, 1, 2, 3),
            ams=2, fms=4,
            ssg_eg=(8, 0, 10, 15),
        )
        toks = encode_fm_patch_v7(patch)
        assert len(toks) == FM_PATCH_TOKENS_V7
        decoded = decode_fm_patch_v7(toks, 0)
        assert decoded is not None
        assert decoded.algorithm == patch.algorithm
        assert decoded.feedback == patch.feedback
        assert decoded.tl == patch.tl
        assert decoded.ar == patch.ar
        assert decoded.dr == patch.dr
        assert decoded.sr == patch.sr
        assert decoded.rr == patch.rr
        assert decoded.sl == patch.sl
        assert decoded.mul == patch.mul
        assert decoded.dt == patch.dt
        assert decoded.ks == patch.ks
        assert decoded.ams == patch.ams
        assert decoded.fms == patch.fms
        assert decoded.ssg_eg == patch.ssg_eg
