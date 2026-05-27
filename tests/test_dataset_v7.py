"""Tests for v7 dataset helpers: augmentation functions and VgmDatasetV7."""

import numpy as np
import pytest
import torch

from genesis_music.dataset_v7 import (
    VgmDatasetV7,
    _augment_tempo,
    _augment_velocity,
)
from genesis_music.tokenizer_v7 import TEMPO_BASE, VEL_BASE
from genesis_music.music_analysis import TEMPO_BINS


# ---------------------------------------------------------------------------
# _augment_tempo
# ---------------------------------------------------------------------------

class TestAugmentTempo:
    def _make_seq(self, bpm_idx: int) -> list[int]:
        """Build a tiny token list containing one TEMPO token."""
        return [1, TEMPO_BASE + bpm_idx, 2]

    def test_tempo_token_changes_on_nonunit_factor(self):
        # 120 BPM is index 60 in TEMPO_BINS (60+60=120)
        bpm_idx = TEMPO_BINS.index(120)
        seq = self._make_seq(bpm_idx)
        result = _augment_tempo(seq, 1.1)
        new_tempo_tok = result[1]
        assert new_tempo_tok != TEMPO_BASE + bpm_idx
        new_bpm = TEMPO_BINS[new_tempo_tok - TEMPO_BASE]
        assert abs(new_bpm - 120 * 1.1) < 2.0  # within 2 BPM

    def test_tempo_unchanged_on_factor_one(self):
        bpm_idx = TEMPO_BINS.index(100)
        seq = self._make_seq(bpm_idx)
        result = _augment_tempo(seq, 1.0)
        assert result[1] == TEMPO_BASE + bpm_idx

    def test_non_tempo_tokens_unchanged(self):
        seq = [0, 1, 2, 500, 999]
        result = _augment_tempo(seq, 0.8)
        non_tempo = [t for t in seq if not (TEMPO_BASE <= t < TEMPO_BASE + len(TEMPO_BINS))]
        result_non_tempo = [t for t in result if not (TEMPO_BASE <= t < TEMPO_BASE + len(TEMPO_BINS))]
        assert non_tempo == result_non_tempo

    def test_tempo_clamped_at_lower_bound(self):
        # Lowest TEMPO_BINS entry is 60 BPM; shifting it down by 20% hits the floor
        bpm_idx = 0  # 60 BPM
        seq = self._make_seq(bpm_idx)
        result = _augment_tempo(seq, 0.5)
        # Must still be a valid TEMPO token
        assert TEMPO_BASE <= result[1] < TEMPO_BASE + len(TEMPO_BINS)

    def test_tempo_clamped_at_upper_bound(self):
        # Highest TEMPO_BINS entry is 240 BPM; shifting up clamps to 240
        bpm_idx = len(TEMPO_BINS) - 1  # 240 BPM
        seq = self._make_seq(bpm_idx)
        result = _augment_tempo(seq, 1.5)
        assert TEMPO_BASE <= result[1] < TEMPO_BASE + len(TEMPO_BINS)
        assert result[1] == TEMPO_BASE + (len(TEMPO_BINS) - 1)

    def test_empty_sequence(self):
        assert _augment_tempo([], 1.1) == []

    def test_multiple_tempo_tokens(self):
        bpm_60 = TEMPO_BINS.index(60)
        bpm_120 = TEMPO_BINS.index(120)
        seq = [TEMPO_BASE + bpm_60, TEMPO_BASE + bpm_120]
        result = _augment_tempo(seq, 1.0)
        assert len(result) == 2
        assert result[0] == TEMPO_BASE + bpm_60
        assert result[1] == TEMPO_BASE + bpm_120


# ---------------------------------------------------------------------------
# _augment_velocity
# ---------------------------------------------------------------------------

class TestAugmentVelocity:
    def test_vel_token_shifted_up(self):
        seq = [VEL_BASE + 5]
        result = _augment_velocity(seq, +1)
        assert result == [VEL_BASE + 6]

    def test_vel_token_shifted_down(self):
        seq = [VEL_BASE + 5]
        result = _augment_velocity(seq, -1)
        assert result == [VEL_BASE + 4]

    def test_clamp_at_upper_bound(self):
        seq = [VEL_BASE + 15]
        result = _augment_velocity(seq, +1)
        assert result == [VEL_BASE + 15]  # clamped

    def test_clamp_at_lower_bound(self):
        seq = [VEL_BASE + 0]
        result = _augment_velocity(seq, -1)
        assert result == [VEL_BASE + 0]  # clamped

    def test_non_vel_tokens_unchanged(self):
        seq = [0, 100, VEL_BASE + 8, 900]
        result = _augment_velocity(seq, +1)
        assert result[0] == 0
        assert result[1] == 100
        assert result[2] == VEL_BASE + 9
        assert result[3] == 900

    def test_zero_shift_is_identity(self):
        seq = list(range(VEL_BASE, VEL_BASE + 16))
        result = _augment_velocity(seq, 0)
        assert result == seq

    def test_empty_sequence(self):
        assert _augment_velocity([], +1) == []


# ---------------------------------------------------------------------------
# VgmDatasetV7
# ---------------------------------------------------------------------------

class TestVgmDatasetV7:
    def _make_tokens(self, n_tokens: int) -> np.ndarray:
        return np.arange(n_tokens, dtype=np.int16)

    def test_basic_shape(self):
        seq_len = 64
        # chunk_size = seq_len + 1; need exactly 10 full chunks
        tokens = self._make_tokens((seq_len + 1) * 10)
        ds = VgmDatasetV7(tokens, seq_len=seq_len)
        assert len(ds) == 10

    def test_item_shape(self):
        seq_len = 32
        tokens = self._make_tokens(seq_len * 5 + 1)
        ds = VgmDatasetV7(tokens, seq_len=seq_len)
        item = ds[0]
        assert "input_ids" in item
        assert "labels" in item
        assert item["input_ids"].shape == (seq_len,)
        assert item["labels"].shape == (seq_len,)

    def test_labels_are_shifted(self):
        seq_len = 8
        # tokens: [0, 1, 2, ..., 8] (9 values = one chunk of 8+1)
        tokens = np.arange(9, dtype=np.int16)
        ds = VgmDatasetV7(tokens, seq_len=seq_len)
        item = ds[0]
        # input_ids[i] + 1 == labels[i] for a simple sequential token array
        diff = item["labels"].int() - item["input_ids"].int()
        assert (diff == 1).all()

    def test_pad_replaced_with_minus100(self):
        from genesis_music.tokenizer_v7 import PAD
        seq_len = 4
        # Craft tokens where PAD (0) appears
        tokens = np.array([1, PAD, 2, PAD, 3], dtype=np.int16)
        ds = VgmDatasetV7(tokens, seq_len=seq_len)
        item = ds[0]
        # labels = tokens[1:] with PAD→-100
        expected_labels = torch.tensor([tokens[1+i] for i in range(seq_len)], dtype=torch.int64)
        expected_labels[expected_labels == PAD] = -100
        assert torch.equal(item["labels"], expected_labels)

    def test_too_small_raises(self):
        with pytest.raises(ValueError):
            VgmDatasetV7(np.zeros(4, dtype=np.int16), seq_len=8)

    def test_all_items_unique_start(self):
        seq_len = 16
        n_chunks = 5
        tokens = np.arange(seq_len * n_chunks + 1, dtype=np.int16)
        ds = VgmDatasetV7(tokens, seq_len=seq_len)
        starts = [ds[i]["input_ids"][0].item() for i in range(len(ds))]
        assert len(set(starts)) == len(starts)


# ---------------------------------------------------------------------------
# Augmentation consistency: base + augmented sequences have same length
# ---------------------------------------------------------------------------

class TestAugmentationConsistency:
    def test_tempo_and_vel_preserve_length(self):
        from genesis_music.tokenizer_v7 import VEL_BASE, TEMPO_BASE
        bpm_idx = TEMPO_BINS.index(120)
        seq = [0, TEMPO_BASE + bpm_idx, VEL_BASE + 7, 100, 200]
        assert len(_augment_tempo(seq, 0.9)) == len(seq)
        assert len(_augment_velocity(seq, -1)) == len(seq)
