"""Tests for the v7 hardware-cluster classifier."""

import numpy as np
import pytest

from genesis_music.clusters_v7 import classify_song, N_FEATURES


def _feat(**kwargs) -> np.ndarray:
    """Build a feature vector from named kwargs, defaulting all to 0."""
    f = np.zeros(N_FEATURES, dtype=np.float32)
    idx = {
        "pan_rate": 0, "max_fb": 1, "ic_rate": 2, "dac_frac": 3,
        "psg_frac": 4, "ssg_eg": 5, "lfo_frac": 6, "ch3_frac": 7,
    }
    for k, v in kwargs.items():
        f[idx[k]] = v
    return f


class TestCluster1:
    def test_all_zeros_is_cluster1(self):
        assert classify_song(np.zeros(N_FEATURES, dtype=np.float32)) == 1

    def test_low_psg_is_cluster1(self):
        assert classify_song(_feat(psg_frac=0.30)) == 1

    def test_low_dac_is_cluster1(self):
        assert classify_song(_feat(dac_frac=0.35)) == 1


class TestCluster2:
    def test_heavy_psg(self):
        # psg_frac just above threshold
        assert classify_song(_feat(psg_frac=0.36)) == 2

    def test_high_psg_no_other_triggers(self):
        assert classify_song(_feat(psg_frac=0.80)) == 2

    def test_psg_at_threshold_boundary(self):
        # exactly at threshold (0.35) → does NOT trigger cluster 2
        assert classify_song(_feat(psg_frac=0.35)) == 1


class TestCluster3:
    def test_dac_dominant(self):
        assert classify_song(_feat(dac_frac=0.41)) == 3

    def test_dac_fraction_one(self):
        assert classify_song(_feat(dac_frac=1.0)) == 3

    def test_dac_at_threshold(self):
        # just below threshold (0.40) → does NOT trigger cluster 3
        assert classify_song(_feat(dac_frac=0.399)) == 1


class TestCluster4:
    def test_ssg_eg_usage(self):
        assert classify_song(_feat(ssg_eg=1.0)) == 4

    def test_high_instrument_change_rate(self):
        assert classify_song(_feat(ic_rate=0.51)) == 4

    def test_ic_at_threshold(self):
        # exactly 0.5 → does NOT trigger cluster 4
        assert classify_song(_feat(ic_rate=0.5)) == 1


class TestCluster5:
    def test_pan_rate_above_one(self):
        assert classify_song(_feat(pan_rate=1.01)) == 5

    def test_lfo_active(self):
        assert classify_song(_feat(lfo_frac=0.11)) == 5

    def test_ch3_special(self):
        assert classify_song(_feat(ch3_frac=0.06)) == 5

    def test_pan_at_threshold(self):
        # exactly 1.0 → does NOT trigger cluster 5
        # (combined with nothing else, goes to cluster 1)
        assert classify_song(_feat(pan_rate=1.0)) == 1


class TestPriority:
    def test_cluster5_beats_cluster4(self):
        # Both cluster 5 and 4 conditions are met; cluster 5 should win
        assert classify_song(_feat(pan_rate=2.0, ssg_eg=1.0)) == 5

    def test_cluster5_beats_cluster3(self):
        assert classify_song(_feat(lfo_frac=0.5, dac_frac=0.9)) == 5

    def test_cluster4_beats_cluster3(self):
        assert classify_song(_feat(ssg_eg=1.0, dac_frac=0.9)) == 4

    def test_cluster4_beats_cluster2(self):
        assert classify_song(_feat(ic_rate=1.0, psg_frac=0.8)) == 4

    def test_cluster3_beats_cluster2(self):
        assert classify_song(_feat(dac_frac=0.9, psg_frac=0.8)) == 3
