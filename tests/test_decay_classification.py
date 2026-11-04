"""Tests for the `DecayClassification` result container.

The properties (`sali_check`, `gali_check`, `sali_time`, `gali_time`) are
derived from the raw `classify_decay` dictionaries, so these tests feed them
handcrafted label/time arrays and need no integration.  Per-pair SALI results
use the (15, n_orbits) layout produced by ``detect_chaos(separate_sali=True)``.
"""

import numpy as np
import pytest

from ocd_gd._decay_classifier import classify_decay, is_chaotic, is_regular
from ocd_gd._types import DecayClassification


def indicator_dict(labels, t_onset=None, t_floor=None):
    """Minimal `classify_decay`-style dict holding only what the properties read."""
    result = {"label": np.asarray(labels)}
    if t_onset is not None:
        result["t_onset"] = np.asarray(t_onset, dtype=float)
    if t_floor is not None:
        result["t_floor"] = np.asarray(t_floor, dtype=float)
    return result


# =========================================================================
# *_check TESTS
# =========================================================================


@pytest.mark.parametrize("indicator", ["sali", "gali"])
def test_check_equals_is_chaotic_for_every_label_code(indicator):
    """Check arrays are 1.0 exactly where the label is even and > 0."""
    labels = np.array([0, 1, 2, 3, 4, 5, 6])
    dc = DecayClassification(sali=indicator_dict(labels), gali=indicator_dict(labels))
    check = getattr(dc, f"{indicator}_check")

    np.testing.assert_array_equal(check, is_chaotic(labels).astype(float))
    assert check.dtype == np.float64


def test_undetermined_orbits_are_currently_reported_as_not_chaotic():
    """Label 0 maps to check == 0, i.e. it is indistinguishable from regular.

    Documents the current rule; update this test if undetermined orbits are
    ever reported as NaN instead.
    """
    dc = DecayClassification(sali=indicator_dict([0]), gali=indicator_dict([0]))
    assert dc.sali_check[0] == 0.0
    assert dc.gali_check[0] == 0.0


def test_sali_check_reduces_fifteen_pairs_with_any():
    """With (15, N) labels an orbit is chaotic if any of its 15 pairs is."""
    labels = np.ones((15, 3), dtype=int)  # all pairs regular (plateau)
    labels[7, 1] = 2  # one chaotic pair for orbit 1
    labels[:, 2] = 4  # every pair chaotic for orbit 2
    dc = DecayClassification(
        sali=indicator_dict(labels), gali=indicator_dict([1, 1, 1])
    )

    np.testing.assert_array_equal(dc.sali_check, [0.0, 1.0, 1.0])
    assert dc.sali_check.shape == (3,)


def test_gali_check_is_per_orbit():
    """GALI results are already one label per orbit and pass straight through."""
    dc = DecayClassification(
        sali=indicator_dict([1, 1, 1]), gali=indicator_dict([1, 6, 3])
    )
    np.testing.assert_array_equal(dc.gali_check, [0.0, 1.0, 0.0])


# =========================================================================
# *_time TESTS
# =========================================================================


@pytest.mark.parametrize("indicator", ["sali", "gali"])
def test_time_falls_back_from_onset_to_floor_to_inf(indicator):
    """Detection time is t_onset, else t_floor, else inf; onset wins over floor."""
    t_onset = [5.0, np.nan, 7.0, np.nan]
    t_floor = [np.nan, 3.0, 9.0, np.nan]
    result = indicator_dict([4, 6, 4, 0], t_onset=t_onset, t_floor=t_floor)
    dc = DecayClassification(sali=result, gali=result)

    np.testing.assert_array_equal(
        getattr(dc, f"{indicator}_time"), [5.0, 3.0, 7.0, np.inf]
    )


@pytest.mark.parametrize("indicator", ["sali", "gali"])
def test_time_is_inf_when_the_time_keys_are_missing(indicator):
    """Dicts without t_onset / t_floor give an infinite detection time."""
    result = indicator_dict([1, 2])
    dc = DecayClassification(sali=result, gali=result)
    assert np.all(np.isinf(getattr(dc, f"{indicator}_time")))


def test_sali_time_takes_the_earliest_pair_on_the_15_by_n_layout():
    """With per-pair SALI results the detection time is the earliest over pairs."""
    t_onset = np.full((15, 2), np.nan)
    t_floor = np.full((15, 2), np.nan)
    t_onset[3, 0] = 4.0  # pair 3 detects chaos by onset at t = 4
    t_floor[9, 0] = 2.5  # pair 9 reaches the floor earlier, at t = 2.5
    labels = np.ones((15, 2), dtype=int)
    dc = DecayClassification(
        sali=indicator_dict(labels, t_onset=t_onset, t_floor=t_floor),
        gali=indicator_dict([1, 1]),
    )

    np.testing.assert_array_equal(dc.sali_time, [2.5, np.inf])
    assert dc.sali_time.shape == (2,)


# =========================================================================
# CONTAINER TESTS
# =========================================================================


def test_lyapunov_fields_default_to_none():
    """`lyapunov_check` and `lyapunov_time` are optional."""
    dc = DecayClassification(sali=indicator_dict([1]), gali=indicator_dict([1]))
    assert dc.lyapunov_check is None
    assert dc.lyapunov_time is None


# =========================================================================
# CONSISTENCY WITH classify_decay
# =========================================================================


@pytest.fixture
def regular_plateau_to_power_result():
    """classify_decay output for a regular plateau -> t^-3 curve (onset at t = 10)."""
    t = np.linspace(0.0, 30.0, 3000)
    tt = np.maximum(t, t[1])
    rng = np.random.default_rng(0)
    y = -0.3 - 3.0 * np.maximum(np.log10(tt / 10.0), 0.0)
    ts = (10 ** (y + rng.normal(0.0, 0.05, t.size)))[None]
    return classify_decay(ts, t)


def test_plateau_to_power_fixture_is_regular(regular_plateau_to_power_result):
    """Premise of the test below: this curve is classified as regular (label 5)."""
    assert is_regular(regular_plateau_to_power_result["label"])[0]


def test_regular_orbit_has_infinite_detection_time(regular_plateau_to_power_result):
    """A regular orbit never detects chaos, so its detection time must be inf.

    Regression test: t_onset used to be filled for regular plateau -> power-law
    fits, which made *_time finite for orbits that are not chaotic.
    """
    assert np.isnan(regular_plateau_to_power_result["t_onset"][0])
    dc = DecayClassification(
        sali=regular_plateau_to_power_result, gali=regular_plateau_to_power_result
    )
    assert np.isinf(dc.sali_time[0])
    assert np.isinf(dc.gali_time[0])
