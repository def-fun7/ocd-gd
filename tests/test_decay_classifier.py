"""Tests for the decay-law classifier (`classify_decay`) and its helpers.

All inputs are synthetic indicator curves with a known decay law, so these
tests need no AGAMA integration and run in well under a second.  Time is
expressed in orbital periods, starting at t = 0 like the integrator output.
"""

import numpy as np
import pytest

from ocd_gd._decay_classifier import (
    LABELS,
    PRESETS,
    classify_decay,
    crossover_onset,
    is_chaotic,
    is_regular,
)

N_T = 3000
T_END = 30.0
N_GRID = 64  # default log-grid size used inside classify_decay


@pytest.fixture
def time_axis():
    """3000 samples over 30 orbital periods, including the t = 0 sample."""
    return np.linspace(0.0, T_END, N_T)


@pytest.fixture
def rng():
    """Seeded generator so every noisy curve is reproducible."""
    return np.random.default_rng(12345)


# =========================================================================
# SYNTHETIC CURVE HELPERS
# =========================================================================

# kind -> (expected label, is chaotic)
DECAY_LAWS = {
    "plateau": (1, False),
    "power_law": (3, False),
    "plateau_to_power": (5, False),
    "exponential": (2, True),
    "plateau_to_exponential": (4, True),
}


def log10_curve(kind, t):
    """Noise-free log10(indicator) for each decay law (starts near 10**-0.3)."""
    dt = t[1]
    tt = np.maximum(t, dt)  # avoid log(0) at t = 0
    if kind == "plateau":
        return np.full_like(t, -0.3)
    if kind == "power_law":  # indicator ~ t^-2
        return -0.3 - 2.0 * np.log10(tt / dt)
    if kind == "plateau_to_power":  # flat until t = 5, then ~ t^-4
        return -0.3 - 4.0 * np.maximum(np.log10(tt / 5.0), 0.0)
    if kind == "exponential":  # 0.08 decades per period
        return -0.3 - 0.08 * t
    if kind == "plateau_to_exponential":  # flat until t = 15, then 0.5 dec/period
        return -0.3 - 0.5 * np.maximum(t - 15.0, 0.0)
    if kind == "exponential_to_floor":  # 2 dec/period: reaches 1e-12 near t = 5.9
        return -0.3 - 2.0 * t
    if kind == "steep_power_law":  # indicator ~ t^-6 (regular GALI_6 in 3 DOF)
        return -0.3 - 6.0 * np.log10(tt / dt)
    raise ValueError(kind)


def noisy(kind, t, rng, sigma=0.03):
    """Indicator values (not log) with Gaussian noise in log10 space."""
    y = log10_curve(kind, t)
    return 10 ** (y + rng.normal(0.0, sigma, y.size))


# =========================================================================
# LABEL CONVENTION TESTS
# =========================================================================


def test_label_codes_follow_parity_convention():
    """Odd codes must be regular, even codes > 0 chaotic, 0 undetermined."""
    assert set(LABELS) == {0, 1, 2, 3, 4, 5, 6}
    for code, text in LABELS.items():
        if code == 0:
            assert text == "undetermined"
        elif code % 2 == 1:
            assert text.startswith("regular"), f"code {code}: {text}"
        else:
            assert text.startswith("chaotic"), f"code {code}: {text}"


@pytest.mark.parametrize("code", [0, 1, 2, 3, 4, 5, 6])
def test_is_chaotic_and_is_regular_partition_the_codes(code):
    """Every decided code is exactly one of regular/chaotic; 0 is neither."""
    chaotic = bool(is_chaotic(code))
    regular = bool(is_regular(code))
    if code == 0:
        assert not chaotic and not regular
    else:
        assert chaotic != regular
        assert chaotic == (code % 2 == 0)


def test_is_chaotic_is_vectorised():
    """The helpers accept arrays and return boolean arrays of the same shape."""
    labels = np.array([[0, 1, 2], [3, 4, 6]])
    np.testing.assert_array_equal(
        is_chaotic(labels), [[False, False, True], [False, True, True]]
    )
    np.testing.assert_array_equal(
        is_regular(labels), [[False, True, False], [True, False, False]]
    )


# =========================================================================
# DECAY-LAW RECOVERY TESTS
# =========================================================================


@pytest.mark.parametrize("kind", list(DECAY_LAWS))
def test_recovers_label_and_score_for_each_decay_law(kind, time_axis, rng):
    """Each synthetic law gets its own label and a score on the right side."""
    expected_label, chaotic = DECAY_LAWS[kind]
    result = classify_decay(noisy(kind, time_axis, rng)[None], time_axis)

    assert result["label"][0] == expected_label
    if chaotic:
        assert result["score"][0] > 0.9
    else:
        assert result["score"][0] < 0.1


@pytest.mark.parametrize("kind", list(DECAY_LAWS))
def test_class_is_stable_under_realistic_noise(kind, time_axis, rng):
    """With 0.1 dex noise no realisation may cross the regular/chaotic line."""
    _, chaotic = DECAY_LAWS[kind]
    ts = np.array([noisy(kind, time_axis, rng, sigma=0.1) for _ in range(30)])
    labels = classify_decay(ts, time_axis)["label"]

    if chaotic:
        assert np.all(is_chaotic(labels))
    else:
        assert not np.any(is_chaotic(labels))


def test_exponential_rate_is_recovered(time_axis, rng):
    """`rate` is the natural-log decay rate per unit time of the indicator."""
    ts = noisy("exponential", time_axis, rng)[None]
    rate = classify_decay(ts, time_axis)["rate"][0]
    np.testing.assert_allclose(rate, 0.08 * np.log(10), rtol=0.05)


def test_onset_time_and_rate_of_hinge_exponential_are_recovered(time_axis, rng):
    """The plateau -> exponential fit returns tau ~ 15 and the true rate."""
    ts = noisy("plateau_to_exponential", time_axis, rng)[None]
    result = classify_decay(ts, time_axis)

    assert result["label"][0] == 4
    assert abs(result["t_onset"][0] - 15.0) < 1.5
    np.testing.assert_allclose(result["rate"][0], 0.5 * np.log(10), rtol=0.05)


def test_power_law_exponent_is_recovered(time_axis, rng):
    """For an indicator ~ t^-2 the fitted exponent is close to 2."""
    ts = noisy("power_law", time_axis, rng)[None]
    exponent = classify_decay(ts, time_axis)["exponent"][0]
    assert abs(exponent - 2.0) < 0.1


# =========================================================================
# FLOOR CENSORING AND PRESET TESTS
# =========================================================================


def test_floor_hit_is_censored_chaotic(time_axis, rng):
    """A curve that reaches the numerical floor quickly is label 6 with a bound."""
    ts = noisy("exponential_to_floor", time_axis, rng)[None]
    result = classify_decay(ts, time_axis)

    assert result["label"][0] == 6
    assert result["score"][0] == 1.0
    assert result["rate_is_bound"][0] == 1
    assert result["rate"][0] == result["rate_lb"][0]
    # floor 1e-12 is crossed at t ~ (12 - 0.3) / 2 = 5.85; the bound ~ 2 ln10
    assert abs(result["t_floor"][0] - 5.85) < 0.3
    np.testing.assert_allclose(result["rate_lb"][0], 2.0 * np.log(10), rtol=0.1)


@pytest.mark.parametrize(
    ("preset", "expected_label"),
    [
        (None, 6),  # classify_decay defaults: floor 1e-12, exponent cap 4
        ("sali", 6),  # same numbers as the defaults
        ("gali6_3dof", 3),  # floor 1e-30, exponent cap 8: t^-6 is a regular power law
    ],
)
def test_steep_power_law_depends_on_the_preset(preset, expected_label, time_axis, rng):
    """A t^-6 curve (regular GALI_6) is only 'regular' under the GALI preset.

    Documents the sharp edge: feeding a GALI_6-like series to the SALI
    settings classifies it as a floor-hit (chaotic) orbit.
    """
    ts = noisy("steep_power_law", time_axis, rng)[None]
    kwargs = PRESETS[preset] if preset else {}
    label = classify_decay(ts, time_axis, **kwargs)["label"][0]
    assert label == expected_label


def test_presets_are_the_documented_set_and_valid_kwargs(time_axis, rng):
    """Both presets must be accepted by classify_decay as keyword arguments."""
    assert set(PRESETS) == {"sali", "gali6_3dof"}
    ts = noisy("plateau", time_axis, rng)[None]
    for name, kwargs in PRESETS.items():
        result = classify_decay(ts, time_axis, **kwargs)
        assert result["label"][0] == 1, name


# =========================================================================
# UNDETERMINED AND DEGENERATE INPUT TESTS
# =========================================================================


def test_too_short_series_is_undetermined():
    """Fewer valid points than `min_points` -> label 0, NaN score."""
    t = np.arange(1.0, 6.0)
    result = classify_decay(np.ones((1, 5)), t)

    assert result["label"][0] == 0
    assert np.isnan(result["score"][0])
    assert not is_chaotic(result["label"])[0]
    assert not is_regular(result["label"])[0]


def test_all_nan_series_is_undetermined(time_axis):
    """A series with no finite samples cannot be classified."""
    result = classify_decay(np.full((1, N_T), np.nan), time_axis)
    assert result["label"][0] == 0
    assert np.isnan(result["score"][0])


@pytest.mark.parametrize("kind", ["plateau", "exponential"])
def test_non_finite_tail_is_ignored(kind, time_axis, rng):
    """NaN samples (e.g. after an escape) are dropped; the valid part decides."""
    ts = noisy(kind, time_axis, rng)
    truncated = classify_decay(ts[None, :1500], time_axis[:1500])
    with_nan_tail = ts.copy()
    with_nan_tail[1500:] = np.nan
    result = classify_decay(with_nan_tail[None], time_axis)

    assert result["label"][0] == truncated["label"][0]


# =========================================================================
# OUTPUT CONTRACT TESTS
# =========================================================================


@pytest.fixture
def mixed_batch(time_axis, rng):
    """One orbit per decay law plus a floor-hit orbit (6 orbits)."""
    kinds = [*DECAY_LAWS, "exponential_to_floor"]
    return np.array([noisy(k, time_axis, rng) for k in kinds])


def test_output_keys_and_shapes(mixed_batch, time_axis):
    """Every documented key is present with one entry per orbit."""
    result = classify_decay(mixed_batch, time_axis)
    expected_keys = {
        "score", "label", "p_plateau", "p_power", "p_exp", "p_exp_onset",
        "p_power_onset", "delta_bic", "rate", "rate_norm", "rate_lb",
        "rate_is_bound", "t_onset", "exponent", "t_floor", "n_eff",
    }  # fmt: skip
    assert set(result) == expected_keys
    for key, value in result.items():
        assert value.shape == (len(mixed_batch),), key
    assert np.issubdtype(result["label"].dtype, np.integer)
    assert np.issubdtype(result["rate_is_bound"].dtype, np.integer)


def test_posterior_weights_sum_to_one_and_define_the_score(mixed_batch, time_axis):
    """The five model weights are a probability vector; score = chaotic weights."""
    result = classify_decay(mixed_batch, time_axis)
    weights = np.column_stack(
        [result[k] for k in ("p_plateau", "p_power", "p_exp", "p_exp_onset", "p_power_onset")]
    )  # fmt: skip

    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-9)
    np.testing.assert_allclose(
        result["score"], result["p_exp"] + result["p_exp_onset"], atol=1e-12
    )
    assert np.all((result["score"] >= 0.0) & (result["score"] <= 1.0))


def test_delta_bic_sign_matches_the_class(mixed_batch, time_axis):
    """delta_bic > 0 favours chaos, < 0 favours regular, where it is defined.

    It is NaN for floor-hit orbits (no fit) and for curves with no admissible
    decaying model, e.g. a plateau (no decay of at least `min_drop_decades`).
    """
    result = classify_decay(mixed_batch, time_axis)
    delta = result["delta_bic"]
    fitted = result["label"] != 6
    chaotic = fitted & is_chaotic(result["label"])
    regular = fitted & is_regular(result["label"]) & np.isfinite(delta)

    assert np.all(np.isfinite(delta[chaotic]))
    assert np.all(delta[chaotic] > 0)
    assert np.all(delta[regular] < 0)
    assert np.isnan(delta[~fitted]).all()


def test_effective_sample_size_is_bounded(mixed_batch, time_axis):
    """n_eff stays between `min_n_eff` and the log-grid size for fitted orbits."""
    result = classify_decay(mixed_batch, time_axis)
    n_eff = result["n_eff"][result["label"] != 6]
    assert np.all((n_eff >= 6.0) & (n_eff <= N_GRID))


# =========================================================================
# INVARIANCE TESTS
# =========================================================================


def test_one_dimensional_and_per_orbit_time_axes_agree(mixed_batch, time_axis):
    """A shared (n_t,) time axis equals the same axis repeated per orbit."""
    shared = classify_decay(mixed_batch, time_axis)
    per_orbit = classify_decay(mixed_batch, np.tile(time_axis, (len(mixed_batch), 1)))

    np.testing.assert_array_equal(shared["label"], per_orbit["label"])
    np.testing.assert_allclose(shared["score"], per_orbit["score"], atol=1e-12)


def test_batch_result_matches_single_orbit_results(mixed_batch, time_axis):
    """Orbits are classified independently of what else is in the batch."""
    batch = classify_decay(mixed_batch, time_axis)
    for i, row in enumerate(mixed_batch):
        single = classify_decay(row[None], time_axis)
        assert single["label"][0] == batch["label"][i]
        np.testing.assert_allclose(single["score"][0], batch["score"][i], atol=1e-12)


def test_row_order_does_not_matter(mixed_batch, time_axis):
    """Reversing the batch reverses the labels and nothing else."""
    forward = classify_decay(mixed_batch, time_axis)["label"]
    backward = classify_decay(mixed_batch[::-1], time_axis)["label"]
    np.testing.assert_array_equal(forward, backward[::-1])


def test_nan_row_does_not_contaminate_neighbours(mixed_batch, time_axis):
    """An unclassifiable orbit leaves the other orbits' results unchanged."""
    reference = classify_decay(mixed_batch, time_axis)
    corrupted = mixed_batch.copy()
    corrupted[2] = np.nan
    result = classify_decay(corrupted, time_axis)

    assert result["label"][2] == 0
    keep = [0, 1, 3, 4, 5]
    np.testing.assert_array_equal(result["label"][keep], reference["label"][keep])


def test_time_unit_rescaling_keeps_labels_and_rescales_rates(mixed_batch, time_axis):
    """Changing the time unit (e.g. periods -> Gyr) must not change the labels."""
    factor = 0.978
    base = classify_decay(mixed_batch, time_axis)
    scaled = classify_decay(mixed_batch, time_axis * factor)

    np.testing.assert_array_equal(base["label"], scaled["label"])
    chaotic = is_chaotic(base["label"]) & (base["label"] != 6)
    np.testing.assert_allclose(
        scaled["rate"][chaotic] * factor, base["rate"][chaotic], rtol=1e-6
    )


# =========================================================================
# NORMALISED RATE (rate_norm) TESTS
# =========================================================================


def test_rate_norm_scales_with_orbital_period(time_axis, rng):
    """rate_norm = rate * t_orb for chaotic orbits, 0 regular, NaN undetermined."""
    chaotic = noisy("exponential", time_axis, rng)
    regular = noisy("plateau", time_axis, rng)
    undetermined = np.full(N_T, np.nan)
    ts = np.array([chaotic, regular, undetermined])
    t_orb = np.array([2.0, 3.0, 4.0])

    result = classify_decay(ts, time_axis, t_orb=t_orb)

    np.testing.assert_allclose(result["rate_norm"][0], result["rate"][0] * 2.0)
    assert result["rate_norm"][1] == 0.0
    assert np.isnan(result["rate_norm"][2])


def test_rate_norm_accepts_scalar_period_and_is_nan_without_it(time_axis, rng):
    """A scalar t_orb is broadcast; omitting it leaves rate_norm all NaN."""
    ts = noisy("exponential", time_axis, rng)[None]

    with_period = classify_decay(ts, time_axis, t_orb=2.5)
    without_period = classify_decay(ts, time_axis)

    np.testing.assert_allclose(
        with_period["rate_norm"][0], with_period["rate"][0] * 2.5
    )
    assert np.isnan(without_period["rate_norm"]).all()


# =========================================================================
# crossover_onset TESTS
# =========================================================================


def test_crossover_onset_finds_exponential_onset_on_a_power_law(time_axis, rng):
    """Power law followed by exponential decay: tau ~ 15 and the true rate."""
    dt = time_axis[1]
    tt = np.maximum(time_axis, dt)
    y = -0.3 - 1.5 * np.log10(tt / dt) - 0.25 * np.maximum(time_axis - 15.0, 0.0)
    ts = (10 ** (y + rng.normal(0.0, 0.03, y.size)))[None]

    tau, rate, slope, edge = crossover_onset(ts, time_axis)

    assert abs(tau[0] - 15.0) < 1.5
    np.testing.assert_allclose(rate[0], 0.25 * np.log(10), rtol=0.1)
    assert slope[0] < 0  # power-law phase decays
    assert not edge[0]


def test_crossover_onset_rate_is_near_zero_for_a_pure_power_law(time_axis, rng):
    """No bend in the curve -> the fitted exponential rate is ~0.

    The returned onset time is meaningless here, and `edge` is not a reliable
    marker of that (noise can put tau anywhere), so the helper must only be
    applied to orbits already classified as chaotic.
    """
    ts = noisy("power_law", time_axis, rng)[None]
    _, rate, slope, _ = crossover_onset(ts, time_axis)

    assert abs(rate[0]) < 0.05  # vs 0.58 for a genuine exponential phase
    assert abs(slope[0] + 2.0) < 0.1


@pytest.mark.parametrize("level", [0.0, 1e-20])
def test_series_at_the_floor_from_the_start_is_undetermined(level, time_axis):
    """No decay is observable if the indicator starts at the floor."""
    result = classify_decay(np.full((1, N_T), level), time_axis)
    assert result["label"][0] == 0
    assert np.isnan(result["score"][0])
