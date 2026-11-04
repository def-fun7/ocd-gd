"""
Decay Classifier Module.

Classify SALI / GALI time series as regular or chaotic by Bayesian-style
model selection between five decay laws for y(t) = log10(indicator):

    idx  model                  form                                  k   class
    0    plateau                y = a                                 1   regular
    1    power law              y = a + b*log10(t)                    2   regular
    2    exponential            y = a + c*t                           2   chaotic
    3    plateau->exponential   y = a + c*max(t - tau, 0)             3   chaotic
    4    plateau->power law     y = a + b*max(log10(t/tau), 0)        3   regular

Models 3 and 4 are changepoint ("hinge") models: the indicator sits on a plateau
until an onset time tau, then decays.  They describe sticky / delayed behaviour,
and are included symmetrically for both classes so that extra flexibility
cannot favour one class.

Model weights:  w_i ∝ exp(-BIC_i / 2),
    BIC_i = n_eff * ln(RSS_i / n) + k_i * ln(n_eff)
Chaos score = w_2 + w_3.

Theory: Skokos (2001); Skokos, Bountis & Antonopoulos (2007) - chaotic orbits
give exponential decay of SALI/GALI_k, regular orbits give a constant or a
power law.
"""

from __future__ import annotations

__all__ = [
    "LABELS",
    "PRESETS",
    "get_preset",
    "classify_decay",
    "crossover_onset",
    "is_chaotic",
    "is_regular",
]

import numpy as np

LN10 = np.log(10.0)

LABELS = {
    0: "undetermined",
    1: "regular (plateau)",
    3: "regular (power law)",
    5: "regular (plateau -> power law)",
    2: "chaotic (exponential)",
    4: "chaotic (plateau -> exponential)",
    6: "chaotic (floor hit, censored)",
}
# label convention: odd = regular, even (and > 0) = chaotic, 0 = undetermined
_MODEL_TO_LABEL = np.array([1, 3, 2, 4, 5])  # models 0..4 -> label


def is_chaotic(label):
    """True where label is even and > 0 (2, 4, 6).  Safe for 0 = undetermined."""
    label = np.asarray(label)
    return (label > 0) & (label % 2 == 0)


def is_regular(label):
    """True where label is odd and > 0 (1, 3, 5)."""
    label = np.asarray(label)
    return (label > 0) & (label % 2 == 1)


_K = np.array([1, 2, 2, 3, 3])
_CHAOTIC = np.array([False, False, True, True, False])


def _line_fit(x, y):
    """Closed-form OLS of y on x.  Returns slope, intercept, residuals."""
    xm, ym = x.mean(), y.mean()
    sxx = np.sum((x - xm) ** 2)
    if sxx <= 0:
        return 0.0, ym, y - ym
    slope = np.sum((x - xm) * (y - ym)) / sxx
    icpt = ym - slope * xm
    return slope, icpt, y - (slope * x + icpt)


def _hinge_fit(x, y, min_side=3):
    """
    Best hinge model y = a + s*max(x - x_b, 0), scanning breakpoints x_b over the
    grid.  Returns (slope, x_b, residuals) of the minimum-RSS breakpoint.
    """
    best = None
    for j in range(min_side, len(x) - min_side):
        h = np.maximum(x - x[j], 0.0)
        s, _, res = _line_fit(h, y)
        rss = np.sum(res**2)
        if best is None or rss < best[0]:
            best = (rss, s, x[j], res)
    if best is None:  # grid too short
        s, _, res = _line_fit(x, y)
        return s, x[0], res
    return best[1], best[2], best[3]


def _lag1(res):
    """Lag-1 autocorrelation of residuals, clipped to [0, 0.99]."""
    r = res - res.mean()
    d = np.sum(r * r)
    if d <= 0:
        return 0.0
    return float(np.clip(np.sum(r[1:] * r[:-1]) / d, 0.0, 0.99))


def classify_decay(
    ts,
    time,
    *,
    floor=1e-12,
    t_min_frac=0.1,
    n_grid=64,
    min_points=8,
    sigma_floor=0.05,
    min_drop_decades=0.5,
    max_power_exponent=4.0,
    autocorr_correction=True,
    min_n_eff=6.0,
    t_orb=None,
):
    """Classify the decay behavior of chaos indicators.

    Parameters
    ----------
    ts : array_like
        Indicator time series of shape `(n_orbits, n_t)`.
    time : array_like
        Time values of shape `(n_t,)` or `(n_orbits, n_t)`.
    floor : float, default 1e-12
        Numerical precision floor below which values are treated as censored.
    t_min_frac : float, default 0.1
        Fraction of integration time discarded before fitting.
    n_grid : int, default 64
        Number of interpolation points on the log-spaced time grid.
    min_points : int, default 8
        Minimum valid data points required to attempt a fit.
    sigma_floor : float, default 0.05
        Residual variance floor for BIC calculation.
    min_drop_decades : float, default 0.5
        Minimum total decay in decades required to consider a decaying model physical.
    max_power_exponent : float, default 4.0
        Maximum plausible power-law decay exponent before assuming exponential decay.
    autocorr_correction : bool, default True
        Whether to apply AR(1) effective sample size correction to BIC.
    min_n_eff : float, default 6.0
        Minimum effective sample size allowed after AR(1) correction.
    t_orb : float or array_like, optional
        Orbital period(s) in the same units as `time`. Enables calculation of `rate_norm`.

    Returns
    -------
    dict
        Dictionary of arrays (length `n_orbits`):
        - ``score``: chaos probability `w_exp + w_exp_onset` (1.0 if censored, NaN if undetermined).
        - ``label``: integer code, see `LABELS`: odd = regular, even > 0 = chaotic, 0 = undetermined.
        - ``p_plateau``, ``p_power``, ``p_exp``, ``p_exp_onset``, ``p_power_onset``: model posterior weights.
        - ``delta_bic``: min BIC(regular) - min BIC(chaotic); >0 favours chaos.
        - ``rate``: decay rate of the indicator [1/time, natural log].
        - ``rate_norm``: rate * t_orb, analogue of AGAMA's lambda * T_orb.
        - ``rate_lb``: censored lower bound on rate when floor is reached.
        - ``rate_is_bound``: 1 if rate is the lower bound rather than a fit.
        - ``t_onset``: onset time tau from the hinge (plateau -> exponential) fit.
        - ``exponent``: power-law exponent p (indicator ~ t^-p).
        - ``t_floor``: first time the indicator reaches the floor.
        - ``n_eff``: effective sample size used in BIC.
    """
    ts = np.asarray(ts, dtype=float)
    time = np.asarray(time, dtype=float)
    if time.ndim == 1:
        time = np.broadcast_to(time, ts.shape)
    n_orb = ts.shape[0]

    names = (
        "score",
        "delta_bic",
        "rate",
        "rate_lb",
        "t_onset",
        "exponent",
        "t_floor",
        "n_eff",
        "p_plateau",
        "p_power",
        "p_exp",
        "p_exp_onset",
        "p_power_onset",
    )
    out = {k: np.full(n_orb, np.nan) for k in names}
    out["label"] = np.full(n_orb, 0, dtype=int)
    out["rate_is_bound"] = np.zeros(n_orb, dtype=int)
    log_floor = np.log10(floor)
    ln_inv_floor = -np.log(floor)

    for i in range(n_orb):
        t, s = time[i], ts[i]
        ok = np.isfinite(t) & np.isfinite(s) & (t > 0)
        t, s = t[ok], np.maximum(s[ok], 1e-300)  # zeros count as floor hits
        if t.size < min_points:
            continue

        # ---- censoring by the numerical floor --------------------------------
        hit = np.flatnonzero(s <= floor)
        end = hit[0] if hit.size else t.size
        if hit.size and hit[0] == 0:
            continue
        if hit.size:
            t_f = t[hit[0]]
            out["t_floor"][i] = t_f
            if t_f > t[0] and s[0] > floor:
                out["rate_lb"][i] = np.log(s[0] / floor) / (t_f - t[0])
            span = np.log10(t_f / t[0])
            p_req = -log_floor / span if span > 0 else np.inf  # exponent needed
            if p_req > max_power_exponent:
                out["score"][i] = 1.0
                out["p_exp"][i] = 1.0
                for k in ("p_plateau", "p_power", "p_exp_onset", "p_power_onset"):
                    out[k][i] = 0.0
                out["label"][i] = 6
                out["rate"][i] = out["rate_lb"][i]
                out["rate_is_bound"][i] = 1
                continue
        t, y = t[:end], np.log10(s[:end])
        if t.size < min_points:
            continue

        # ---- window + log-spaced resampling ----------------------------------
        t_hi = t[-1]
        t_lo = max(t_min_frac * t_hi, t[0])
        if np.sum(t >= t_lo) < min_points or t_hi <= t_lo:
            continue
        tg = np.geomspace(t_lo, t_hi, n_grid)
        yg = np.interp(tg, t, y)
        n = tg.size
        xl = np.log10(tg)
        span_lin, span_log = tg[-1] - tg[0], xl[-1] - xl[0]

        # ---- five fits --------------------------------------------------------
        res = [None] * 5
        slope = np.zeros(5)
        onset = np.full(5, np.nan)
        res[0] = yg - yg.mean()
        slope[1], _, res[1] = _line_fit(xl, yg)  # b  [dec / dec]
        slope[2], _, res[2] = _line_fit(tg, yg)  # c  [dec / time]
        slope[3], tau, res[3] = _hinge_fit(tg, yg)
        onset[3] = tau
        slope[4], xtau, res[4] = _hinge_fit(xl, yg)
        onset[4] = 10**xtau
        rss = np.array([np.sum(r**2) for r in res])

        # ---- effective sample size (AR(1) correction) -------------------------
        if autocorr_correction:
            r1 = _lag1(res[int(np.argmin(rss))])
            n_eff = float(np.clip(n * (1 - r1) / (1 + r1), min_n_eff, n))
        else:
            n_eff = float(n)

        rss_f = np.maximum(rss, n * sigma_floor**2)
        bic = n_eff * np.log(rss_f / n) + _K * np.log(n_eff)

        # ---- physical admissibility: decaying models must really decay --------
        drop = np.array(
            [
                np.inf,
                -slope[1] * span_log,
                -slope[2] * span_lin,
                -slope[3] * (tg[-1] - onset[3]),
                -slope[4] * (xl[-1] - np.log10(onset[4])),
            ]
        )
        bic[(np.arange(5) > 0) & ~(drop >= min_drop_decades)] = np.inf

        w = np.exp(-0.5 * (bic - bic.min()))
        w /= w.sum()

        # ---- outputs ----------------------------------------------------------
        reg_b = bic[~_CHAOTIC].min()
        cha_b = bic[_CHAOTIC].min()
        (
            out["p_plateau"][i],
            out["p_power"][i],
            out["p_exp"][i],
            out["p_exp_onset"][i],
            out["p_power_onset"][i],
        ) = w
        out["score"][i] = w[2] + w[3]
        out["label"][i] = _MODEL_TO_LABEL[int(np.argmax(w))]
        if np.isfinite(reg_b) and np.isfinite(cha_b):
            out["delta_bic"][i] = reg_b - cha_b
        best_c = 3 if bic[3] < bic[2] else 2
        out["rate"][i] = -slope[best_c] * LN10
        out["t_onset"][i] = (
            onset[3] if out["label"][i] == 4 else np.nan
        )  # chaotic hinge only
        out["exponent"][i] = -slope[1]
        out["n_eff"][i] = n_eff

    # ---- normalised rate, analogue of AGAMA's lambda * T_orb ----------------
    rn = np.full(n_orb, np.nan)
    if t_orb is not None:
        t_orb = np.broadcast_to(np.asarray(t_orb, dtype=float), (n_orb,))
        rn[is_chaotic(out["label"])] = (out["rate"] * t_orb)[is_chaotic(out["label"])]
        rn[is_regular(out["label"])] = 0.0
    out["rate_norm"] = rn
    return out


# --------------------------------------------------------------------------- #
# indicator presets
# --------------------------------------------------------------------------- #
# Use as  classify_decay(ts, time, **PRESETS["gali6_3dof"]).
#   gali6_3dof : floor / cap chosen on one AGAMA dataset (962 orbits, 30 orbital periods);
#                regular GALI_6 there decays as t^-p with p ~ 5.3-6.8 (5th-95th pct).
#                The floor was tuned in-sample - re-check it on new potentials.
#   sali       : defaults tuned on Henon-Heiles (2 DOF); NOT yet validated on 3-DOF data.
PRESETS = {
    "gali6_3dof": dict(floor=1e-30, max_power_exponent=8.0),
    "sali": dict(floor=1e-12, max_power_exponent=4.0),
}


def get_preset(name):
    """Return a copy of the classify_decay keyword preset `name` ({} for None)."""
    if name is None:
        return {}
    try:
        return dict(PRESETS[name])
    except KeyError:
        raise ValueError(
            f"Unknown preset {name!r}; valid presets are {sorted(PRESETS)} or None."
        ) from None


# --------------------------------------------------------------------------- #
# optional helper: onset of exponential decay on top of a power law
# --------------------------------------------------------------------------- #


def crossover_onset(ts, time, floor_fit=1e-30, n_grid=80, t_min_frac=0.02, min_side=4):
    """
    Onset time of exponential decay for indicators that already decay as a power law
    (GALI_k with k > N).  Fits, per orbit,

        y = a + b*log10(t) + c*max(t - tau, 0),     c < 0,     y = log10(indicator)

    scanning tau over a log-spaced grid (closed-form OLS for a, b, c at each tau).

    Returns (tau, rate, slope, edge), each of length n_orbits:
      tau   onset time, in the units of `time` (pass time / T_orb for orbital periods)
      rate  exponential decay rate -c*ln10 [1/time]
      slope power-law slope b (regular-phase exponent is -b)
      edge  True if tau sits within ~2 grid points of the window edge, i.e. the
            fit found no clear bend; exclude these from onset statistics.
    Use only on orbits already classified as chaotic.
    """
    ts = np.asarray(ts, dtype=float)
    time = np.asarray(time, dtype=float)
    if time.ndim == 1:
        time = np.broadcast_to(time, ts.shape)
    n = ts.shape[0]
    tau = np.full(n, np.nan)
    rate = np.full(n, np.nan)
    b_out = np.full(n, np.nan)
    edge = np.zeros(n, dtype=bool)
    for i in range(n):
        t, s = time[i], np.maximum(ts[i], 1e-300)
        ok = np.isfinite(t) & (t > 0) & np.isfinite(s)
        t, s = t[ok], s[ok]
        hit = np.flatnonzero(s <= floor_fit)
        if hit.size:
            t, s = t[: hit[0]], s[: hit[0]]
        if t.size < 12:
            continue
        lo, hi = max(t_min_frac * t[-1], t[0]), t[-1]
        if hi <= lo:
            continue
        tg = np.geomspace(lo, hi, n_grid)
        yg = np.interp(tg, t, np.log10(s))
        xl = np.log10(tg)
        best = None
        for j in range(min_side, n_grid - min_side):
            X = np.column_stack([np.ones(n_grid), xl, np.maximum(tg - tg[j], 0.0)])
            beta, *_ = np.linalg.lstsq(X, yg, rcond=None)
            rss = np.sum((yg - X @ beta) ** 2)
            if best is None or rss < best[0]:
                best = (rss, tg[j], beta, j)
        _, tau[i], beta, j_best = best
        b_out[i], rate[i] = beta[1], -beta[2] * LN10
        edge[i] = (j_best <= min_side + 1) or (j_best >= n_grid - min_side - 2)
    return tau, rate, b_out, edge
