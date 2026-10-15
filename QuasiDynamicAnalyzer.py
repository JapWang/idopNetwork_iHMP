# -*- coding: utf-8 -*-
"""Quasi-dynamic (pseudo-time) analysis.

Column transforms, quasi-dynamic series construction, power-law fitting
``y = a * x^b``, power-function sampling, SWT stationary-wavelet denoising for
dynamic (time-series) data, and power-law fit visualisation.

Class names, function names, parameter defaults and all computations are
unchanged from the original implementation.
"""
from __future__ import annotations

import io
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# Data loading
def load_csv(file):
    return pd.read_csv(file, index_col=0)


def _numeric_frame_for_transform(data: pd.DataFrame) -> pd.DataFrame:
    """Cast columns to float; raise a clear error if any column is non-numeric."""
    if data.empty:
        return data.copy()
    try:
        return data.apply(pd.to_numeric, errors="raise").astype(np.float64)
    except ValueError as e:
        raise ValueError(
            "Data transformation requires all data columns to be numeric "
            "(object/text columns cannot be scaled or log1p-transformed). "
            "Check the CSV: only the row-index column should precede the data."
        ) from e


# Data transformation
def preprocess(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """Apply the column-wise transform; non-numeric cells become NaN."""
    df = df.apply(pd.to_numeric, errors="coerce")
    transforms = {
        "None": lambda x: x,
        "Log10_1p": lambda x: np.log10(1 + x),
        "Minmax_0_1": lambda x: (x - x.min()) / (x.max() - x.min()).replace(0, 1),
        "Z_min_add1": lambda x: x - x.min() + 1,
    }
    if method not in transforms:
        raise ValueError(f"unsupported transformation: {method}")
    return transforms[method](df)


def data_transformation(data: pd.DataFrame, scaler_type: str) -> pd.DataFrame:
    """Entry point for the preprocessing kernel."""
    return preprocess(data, scaler_type)


# Static DataFrame -> quasi-dynamic DataFrame:
#   y_j(s_i) -> y_j(tau_i)
#   T_i = sum_j y_j(s_i),  tau_i = sigma(T_i)  with tau_1 <= ... <= tau_n
def get_quasi_dynamic_df(
    data: pd.DataFrame,
    log_index: bool = False,
) -> pd.DataFrame:
    """Build quasi-dynamic data, optionally using log of the row-sum index."""
    row_sum = data.sum(axis=1)
    # Sort by positional index: .loc would expand duplicated row labels.
    sort_pos = np.argsort(row_sum.values, kind="stable")
    quasi_dynamic_df = data.iloc[sort_pos].copy()
    quasi_index = row_sum.values[sort_pos].astype(float)
    if log_index:
        valid = np.isfinite(quasi_index) & (quasi_index > 0)
        quasi_dynamic_df = quasi_dynamic_df.iloc[valid].copy()
        quasi_index = np.log(quasi_index[valid])
    quasi_dynamic_df.index = pd.Index(quasi_index)
    return quasi_dynamic_df


# y = a * x^{b}
def power_equation(
    x: np.ndarray,
    a: float,
    b: float,
) -> np.ndarray:
    return a * np.power(x, b)


def fit_power_loglinear(
    x: np.ndarray,
    y: np.ndarray,
    *,
    clip_a: Tuple[float, float] = (-np.inf, np.inf),
    clip_b: Tuple[float, float] = (-np.inf, np.inf),
) -> Tuple[float, float]:
    """Fit ``y = a * x^b`` by log-log ordinary least squares.

    Least squares in log-log space assumes multiplicative (log-normal) noise,
    which is appropriate for biological data spanning orders of magnitude, and
    it has a closed-form solution (no optimiser initialisation).

    Pairs with ``x <= 0``, ``y <= 0`` or non-finite values are dropped. If
    fewer than two pairs remain, or ``log x`` has zero variance, the fallback
    ``(a, b) = (1.0, 0.5)`` is returned.

    Raises
    ------
    ValueError
        If ``x`` and ``y`` differ in length.
    """
    x_arr = np.asarray(x, dtype=np.float64).ravel()
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    if x_arr.shape != y_arr.shape:
        raise ValueError(
            f"x and y must have the same length: "
            f"x.shape={x_arr.shape}, y.shape={y_arr.shape}"
        )

    mask = (x_arr > 0) & (y_arr > 0) & np.isfinite(x_arr) & np.isfinite(y_arr)
    if int(mask.sum()) < 2:
        return 1.0, 0.5

    log_x = np.log(x_arr[mask])
    log_y = np.log(y_arr[mask])
    try:
        result = stats.linregress(log_x, log_y)
        slope = float(result.slope)
        intercept = float(result.intercept)
    except Exception:
        return 1.0, 0.5
    if not (np.isfinite(slope) and np.isfinite(intercept)):
        return 1.0, 0.5

    # Clip the intercept so exp() cannot overflow: exp(690) ~ 1.4e299.
    safe_intercept = float(np.clip(intercept, -690.0, 690.0))
    a = float(np.exp(safe_intercept))
    b = slope
    a = float(np.clip(a, clip_a[0], clip_a[1]))
    b = float(np.clip(b, clip_b[0], clip_b[1]))
    return a, b


def power_fitting(df_qd: pd.DataFrame, n_samples: int = 30, ignore_zero: bool = True):
    """Fit power curves in standardised log-time and sample a uniform grid.

    Non-positive / non-finite observations are ignored per feature. Features
    with fewer than two usable observations keep NaN parameters so that
    clustering can exclude them.

    Parameters
    ----------
    df_qd:
        Quasi-dynamic table; index is quasi-time (positive), columns are features.
    n_samples:
        Number of samples on the power curve (>= 2).
    ignore_zero:
        Whether to ignore zeros (and negative values) when fitting. ``True`` is
        required for zero-inflated data such as microbiome counts, otherwise
        zeros flatten the curves. With ``False`` zeros enter the log statistics
        and ``log(0) = -inf`` turns the feature parameters into NaN.

    The returned ``params`` table gains an ``r2`` column: goodness of fit in log
    space over the positive observations, usable to filter out flat features.
    """
    if not isinstance(n_samples, int) or n_samples < 2:
        raise ValueError("n_samples must be an integer >= 2")
    t, y = df_qd.index.to_numpy(float), df_qd.to_numpy(float)
    keep = np.isfinite(t) & (t > 0)
    t, y = t[keep], y[keep]

    if len(t) < 2 or np.unique(t).size < 2:
        raise ValueError("power-law fitting needs at least two distinct positive time points")
    u = np.log(t)
    m, d = u.mean(), u.std(ddof=0)
    z = (u - m) / d

    positive = (y > 0) & np.isfinite(y)
    if ignore_zero:
        n_positive = positive.sum(axis=0)
        n_safe = np.maximum(n_positive, 1)
        r = np.zeros_like(y)
        np.log(y, out=r, where=positive)
    else:
        # Do not ignore zeros: non-positive observations enter the log statistics.
        counted = np.isfinite(y)
        n_positive = counted.sum(axis=0)
        n_safe = np.maximum(n_positive, 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            raw_log = np.log(y)
        r = np.where(np.isfinite(raw_log), raw_log, -np.inf)

    sum_z = positive.T @ z
    sum_z2 = positive.T @ (z**2)
    sum_r = r.sum(axis=0)
    sum_zr = z @ r

    z_bar, r_bar = sum_z / n_safe, sum_r / n_safe
    S_zz = sum_z2 - sum_z**2 / n_safe
    S_zr = sum_zr - sum_z * sum_r / n_safe

    valid = (n_positive >= 2) & (S_zz > 1e-12)
    beta = np.divide(S_zr, S_zz, out=np.full(y.shape[1], np.nan), where=valid)
    alpha = r_bar - beta * z_bar

    c = np.exp(alpha)
    b = beta / d
    # a = exp(alpha - b*m); clip the exponent before exp() to avoid overflow.
    a = np.exp(np.clip(alpha - b * m, -690.0, 690.0))

    # R^2 is evaluated only on the positive observations that entered the fit.
    with np.errstate(invalid="ignore"):
        resid = np.where(positive, r - (alpha[None, :] + z[:, None] * beta[None, :]), 0.0)
    ss_res = (resid**2).sum(axis=0)
    ybar = np.where(positive, r, 0.0).sum(axis=0) / n_safe
    dev = np.where(positive, r - ybar[None, :], 0.0)
    ss_tot = (dev**2).sum(axis=0)
    r2 = 1.0 - np.divide(ss_res, ss_tot, out=np.full(y.shape[1], np.nan),
                         where=ss_tot > 1e-12)

    params = pd.DataFrame(
        {"c": c, "beta": beta, "a": a, "b": b, "n_positive": n_positive, "r2": r2},
        index=df_qd.columns.copy(),
    )
    params.index.name = "feature"

    sample_t = np.linspace(t.min(), t.max(), n_samples)
    sample_z = (np.log(sample_t) - m) / d
    fitted = np.exp(alpha + sample_z[:, None] * beta)

    samples = pd.DataFrame(
        fitted, index=pd.Index(sample_t, name="quasi time"), columns=df_qd.columns,
    )

    return params, samples


def get_power_function_params(quasi_dynamic_df: pd.DataFrame) -> pd.DataFrame:
    return power_fitting(quasi_dynamic_df)[0]


# Chebyshev nodes
def chebyshev_nodes(
    n: int,
    a: float,
    b: float,
) -> np.ndarray:
    nodes = [
        0.5 * (a + b) + 0.5 * (b - a) * np.cos((2 * i + 1) * np.pi / (2 * n))
        for i in range(n)
    ]
    return np.sort(np.array(nodes))


# Power-law curve samples
def get_power_function_sample(
    quasi_dynamic_df: pd.DataFrame, n_samples: int = 30,
) -> pd.DataFrame:
    return power_fitting(quasi_dynamic_df, n_samples=n_samples)[1]


"""Dynamic (time-series) fitting: SWT stationary-wavelet soft-threshold denoising.

Per channel, a Stationary Wavelet Transform is applied; the noise level is
estimated from the finest-level detail coefficients via
``sigma = MAD / 0.6745`` and the threshold is ``alpha * sigma * sqrt(2 log N)``
(soft thresholding), followed by the inverse transform.

This runs alongside the power-law route for static data and backs the "data
type" switch on the curve-fitting page.

Requires PyWavelets (imported lazily, with a clear error if missing).
"""


DEFAULT_WAVELET = "rbio3.9"
DEFAULT_LEVEL = 2
DEFAULT_ALPHA = 0.5


def swt_denoise(
    data: pd.DataFrame,
    *,
    wavelet: str = DEFAULT_WAVELET,
    level: int = DEFAULT_LEVEL,
    alpha: float = DEFAULT_ALPHA,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply SWT soft-threshold denoising to every channel.

    Parameters
    ----------
    data:
        ``(n_timepoints, n_channels)`` time series (index = time / sample).
    wavelet:
        Wavelet basis, default ``"rbio3.9"``.
    level:
        Upper bound on the SWT decomposition level; the effective level is
        ``min(level, pywt.swt_max_level(N))``.
    alpha:
        Threshold coefficient (``alpha * sigma * sqrt(2 log N)``).

    Returns
    -------
    fitted, params
        ``fitted`` has the same shape and index as the input; ``params`` holds
        per-channel ``wavelet / level / sigma / threshold`` diagnostics.
    """
    try:
        import pywt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Dynamic-data fitting (SWT denoising) requires PyWavelets: "
            "pip install PyWavelets"
        ) from exc

    if data.empty:
        raise ValueError("input data is empty; cannot run SWT denoising")

    n_timepoints = data.shape[0]
    effective_level = min(int(level), pywt.swt_max_level(n_timepoints))
    fitted = data.copy()
    records: list[dict[str, object]] = []

    for column in data.columns:
        signal = data[column].to_numpy(dtype=float)

        if effective_level < 1:
            # Series too short for SWT: return unchanged.
            fitted[column] = signal
            records.append({
                "feature": column, "wavelet": wavelet, "level": 0,
                "sigma": 0.0, "threshold": 0.0, "denoised": False,
            })
            continue

        coefficients = pywt.swt(signal, wavelet=wavelet, level=effective_level)

        # Robust noise estimate from the finest-level detail coefficients.
        detail = coefficients[-1][1]
        sigma = float(np.median(np.abs(detail - np.median(detail))) / 0.6745)
        threshold = float(alpha * sigma * np.sqrt(2.0 * np.log(n_timepoints)))

        if sigma > 0.0:
            coefficients_fit = [
                (approx, pywt.threshold(detail_layer, threshold, mode="soft"))
                for approx, detail_layer in coefficients
            ]
            denoised = True
        else:
            # Constant channel (e.g. all zeros): thresholding would divide by zero.
            coefficients_fit = list(coefficients)
            denoised = False

        restored = pywt.iswt(coefficients_fit, wavelet=wavelet)
        fitted[column] = np.nan_to_num(
            restored, nan=0.0, posinf=0.0, neginf=0.0
        )

        records.append({
            "feature": column, "wavelet": wavelet, "level": int(effective_level),
            "sigma": sigma, "threshold": threshold, "denoised": denoised,
        })

    params = pd.DataFrame(records).set_index("feature")
    return fitted, params


"""Upload-to-results workflow for the curve-fitting step.

Two data types are supported:

``quasi_dynamic`` (default)
    Static data -> column transform -> sort rows by row sum into a
    quasi-dynamic series -> power-law fit ``y = a * x^b``.

``dynamic``
    Time series (e.g. 12-lead ECG) -> per-channel SWT soft-threshold denoising.

Both routes export the same three-table contract used downstream; in dynamic
mode the table meanings are redefined as:

    quasi_dynamic/   observed data (raw waveform, indexed by time)
    curve_sample/    SWT-denoised waveform (solver design/response input)
    curve_params/    per-channel SWT parameters (wavelet / level / sigma / threshold)
"""

import zipfile


# -- power-law fit figure (publication layout) ------------------------------
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Module palette, defined separately from the clustering palette.
_MODULE_PALETTE = (
    "#D35400", "#2E86C1", "#1E8449", "#C0392B", "#6C3483", "#117A65",
    "#B7950B", "#5D6D7E", "#A04000", "#1A5276",
)


def _module_color(module: str) -> str:
    """Stable colour per module name (M1, M2, ...)."""
    try:
        idx = int(str(module).lstrip("Mm")) - 1
    except (TypeError, ValueError):
        idx = 0
    return _MODULE_PALETTE[idx % len(_MODULE_PALETTE)]


def _fit_r2(observed: np.ndarray, fitted: np.ndarray) -> float:
    """R^2 in log space over positive observations (same convention as power_fitting)."""
    y = np.asarray(observed, dtype=float)
    f = np.asarray(fitted, dtype=float)
    pos = np.isfinite(y) & (y > 0) & np.isfinite(f) & (f > 0)
    if int(pos.sum()) < 2:
        return float("nan")
    ly, lf = np.log(y[pos]), np.log(f[pos])
    ss_res = float(np.sum((ly - lf) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")


def _power_curve_from_params(time, c: float, beta: float) -> np.ndarray:
    """Rebuild the curve from ``power_fitting`` parameters: ``y = c * exp(beta*z)``.

    ``z`` is the standardised log-time ``(log t - mean(log t)) / std(log t)``,
    so ``c`` is the curve height at the geometric-mean time.
    """
    t = np.asarray(time, dtype=float)
    ok = np.isfinite(t) & (t > 0)
    out = np.full(t.shape, np.nan, dtype=float)
    if int(ok.sum()) < 1:
        return out
    u = np.log(t)
    sd = float(u.std(ddof=0))
    z = (u - u.mean()) / (sd if sd > 1e-12 else 1.0)
    out[ok] = float(c) * np.exp(float(beta) * z[ok])
    return out


def plot_power_law_fit(
    observed_df: pd.DataFrame,
    params_df: pd.DataFrame,
    *,
    ncols: int = 4,
    max_rows: int = 9,
    figsize_per_panel: Tuple[float, float] = (8.0, 4.94),
    marker_size: float = 34.0,
    marker_alpha: float = 0.6,
    marker_linewidth: float = 1.1,
    line_width: float = 2.8,
    axis_margin: float = 0.06,
    y_margin: float = 0.28,
    log_y: bool = False,
    log_x: bool = False,
    module_map: Optional[dict] = None,
    title: str = "",
    condition_label: str = "",
    show_beta: bool = True,
    show_r2: bool = True,
    page_size: Optional[int] = None,
) -> List[Figure]:
    """Draw a publication-style power-law fit figure (observations + fitted curve).

    Differences from :func:`plot_curve_fitting`
    -------------------------------------------
    - the x-axis uses the real quasi-time (``observed_df.index``) rather than a
      sequence number, matching the definition of the power law;
    - only positive observations that entered the fit are drawn; zeros are not
      stacked along the bottom;
    - the fitted curve is rebuilt from ``(c, beta)`` on the full time grid, so
      it passes through the scatter rather than through 30 sampled points;
    - each panel is annotated with ``beta`` and ``R2`` and coloured by module.

    Parameters
    ----------
    observed_df:
        Quasi-dynamic observation table (index = quasi-time, columns = variables).
    params_df:
        Curve parameters for this condition (index = variables), with ``c`` and
        ``beta``; an existing ``r2`` column is used if present.
    ncols, max_rows:
        Grid columns and maximum rows per page (automatic pagination).
    figsize_per_panel:
        Panel size in inches; the default ``(8.0, 4.94)`` has a golden-ratio
        aspect (1.619).
    log_y:
        Log-scale the y-axis. Useful when variables span orders of magnitude.
    log_x:
        Log-scale the x-axis. Usually unnecessary because quasi-time is already
        on a log scale.
    module_map:
        ``{variable: "M1"}``; when given, colours by module and prints the module
        label in the panel corner.
    condition_label:
        Condition name added to the figure title.
    page_size:
        Variables per page; ``None`` means ``ncols * max_rows``.

    Returns
    -------
    List[Figure]
        One figure per page.
    """
    cols = [c for c in observed_df.columns if c in params_df.index]
    if not cols:
        raise ValueError("observed_df and params_df share no variables")

    per_page = int(page_size) if page_size else max(1, int(ncols) * int(max_rows))
    n_pages = int(np.ceil(len(cols) / per_page))
    t_all = observed_df.index.to_numpy(float)
    t_grid = np.linspace(float(np.nanmin(t_all)), float(np.nanmax(t_all)), 1000)

    figures: List[Figure] = []
    for page in range(n_pages):
        chunk = cols[page * per_page:(page + 1) * per_page]
        nrow = int(np.ceil(len(chunk) / ncols))
        sw, sh = figsize_per_panel
        fig, axes = plt.subplots(
            nrow, ncols, figsize=(sw * ncols, sh * nrow),
            dpi=150, squeeze=False,
            layout="constrained",
        )
        # Padding so neighbouring panels do not collide.
        if hasattr(fig, "set_constrained_layout_pads"):
            fig.set_constrained_layout_pads(w_pad=0.035, h_pad=0.055,
                                            wspace=0.035, hspace=0.055)

        for i, name in enumerate(chunk):
            ax = axes[i // ncols][i % ncols]
            y = observed_df[name].to_numpy(float)
            c = float(params_df.loc[name, "c"])
            beta = float(params_df.loc[name, "beta"])
            y_fit_obs = _power_curve_from_params(t_all, c, beta)
            y_fit_grid = _power_curve_from_params(t_grid, c, beta)

            pos = np.isfinite(y) & (y > 0) & np.isfinite(t_all)
            zero = np.isfinite(y) & (y <= 0) & np.isfinite(t_all)

            color = _module_color(module_map.get(name, "")) if module_map else "#1F77B4"

            # Ignored (non-positive) points are marked in pale grey; these cannot
            # be shown on a log y-axis.
            if zero.any() and not log_y:
                ax.scatter(t_all[zero], np.full(int(zero.sum()), np.nanmin(y[pos])
                                                if pos.any() else 0.0),
                           s=marker_size * 0.5, marker="x", linewidths=0.5,
                           color="#BDC3C7", alpha=0.6, zorder=1,
                           label="ignored (<=0)" if i == 0 else None)
            # Positive observations used by the fit.
            ax.scatter(t_all[pos], y[pos], s=marker_size, marker="o",
                       facecolors="none", edgecolors=color,
                       linewidths=marker_linewidth, alpha=marker_alpha, zorder=2,
                       label="observed (used)" if i == 0 else None)
            # Fitted curve on the full time grid.
            ax.plot(t_grid, y_fit_grid, "-", color=color, linewidth=line_width,
                    solid_capstyle="round", zorder=3,
                    label=r"$y=c\,e^{\beta z}$" if i == 0 else None)

            if log_y:
                ax.set_yscale("log")
            if log_x:
                ax.set_xscale("log")
                # Quasi-time is already log-scaled, so the span may be under one
                # decade and matplotlib would emit overlapping minor ticks.
                lo_x = max(float(np.nanmin(t_all)), 1e-12)
                hi_x = float(np.nanmax(t_all))
                decades = np.log10(hi_x) - np.log10(lo_x)
                if decades < 1.0:
                    ticks = np.linspace(lo_x, hi_x, 6)
                    ax.set_xticks(ticks)
                    ax.set_xticklabels([f"{v:.2f}" for v in ticks], fontsize=7.5)

            # Do not call set_xlim/set_ylim here: an explicit limit overrides
            # ax.margins() and the padding would collapse to zero.
            ax.margins(x=axis_margin, y=y_margin)

            r2 = (float(params_df.loc[name, "r2"])
                  if "r2" in params_df.columns else _fit_r2(y, y_fit_obs))
            tag = []
            if show_beta:
                tag.append(rf"$\beta$={beta:+.3f}")
            if show_r2 and np.isfinite(r2):
                tag.append(rf"$R^2$={r2:.3f}")
            ax.set_title(name, fontsize=9.5, pad=16 if tag else 6)
            ax.set_xlabel("Quasi-time (\u03c4)", fontsize=8.5)
            if i % ncols == 0:
                ax.set_ylabel("Abundance (log scale)" if log_y else "Abundance",
                              fontsize=8.5)
            if tag:
                ax.text(0.5, 1.02, "  ".join(tag), transform=ax.transAxes,
                        ha="center", va="bottom", fontsize=8)
            if module_map and name in module_map:
                ax.text(0.03, 0.96, str(module_map[name]), transform=ax.transAxes,
                        ha="left", va="top", fontsize=8.5, color=color,
                        fontweight="bold")
            ax.tick_params(labelsize=8, width=0.9, length=3.5)
            for spine in ax.spines.values():
                spine.set_linewidth(0.9)
            ax.grid(True, linestyle=":", linewidth=0.55, alpha=0.45)

        for j in range(len(chunk), nrow * ncols):
            axes[j // ncols][j % ncols].set_visible(False)

        head = " ".join(x for x in (title, condition_label) if x)
        if n_pages > 1:
            head = f"{head}  (page {page + 1}/{n_pages})".strip()
        if head:
            fig.suptitle(head, fontsize=11)
        if n_pages == 1 and len(chunk) > 0:
            h, l = axes[0][0].get_legend_handles_labels()
            if h:
                fig.legend(h, l, loc="outside lower center", ncol=3, fontsize=8,
                           frameon=False)
        figures.append(fig)

    return figures


def plot_power_law_summary(
    params_df: pd.DataFrame,
    *,
    module_map: Optional[dict] = None,
    condition_label: str = "",
    figsize: Tuple[float, float] = (11.0, 3.4),
    bins: int = 24,
) -> Figure:
    """Parameter summary: exponent beta, R^2 distribution, and beta vs R^2.

    Parameters
    ----------
    params_df:
        Curve parameter table (with ``beta`` / ``r2``).
    module_map:
        ``{variable: "M1"}``; when given, colours points by module.
    """
    beta = pd.to_numeric(params_df["beta"], errors="coerce").to_numpy(float)
    r2 = (pd.to_numeric(params_df["r2"], errors="coerce").to_numpy(float)
          if "r2" in params_df.columns else np.full(len(params_df), np.nan))
    names = list(params_df.index)
    colors = ([_module_color(module_map.get(n, "")) for n in names]
              if module_map else ["#2E86C1"] * len(names))

    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=150, layout="constrained")

    ax = axes[0]
    ax.hist(beta[np.isfinite(beta)], bins=bins, color="#2E86C1",
            edgecolor="white", linewidth=0.6)
    ax.axvline(0.0, color="#C0392B", linestyle="--", linewidth=1.2)
    ax.set_xlabel(r"power-law exponent $\beta$", fontsize=9)
    ax.set_ylabel("number of variables", fontsize=9)
    ax.set_title(rf"$\beta$ distribution (median={np.nanmedian(beta):+.3f})", fontsize=10)
    ax.tick_params(labelsize=8)
    ax.grid(True, linestyle=":", alpha=0.4)

    ax = axes[1]
    ax.hist(r2[np.isfinite(r2)], bins=bins, color="#1E8449",
            edgecolor="white", linewidth=0.6)
    ax.set_xlabel(r"goodness of fit $R^2$ (log space)", fontsize=9)
    ax.set_ylabel("number of variables", fontsize=9)
    ax.set_title(rf"$R^2$ distribution (median={np.nanmedian(r2):.3f})", fontsize=10)
    ax.tick_params(labelsize=8)
    ax.grid(True, linestyle=":", alpha=0.4)

    ax = axes[2]
    ax.scatter(beta, r2, s=28, c=colors, alpha=0.8, edgecolors="white",
               linewidths=0.5)
    ax.set_xlabel(r"exponent $\beta$", fontsize=9)
    ax.set_ylabel(r"$R^2$", fontsize=9)
    ax.set_title(r"$\beta$ vs $R^2$", fontsize=10)
    ax.tick_params(labelsize=8)
    ax.grid(True, linestyle=":", alpha=0.4)

    head = "Power-law fit summary"
    if condition_label:
        head = f"{head} - {condition_label}"
    fig.suptitle(head, fontsize=11)
    return fig


def fit_uploaded_csv(
    content: bytes, *, data_mode: str = DATA_MODE_QUASI,
    first_transform: str = "Z_min_add1",
    second_transform: str = "Log10_1p", n_samples: int = 30,
    trim_percent: float = 1.0,
    swt_wavelet: str = SWT_DEFAULT_WAVELET,
    swt_level: int = SWT_DEFAULT_LEVEL,
    swt_alpha: float = SWT_DEFAULT_ALPHA,
    ignore_zero: bool = True,
) -> dict[str, pd.DataFrame]:
    """Fit an uploaded CSV according to the selected data type.

    Parameters
    ----------
    ignore_zero:
        Whether to ignore zeros when fitting the power law (quasi-dynamic route
        only). Zero-inflated data require ``True``, otherwise zeros flatten many
        curves and break the clustering. See :func:`power_fitting`.

    Returns
    -------
    dict
        Keys are ``quasi_dynamic`` / ``curve_params`` / ``curve_sample``.
        ``curve_params`` carries an additional ``r2`` column for variable
        filtering.
    """
    original = pd.read_csv(io.BytesIO(content), index_col=0)
    if original.empty or original.columns.has_duplicates:
        raise ValueError("the CSV must contain data and feature names must be unique")

    if str(data_mode) == DATA_MODE_DYNAMIC:
        numeric = original.apply(pd.to_numeric, errors="coerce")
        if numeric.to_numpy(dtype=float).size == 0 or not np.isfinite(
            numeric.to_numpy(dtype=float)
        ).any():
            raise ValueError("dynamic mode requires numeric time-series columns")
        fitted, params = swt_denoise(
            numeric,
            wavelet=str(swt_wavelet),
            level=int(swt_level),
            alpha=float(swt_alpha),
        )
        return {
            "quasi_dynamic": numeric,
            "curve_sample": fitted,
            "curve_params": params,
        }

    if not 0 <= trim_percent < 100:
        raise ValueError("trim_percent must be in [0, 100)")
    transformed = preprocess(preprocess(original, first_transform), second_transform)
    sums = transformed.sum(axis=1).to_numpy()
    order = np.argsort(sums, kind="stable")
    quasi = transformed.iloc[order].copy()
    quasi.index = pd.Index(np.log1p(sums[order]), name="quasi time")
    quasi = quasi.loc[quasi.index > 0]
    quasi = quasi.iloc[int(trim_percent / 100 * len(quasi)):]
    params, samples = power_fitting(quasi, n_samples=n_samples, ignore_zero=ignore_zero)
    return {"quasi_dynamic": quasi, "curve_params": params, "curve_sample": samples}


def build_fitting_export(results: dict[str, dict[str, pd.DataFrame]]) -> bytes:
    """Build the three-table ZIP export consumed by downstream steps."""
    buffer = io.BytesIO()
    used = set()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, tables in results.items():
            folder = name.replace("\\", "_").replace("/", "_").strip() or "export"
            if folder in used:
                raise ValueError("duplicate file name after normalisation; rename and retry")
            used.add(folder)
            for table in ("quasi_dynamic", "curve_params", "curve_sample"):
                archive.writestr(f"{folder}/{table}.csv", tables[table].to_csv(index=True))
    return buffer.getvalue()


import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

font_prop = None  # Set by the application layer for CJK font support


def plot_curve_fitting(
    df_scatter,
    df_curve,
    plot_scatter_type="line",
    show_curve=True,

    # layout
    nrow=4,
    ncol=3,
    nsubfig=4,
    scatter_x="sequence",
    scatter_size=100,
    scatter_linewidth=2,

    margins_x=0.1,
    margins_y=0.2,

    # labels
    # title="",

    # colours
    color_scatter="#1F77B4",
    color_curve="#D62728",
    subfig_background_color="#FFFFFF",
):
    plt.rcParams.update({
        "axes.linewidth": 1.2,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "font.family": "sans-serif",
        # This rcParams update is global; keep the ASCII hyphen as the minus
        # sign, otherwise CJK fonts render U+2212 as a missing glyph.
        "axes.unicode_minus": False,
    })

    GOLDEN_RATIO = 1.618
    base_height = 1.5
    base_width = base_height * GOLDEN_RATIO

    selected_cols = df_scatter.columns.tolist()
    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(base_width * ncol, base_height * nrow),
        sharex=True,
        dpi=300
    )

    if nrow * ncol == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    for i, col in enumerate(selected_cols[:nsubfig]):
        if scatter_x == "index":
            x = df_scatter.index
        elif scatter_x == "sequence":
            x = np.arange(len(df_scatter))
        else:
            x = np.arange(len(df_scatter))

        if plot_scatter_type == "scatter":
            axes[i].scatter(
                x,
                df_scatter[col],
                alpha=0.75,
                s=scatter_size,
                facecolors='none',
                edgecolors=color_scatter,
                linewidth=0.8,
                zorder=1
            )
        elif plot_scatter_type == "line":
            axes[i].plot(
                x,
                df_scatter[col],
                alpha=0.8,
                color=color_scatter,
                linewidth=scatter_linewidth,
                zorder=1
            )

        if show_curve and df_curve is not None and col in df_curve.columns:
            axes[i].plot(
                df_curve.index,
                df_curve[col],
                color=color_curve,
                linewidth=2.5,
                zorder=5
            )

        axes[i].set_title(col, fontsize=11, fontweight='bold', fontproperties=font_prop, pad=6)
        axes[i].margins(x=margins_x, y=margins_y)

        # Limit the number of ticks to avoid crowding.
        axes[i].xaxis.set_major_locator(plt.MaxNLocator(4))
        axes[i].yaxis.set_major_locator(plt.MaxNLocator(4))

        axes[i].tick_params(top=True, right=True, direction='in')

        for label in axes[i].get_xticklabels():
            label.set_fontproperties(font_prop)
        for label in axes[i].get_yticklabels():
            label.set_fontproperties(font_prop)

        axes[i].set_facecolor(subfig_background_color)

    n_plotted = min(len(selected_cols), nsubfig)
    for j in range(n_plotted, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=1.0)

    return fig


def plot_curve_fitting_compare(
    df_scatter_list,
    df_curve_list,
    label_list,
    show_curve=True,
    nrow=2,
    ncol=2,
    nsubfig=4,
):
    scatter_colors = ['#F9B3AD', '#C9A1CA', '#76C2AF', '#E5C68F', '#C59FCE', '#A0D8E7']
    curve_colors = ['#9E2223', '#A06EA5', '#52B793', '#DDB866', '#B488C2', '#79C6DF']

    selected_cols = df_scatter_list[0].columns.tolist()
    fig, axes = plt.subplots(nrow, ncol, figsize=(6, 3), sharex=True, sharey=True, dpi=300)
    axes = np.array(axes).flatten()

    all_handles = []
    all_labels = []

    for i, col in enumerate(selected_cols[:nsubfig]):
        ax = axes[i]

        for idx, (df_scatter, df_curve) in enumerate(zip(df_scatter_list, df_curve_list)):
            color_scatter = scatter_colors[idx % len(scatter_colors)]
            color_curve = curve_colors[idx % len(scatter_colors)]
            label = label_list[idx]

            scatter = ax.scatter(
                df_scatter.index, df_scatter[col],
                alpha=0.95, s=100, facecolors='none',
                edgecolors=color_scatter, linewidth=1,
            )
            curve = None
            if show_curve:
                curve = ax.plot(
                    df_curve.index, df_curve[col],
                    color=color_curve, linewidth=3,
                )[0]

            if i == 0:
                all_handles.append(scatter)
                all_labels.append(f"{label} observed")
                if show_curve:
                    all_handles.append(curve)
                    all_labels.append(f"{label} fitted")

        ax.set_title(col, fontsize=11, fontproperties=font_prop)
        ax.margins(x=0.1, y=0.3)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        for label in ax.get_xticklabels():
            label.set_fontproperties(font_prop)
        for label in ax.get_yticklabels():
            label.set_fontproperties(font_prop)

    fig.legend(
        all_handles, all_labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.15),
        ncol=4,
        fontsize=8,
        frameon=False,
        prop=font_prop
    )

    n_plotted = min(len(selected_cols), nsubfig)
    for j in range(n_plotted, len(axes)):
        axes[j].set_visible(False)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    return fig
