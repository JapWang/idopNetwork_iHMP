# -*- coding: utf-8 -*-
"""
Network reconstruction for static and quasi-dynamic data.

Legendre basis expansion with constrained sparse regression
(``IDOPRegressor`` / ``StaticIDOPRegressor``), a multi-window LASSO edge
selector, a dynamic (time-series) solver combining integrated Fourier and
Legendre bases in a two-stage least-squares fit, and interactive HTML network
figures. Class names, function names, parameter defaults and all computations
are unchanged from the original implementation.
"""
from __future__ import annotations

from math import comb
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import eval_legendre, legendre
from sklearn.linear_model import Lasso

# Imports from the original per-module file headers, collected here.
import hashlib
import json
import os
import zipfile

import matplotlib.pyplot as plt

MIN_ASGL_MAX_ORDER = 5


def polynomial_basis_expansion(
    data: pd.DataFrame,
    max_order: int,
) -> pd.DataFrame:
    """
    Legendre basis expansion; column ``(k, r)`` is ``y_k(tau) * P_r(tau_hat)``.

    Orders run over ``r = 0 .. max_order - 1`` so every variable contributes
    ``max_order`` columns (``P_0 == 1``, so r=0 is ``y_k`` itself). The index of
    ``data`` is the time axis, normalised internally to ``tau_hat in [-1, 1]``
    (set to 0 when the index is constant). Columns are ordered by variable and then
    by order, named ``f"{feature}_o({r})"``, so downstream code can group by
    ``k * max_order + r``.
    """
    values = data.values.astype(float)
    n_samples, n_features = values.shape

    tau = np.asarray(data.index, dtype=float)
    tau_min = float(tau.min())
    tau_max = float(tau.max())
    if tau_max > tau_min:
        tau_hat = -1.0 + 2.0 * (tau - tau_min) / (tau_max - tau_min)
    else:
        tau_hat = np.zeros_like(tau)

    per_order_tau = [eval_legendre(order, tau_hat) for order in range(max_order)]
    basis_tau = np.stack(per_order_tau, axis=0)
    basis_arr = basis_tau[:, :, np.newaxis] * values[np.newaxis, :, :]
    basis_arr = basis_arr.transpose(1, 2, 0)

    columns = [
        f"{data.columns[i]}_o({order})"
        for i in range(n_features)
        for order in range(max_order)
    ]
    flat = basis_arr.reshape(n_samples, n_features * max_order)
    return pd.DataFrame(flat, index=data.index, columns=columns)


def tiger_lop_basis_expansion(
    data: pd.DataFrame,
    max_order: int,
) -> pd.DataFrame:
    """
    TIGER/LOP-style per-source basis: integrate Legendre polynomials of a curve.

    Each source curve ``x_k(t)`` is scaled to ``[-1, 1]`` and ``P_r(x_k)`` is
    trapezoid-integrated along that scaled coordinate. Every basis column starts at
    0, so the self effect is carried by the target initial value alone and the
    cross effects start at 0 by construction.
    """
    values = data.values.astype(float)
    n_samples, n_features = values.shape
    out = np.zeros((n_samples, n_features * max_order), dtype=float)
    columns: list[str] = []

    for k, feature_name in enumerate(data.columns):
        x = values[:, k]
        # Replace non-finite entries with the column median so that Inf/Inf does
        # not turn the whole basis column into NaN.
        finite_mask = np.isfinite(x)
        if not np.all(finite_mask):
            finite_vals = x[finite_mask]
            if finite_vals.size == 0:
                # Entire column non-finite: fall back to zeros (filtered downstream).
                out[:, k * max_order : (k + 1) * max_order] = 0.0
                for order in range(max_order):
                    columns.append(f"{feature_name}_o({order})")
                continue
            fill_val = float(np.median(finite_vals))
            x = np.where(finite_mask, x, fill_val)
        x_min = float(np.min(x))
        x_max = float(np.max(x))
        if x_max > x_min:
            x_scaled = -1.0 + 2.0 * (x - x_min) / (x_max - x_min)
        else:
            x_scaled = np.zeros_like(x)
        dx = np.diff(x_scaled)
        for order in range(max_order):
            deriv_values = eval_legendre(order, x_scaled)
            integral = np.zeros(n_samples, dtype=float)
            if n_samples > 1:
                increments = 0.5 * (
                    deriv_values[1:] + deriv_values[:-1]
                ) * dx
                integral[1:] = np.cumsum(increments)
            out[:, k * max_order + order] = integral
            columns.append(f"{feature_name}_o({order})")

    return pd.DataFrame(out, index=data.index, columns=columns)


def polynomial_basis_expansion_integral_analytic(
    data: pd.DataFrame,
    power_function_params: pd.DataFrame,
    max_order: int,
) -> pd.DataFrame:
    """
    Analytic design-matrix columns for integral mode.

    Assumes ``y_k(tau) = a_k * tau**b_k`` and evaluates
    ``int_{tau_1}^{tau} a_k s^{b_k} P_r(tau_hat(s)) ds`` at each sample, which
    removes the trapezoid error of ``cumulative_trapezoid``. ``a_k`` and ``b_k``
    come from the ``"a"`` / ``"b"`` rows of ``power_function_params``. Expanding
    ``P_r(x) = sum_m c_{r,m} x^m`` with ``x = alpha*s + beta`` gives per-term
    antiderivatives ``s^{b_k+j+1} / (b_k+j+1)``; the ``b_k+j+1 == 0`` case is
    handled by a ``ln`` branch. Column order matches
    :func:`polynomial_basis_expansion`.
    """
    tau = np.asarray(data.index, dtype=float)
    n_samples = tau.shape[0]
    feature_names = list(data.columns)
    n_features = len(feature_names)

    params = power_function_params.reindex(feature_names)
    if params.isna().to_numpy().any():
        missing = params.index[params.isna().any(axis=1)].tolist()
        raise ValueError(
            "power_function_params is not aligned with the curve columns; missing a/b:"
            f"{missing}",

        )
    a_arr = params["a"].to_numpy(dtype=float)
    b_arr = params["b"].to_numpy(dtype=float)

    tau_min = float(tau.min())
    tau_max = float(tau.max())
    tau1 = tau_min
    if tau_max > tau_min:
        alpha = 2.0 / (tau_max - tau_min)
        beta = -1.0 - alpha * tau_min
    else:
        alpha, beta = 0.0, 0.0

    out = np.zeros((n_samples, n_features * max_order), dtype=float)
    for r in range(max_order):
        cr = legendre(r).coef[::-1]
        d_rj = np.zeros(r + 1, dtype=float)
        for j in range(r + 1):
            acc = 0.0
            for m in range(j, r + 1):
                acc += cr[m] * comb(m, j) * (alpha ** j) * (beta ** (m - j))
            d_rj[j] = acc
        for k in range(n_features):
            col_idx = k * max_order + r
            col_vals = np.zeros(n_samples, dtype=float)
            for j in range(r + 1):
                exp = b_arr[k] + j + 1
                coef = a_arr[k] * d_rj[j]
                if coef == 0.0:
                    continue
                if abs(exp) < 1e-12:
                    safe_tau1 = tau1 if tau1 > 0 else np.finfo(float).tiny
                    safe_tau = np.where(tau > 0, tau, np.finfo(float).tiny)
                    col_vals += coef * (np.log(safe_tau) - np.log(safe_tau1))
                else:
                    col_vals += coef * (
                        np.power(tau, exp) - np.power(tau1, exp)
                    ) / exp
            out[:, col_idx] = col_vals

    columns = [
        f"{feature_names[k]}_o({r})"
        for k in range(n_features)
        for r in range(max_order)
    ]
    return pd.DataFrame(out, index=data.index, columns=columns)


def align_response_to_design(
    response_df: pd.DataFrame,
    target_index: pd.Index | np.ndarray,
) -> pd.DataFrame:
    """
    Linearly interpolate the response onto ``target_index``.

    The source index and the target index generally differ, so a plain ``reindex``
    would produce all-NaN; values are aligned by numeric index with
    :func:`numpy.interp` (sources sorted ascending, points outside the source range
    taking the nearest endpoint, i.e. ``fill_value=(y[0], y[-1])``). Column names
    and order are preserved; the result is indexed by ``target_index``. For input
    already on the target index this is a no-op.
    """
    src_idx = np.asarray(response_df.index, dtype=float)
    tgt_idx = np.asarray(target_index, dtype=float)
    order = np.argsort(src_idx, kind="stable")
    src_sorted = src_idx[order]
    values_sorted = response_df.values.astype(float)[order]
    n_targets = values_sorted.shape[1]
    out = np.empty((len(tgt_idx), n_targets), dtype=float)
    for j in range(n_targets):
        out[:, j] = np.interp(tgt_idx, src_sorted, values_sorted[:, j])
    return pd.DataFrame(out, index=pd.Index(tgt_idx), columns=response_df.columns)


def _apply_column_normalization(
    Xv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pure L2 unit-scaling of the non-intercept design columns (no centring).

    Column 0 is assumed to be the intercept and is left untouched; every other
    column is divided by its own L2 norm, so columns have unit L2 length in the
    scaled space while their means are unchanged. Returns
    ``(Xv_scaled, mu, sigma)`` with ``mu`` all zeros (a placeholder that keeps the
    :func:`_denormalize_coefficients` interface uniform), ``sigma[0] = 1`` and
    ``sigma[c]`` the original L2 norm (set to 1 for norms below ``1e-12``). Because
    this is a pure column scaling, "fixed intercept = y0" OLS / ASGL in the scaled
    space is exactly equivalent to the original space, so the effect decomposition
    ``predict_j = W[0, j] + sum_k effect_{k->j}`` keeps its meaning.
    """
    p = Xv.shape[1]
    Xv_scaled = Xv.astype(float, copy=True)
    mu = np.zeros(p, dtype=float)
    sigma = np.ones(p, dtype=float)
    if p <= 1:
        return Xv_scaled, mu, sigma
    body = Xv_scaled[:, 1:]
    norms = np.linalg.norm(body, axis=0)
    sigma[1:] = np.where(norms > 1e-12, norms, 1.0)
    Xv_scaled[:, 1:] = body / sigma[1:]
    return Xv_scaled, mu, sigma


def _denormalize_coefficients(
    W_scaled: np.ndarray, mu: np.ndarray, sigma: np.ndarray
) -> np.ndarray:
    """
    Map the scaled-space coefficient matrix back to the original space.

    Given ``X_scaled = [1 | (Phi - mu) / sigma]`` and ``W_scaled = [beta_t; theta_t]``
    this returns ``theta_orig = theta_t / sigma`` and
    ``beta_orig = beta_t - mu^T theta_orig``, so that
    ``X_orig @ W_orig == X_scaled @ W_scaled``. The current scaling has ``mu == 0``,
    so the formulas degenerately reduce to ``beta_orig = beta_t``; ``mu`` is kept in
    the signature so a different normalisation could reuse the same interface.
    """
    W = np.zeros_like(W_scaled, dtype=float)
    W[1:, :] = W_scaled[1:, :] / sigma[1:, np.newaxis]
    W[0, :] = W_scaled[0, :] - mu[1:] @ W[1:, :]
    return W


def _project_nonneg_self_keep_intercept(
    w: np.ndarray,
    Xv: np.ndarray,
    intercept_idx: int,
    self_group: np.ndarray,
    pinv: np.ndarray | None = None,
    allow_intercept_relax: bool = False,
) -> None:
    """
    Project so that the self effect ``beta + Phi_self @ theta_self >= 0`` everywhere.

    By default the intercept is left strictly untouched: ``theta_self`` is projected
    by least squares onto the non-negativity requirement, matching the semantics of
    an explicitly locked ``intercept_values``. With ``allow_intercept_relax=True``
    the intercept is raised by exactly the negative residual as a fallback when the
    self columns cannot span the target, which avoids distorting ``theta_self``.
    """
    e = Xv[:, self_group] @ w[self_group] + w[intercept_idx]
    if float(e.min()) >= 0:
        return
    e_clamped = np.maximum(e, 0.0)
    e_self_target = e_clamped - w[intercept_idx]
    if pinv is not None:
        theta_new = pinv @ e_self_target
    else:
        theta_new = np.linalg.lstsq(
            Xv[:, self_group], e_self_target, rcond=None
        )[0]
    w[self_group] = theta_new
    if not allow_intercept_relax:
        return
    e_new = Xv[:, self_group] @ theta_new + w[intercept_idx]
    gap = float(e_new.min())
    if gap < -1e-9:
        w[intercept_idx] += -gap


def _precompute_pinvs(Xv: np.ndarray, groups: list[np.ndarray]) -> dict[int, np.ndarray]:
    """Precompute the pseudo-inverse of every non-intercept group."""
    pinvs: dict[int, np.ndarray] = {}
    for gi, g in enumerate(groups):
        if gi == 0:
            continue
        pinvs[gi] = np.linalg.pinv(Xv[:, g])
    return pinvs


def _asgl_col(
    Xv: np.ndarray,
    y: np.ndarray,
    groups: list[np.ndarray],
    lam: float,
    mix: float,
    coef_weights: np.ndarray,
    group_weights: np.ndarray,
    protected_ids: set[int] | None = None,
    target_idx: int = -1,
    nonneg_self: bool = False,
    max_iter: int = 2000,
    tol: float = 1e-5,
    w_init: np.ndarray | None = None,
    pinvs: dict[int, np.ndarray] | None = None,
    XtX: np.ndarray | None = None,
    Xty: np.ndarray | None = None,
    L: float | None = None,
    fixed_intercept: float | None = None,
) -> np.ndarray:
    """Adaptive Sparse Group Lasso (ASGL) proximal gradient descent."""
    if protected_ids is None:
        protected_ids = set()
    n, p = Xv.shape
    if XtX is None:
        XtX = Xv.T @ Xv
    if Xty is None:
        Xty = Xv.T @ y
    if L is None:
        L = float(np.linalg.norm(XtX, ord=2))
    if L == 0:
        return np.zeros(p)
    step = 1.0 / L

    self_pinv = pinvs.get(target_idx) if pinvs else None
    w = w_init.copy() if w_init is not None else np.zeros(p)
    if fixed_intercept is not None:
        w[0] = fixed_intercept

    for it in range(max_iter):
        w_prev = w.copy()
        grad = XtX @ w - Xty
        u = w - step * grad
        if fixed_intercept is not None:
            u[0] = fixed_intercept

        for gi, g in enumerate(groups):
            if len(g) == 0:
                continue
            if gi == 0 or gi in protected_ids:
                w[g] = u[g]
                continue
            ug = u[g].copy()
            if mix > 0:
                thresh_l1 = lam * mix * coef_weights[g] * step
                ug = np.sign(ug) * np.maximum(np.abs(ug) - thresh_l1, 0.0)
            norm_ug = np.linalg.norm(ug)
            if norm_ug == 0.0:
                w[g] = 0.0
            elif mix < 1:
                pg = float(len(g))
                thresh_grp = (
                    lam * (1.0 - mix) * float(group_weights[gi]) * np.sqrt(pg) * step
                )
                if norm_ug > thresh_grp:
                    w[g] = (1.0 - thresh_grp / norm_ug) * ug
                else:
                    w[g] = 0.0
            else:
                w[g] = ug

        if (it % 10 == 0 or it < 10) and nonneg_self and target_idx > 0:
            _project_nonneg_self_keep_intercept(
                w,
                Xv,
                0,
                groups[target_idx],
                pinv=self_pinv,
                allow_intercept_relax=fixed_intercept is None,
            )

        if np.linalg.norm(w - w_prev) < tol:
            break

    if nonneg_self and target_idx > 0:
        _project_nonneg_self_keep_intercept(
            w,
            Xv,
            0,
            groups[target_idx],
            pinv=self_pinv,
            allow_intercept_relax=fixed_intercept is None,
        )
    return w


def _build_groups(n_features: int, max_order: int) -> list[np.ndarray]:
    """Group list: the intercept group followed by max_order columns per source."""
    groups: list[np.ndarray] = [np.array([0])]
    for k in range(n_features):
        start = 1 + k * max_order
        groups.append(np.arange(start, start + max_order))
    return groups


def _range(values: np.ndarray) -> float:
    """Finite range (max - min) of a 1-D array."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    return float(np.max(finite) - np.min(finite))


class IDOPRegressor:
    """
    Multi-output linear regression fitted by a single-stage self-dominant ASGL.

    Each target ``j`` gets an Adaptive Sparse Group Lasso on the full design matrix
    ``[intercept | Phi]``. Self and cross groups share one objective, but the cross
    group-lasso / L1 penalties are multiplied by ``cross_weight_ratio`` (default 5),
    so cross terms are more expensive to select and the self group stays dominant at
    the same ``alpha``. BIC searches an outer ``max_order`` grid and an inner
    ``mix`` x ``alpha`` grid. ``nonneg_self`` projects the self group after each
    iteration and at convergence, ``max_interactions`` applies Top-K cross-variable
    retention followed by one merged OLS on ``[self_g | keep_cross_g]``, and
    ``adaptive_weights`` switches the ASGL weights from structural to
    signal-adaptive.
    """

    def __init__(
        self,
        max_order: int,
        alpha: float = 1.0,
        mix: float = 0.5,
        fix_mix: bool = False,
        nonneg_self: bool = True,
        basis_decay: float = 0.0,
        max_interactions: int = 0,
        basis_type: str = "integral",
        ebic_gamma: float = 0.0,
        adaptive_weights: bool = False,
        cross_weight_ratio: float = 5.0,
        enforce_effect_constraints: bool = True,
    ):
        self.max_order = max_order
        self.alpha = alpha
        self.mix = mix
        self.fix_mix = fix_mix
        self.nonneg_self = nonneg_self
        self.basis_decay = basis_decay
        self.max_interactions = max_interactions
        self.basis_type = basis_type
        self.ebic_gamma = ebic_gamma
        self.adaptive_weights = adaptive_weights
        self.cross_weight_ratio = float(cross_weight_ratio)
        self.enforce_effect_constraints = bool(enforce_effect_constraints)
        self.coef_: pd.DataFrame | None = None
        self.mse_: float | None = None
        self.bic_order_path_: pd.DataFrame | None = None
        self.bic_alpha_path_: pd.DataFrame | None = None
        self.power_function_params_: pd.DataFrame | None = None
        self.effect_constraint_directions_: pd.Series | None = None
        self.effect_constraint_diagnostics_: pd.DataFrame | None = None

    def _design(self, power_function_sample_df: pd.DataFrame) -> pd.DataFrame:
        basis = tiger_lop_basis_expansion(power_function_sample_df, self.max_order)
        if self.basis_decay > 0.0:
            n_feat = power_function_sample_df.shape[1]
            for r in range(self.max_order):
                scale = float(np.exp(-self.basis_decay * r))
                cols = [k * self.max_order + r for k in range(n_feat)]
                basis.iloc[:, cols] *= max(scale, 1e-10)
        intercept = pd.DataFrame(1.0, index=basis.index, columns=["intercept"])
        return pd.concat([intercept, basis], axis=1)

    def _group_size(self) -> int:
        """Number of design-matrix columns per source variable."""
        return self.max_order

    def _build_self_cross_weights(
        self,
        p: int,
        groups: list[np.ndarray],
        n_targets: int,
        Xv: np.ndarray | None = None,
        Y: np.ndarray | None = None,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Per-target ``(coef_weights, group_weights)`` for the ASGL penalties.

        Structural weights (``adaptive_weights=False``, the default) give self-group
        coefficients weight 1 and cross-group coefficients ``cross_weight_ratio``; the
        intercept group is not penalised inside ``_asgl_col`` and its weight is only a
        placeholder. Adaptive weights first estimate ``W_ols`` by OLS on the full design
        matrix, build the usual ``1 / (|.| + eps)`` ASGL weights and then multiply by the
        self/cross ratio.
        """
        ratio = self.cross_weight_ratio
        caws: list[np.ndarray] = []
        gaws: list[np.ndarray] = []
        eps = 1e-6

        if self.adaptive_weights and Xv is not None and Y is not None:
            W_ols, *_ = np.linalg.lstsq(Xv, Y, rcond=None)
        else:
            W_ols = None

        for j in range(n_targets):
            cw = np.ones(p, dtype=float)
            gw = np.ones(len(groups), dtype=float)
            for gi, g in enumerate(groups):
                if gi == 0:
                    continue
                base = 1.0 if gi == j + 1 else ratio
                if W_ols is not None:
                    elem_aw = 1.0 / (np.abs(W_ols[g, j]) + eps)
                    grp_aw = 1.0 / (float(np.linalg.norm(W_ols[g, j])) + eps)
                else:
                    elem_aw = np.ones(len(g), dtype=float)
                    grp_aw = 1.0
                cw[g] = base * elem_aw
                gw[gi] = base * grp_aw
            caws.append(cw)
            gaws.append(gw)
        return caws, gaws

    def _check_effect_constraints(
        self,
        Xv: np.ndarray,
        W: np.ndarray,
        groups: list[np.ndarray],
        target_names: list[str],
        target_indices: list[int] | None = None,
    ) -> tuple[bool, dict[str, str], list[dict[str, float | str | bool]]]:
        """Check the pointwise hard constraint of self effect against total effect."""
        if not self.enforce_effect_constraints:
            directions = {name: "not_enforced" for name in target_names}
            diagnostics = [
                {
                    "target": name,
                    "valid": True,
                    "direction": "not_enforced",
                    "reason": "not_enforced",
                    "min_delta": np.nan,
                    "max_delta": np.nan,
                    "mean_abs_delta": np.nan,
                    "self_dynamic_range": np.nan,
                    "total_range": np.nan,
                    "self_dynamic_frac": np.nan,
                }
                for name in target_names
            ]
            return True, directions, diagnostics

        directions: dict[str, str] = {}
        diagnostics: list[dict[str, float | str | bool]] = []
        all_valid = True
        gap_min = 1e-6
        mean_gap_min = 0.0

        if target_indices is None:
            target_indices = list(range(len(target_names)))

        for local_j, target_name in enumerate(target_names):
            j = target_indices[local_j]
            self_group = groups[j + 1]
            w_col = W[:, local_j] if W.shape[1] == len(target_names) else W[:, j]
            total_curve = Xv @ w_col
            self_dynamic = Xv[:, self_group] @ w_col[self_group]
            self_curve = w_col[0] + self_dynamic
            delta = self_curve - total_curve
            constraint_delta = delta[1:] if delta.shape[0] > 1 else delta

            min_delta = float(np.min(constraint_delta))
            max_delta = float(np.max(constraint_delta))
            mean_abs_delta = float(np.mean(np.abs(constraint_delta)))
            self_dynamic_range = _range(self_dynamic)
            total_range = _range(total_curve)
            if total_range > 1e-12:
                self_dynamic_frac = self_dynamic_range / total_range
            else:
                self_dynamic_frac = np.inf if self_dynamic_range > 0 else 0.0

            # When the self effect dominates the dynamic range, cross effects are
            # negligible and the direction constraint holds without a gap check.
            if self_dynamic_frac > 0.999:
                direction = "self_only"
                direction_ok = True
            else:
                above_ok = min_delta >= gap_min
                below_ok = max_delta <= -gap_min
                if above_ok:
                    direction = "self_above_total"
                    direction_ok = True
                elif below_ok:
                    direction = "self_below_total"
                    direction_ok = True
                else:
                    direction = "invalid"
                    direction_ok = False

            reason = "ok"
            target_valid = True
            if not direction_ok:
                reason = "direction_violation"
                target_valid = False
            elif direction != "self_only" and mean_abs_delta < mean_gap_min:
                reason = "mean_gap_too_small"
                target_valid = False

            if not target_valid:
                all_valid = False
            directions[target_name] = direction
            diagnostics.append(
                {
                    "target": target_name,
                    "valid": target_valid,
                    "direction": direction,
                    "reason": reason,
                    "min_delta": min_delta,
                    "max_delta": max_delta,
                    "mean_abs_delta": mean_abs_delta,
                    "self_dynamic_range": self_dynamic_range,
                    "total_range": total_range,
                    "self_dynamic_frac": float(self_dynamic_frac),
                }
            )

        return all_valid, directions, diagnostics

    def _repair_effect_constraints(
        self,
        Xv: np.ndarray,
        W: np.ndarray,
        groups: list[np.ndarray],
    ) -> np.ndarray:
        """Push a candidate solution into the feasible direction region via cross offsets."""
        if not self.enforce_effect_constraints or self._group_size() <= self.max_order:
            return W

        repaired = W.copy()
        gap_min = 1e-6
        repair_gap = gap_min * 1.01 + 1e-9
        for j in range(W.shape[1]):
            cross_groups = [
                groups[gi]
                for gi in range(1, len(groups))
                if gi != j + 1 and len(groups[gi]) > 0
            ]
            if not cross_groups:
                continue
            offset_col = int(cross_groups[0][0])
            offset_scale = float(Xv[0, offset_col])
            if abs(offset_scale) < 1e-12:
                continue

            cross_cols = np.concatenate(cross_groups)
            cross_sum = Xv[:, cross_cols] @ repaired[cross_cols, j]
            above_shift = -(float(np.max(cross_sum)) + repair_gap)
            below_shift = repair_gap - float(np.min(cross_sum))
            if abs(above_shift) <= abs(below_shift):
                shift = above_shift
            else:
                shift = below_shift
            repaired[offset_col, j] += shift / offset_scale
        return repaired

    def _select_lasso_cross_support(
        self,
        basis: np.ndarray,
        y_adj: np.ndarray,
        groups: list[np.ndarray],
        target_idx: int,
    ) -> list[int]:
        """Select the cross-source support with Lasso, then refine it with CVXPY."""
        from sklearn.linear_model import Lasso

        candidate_group_ids = [
            gi for gi in range(1, len(groups)) if gi != target_idx + 1
        ]
        if not candidate_group_ids:
            return []

        cross_cols = np.concatenate([groups[gi] - 1 for gi in candidate_group_ids])
        X_cross = basis[:, cross_cols].astype(float, copy=True)
        col_norms = np.linalg.norm(X_cross, axis=0)
        keep_cols = col_norms > 1e-12
        if not np.any(keep_cols):
            return []

        # Safe scaling: a near-zero column norm would divide to Inf and make
        # np.geomspace(Inf, Inf, ...) produce NaN for Lasso.
        safe_norms = np.maximum(col_norms[keep_cols], 1e-8)
        X_scaled = X_cross[:, keep_cols] / safe_norms
        residual_centered = y_adj - float(np.mean(y_adj))
        n_samples = max(X_scaled.shape[0], 1)
        alpha_max = float(np.max(np.abs(X_scaled.T @ residual_centered))) / n_samples
        if not np.isfinite(alpha_max) or alpha_max <= 1e-12:
            return []

        selected_local: np.ndarray | None = None
        last_coef: np.ndarray | None = None
        for alpha in np.geomspace(alpha_max, alpha_max * 1e-4, 30):
            lasso = Lasso(
                alpha=float(alpha),
                fit_intercept=False,
                max_iter=20000,
                tol=1e-6,
                selection="cyclic",
            )
            lasso.fit(X_scaled, residual_centered)
            last_coef = np.asarray(lasso.coef_, dtype=float)
            active = np.flatnonzero(np.abs(lasso.coef_) > 1e-8)
            if active.size > 0:
                selected_local = active
                break

        if selected_local is None:
            group_scores: list[tuple[int, float]] = []
            for gi in candidate_group_ids:
                cols = groups[gi] - 1
                score = float(np.linalg.norm(basis[:, cols].T @ residual_centered))
                group_scores.append((gi, score))
            group_scores.sort(key=lambda item: item[1], reverse=True)
            if not group_scores or group_scores[0][1] <= 1e-12:
                return []
            fallback_n = 1 if self.max_interactions <= 0 else int(self.max_interactions)
            return [gi for gi, _ in group_scores[:fallback_n]]

        kept_global_cols = cross_cols[keep_cols]
        selected_cols = set(kept_global_cols[selected_local].tolist())
        selected_groups = [
            gi
            for gi in candidate_group_ids
            if any(int(col - 1) in selected_cols for col in groups[gi])
        ]

        if self.max_interactions > 0 and len(selected_groups) > self.max_interactions:
            group_scores: list[tuple[int, float]] = []
            for gi in selected_groups:
                cols = groups[gi] - 1
                score = float(np.linalg.norm(basis[:, cols].T @ residual_centered))
                group_scores.append((gi, score))
            group_scores.sort(key=lambda item: item[1], reverse=True)
            selected_groups = [
                gi for gi, _ in group_scores[: int(self.max_interactions)]
            ]

        return selected_groups

    def _fit_one_order(
        self,
        power_function_sample_df: pd.DataFrame,
        quasi_dynamic_df: pd.DataFrame,
        intercept_values: np.ndarray | None,
    ) -> dict:
        """TIGER-style per-target constrained effect decomposition at the current max_order."""
        try:
            import cvxpy as cp
        except ImportError as exc:
            raise RuntimeError("TIGER-style constrained decomposition requires cvxpy") from exc

        X = self._design(power_function_sample_df)
        Y = align_response_to_design(quasi_dynamic_df, X.index).values.astype(float)
        Xv_raw = X.values.astype(float)
        basis = Xv_raw[:, 1:]
        n, p = Xv_raw.shape
        n_targets = Y.shape[1]
        target_names = [str(c) for c in quasi_dynamic_df.columns]
        group_size = self._group_size()
        n_vars = (p - 1) // group_size
        groups = _build_groups(n_vars, group_size)
        fix_intercept = intercept_values is not None
        curve_initial = align_response_to_design(
            power_function_sample_df.reindex(columns=quasi_dynamic_df.columns),
            X.index,
        ).values.astype(float)[0, :]
        y0_arr = (
            np.asarray(intercept_values, dtype=float)
            if fix_intercept
            else curve_initial
        )

        W = np.zeros((p, n_targets), dtype=float)
        W[0, :] = y0_arr
        alpha_rows: list[dict[str, float | int | bool]] = []
        ridge = 1e-6
        l1_scale = max(float(self.alpha), 1e-6) * 1e-4
        gap_min = 1e-6
        mean_gap_min = gap_min
        target_diagnostics: list[dict[str, float | str | bool]] = []
        target_directions: dict[str, str] = {}

        for j, target_name in enumerate(target_names):
            self_cols = groups[j + 1] - 1
            y_adj = Y[:, j] - y0_arr[j]
            cross_group_ids = self._select_lasso_cross_support(
                basis,
                y_adj,
                groups,
                j,
            )
            cross_cols = (
                np.concatenate([groups[gi] - 1 for gi in cross_group_ids])
                if cross_group_ids
                else np.array([], dtype=int)
            )
            active_cols = np.concatenate([self_cols, cross_cols])
            B_active = basis[:, active_cols]
            self_pos = np.arange(len(self_cols))
            cross_pos = np.arange(len(self_cols), len(active_cols))

            direction_options = ["self_above_total", "self_below_total"]
            best_obj = np.inf
            best_theta: np.ndarray | None = None
            best_direction = "invalid"
            best_diag: dict[str, float | str | bool] | None = None

            for direction in direction_options:
                theta = cp.Variable(len(active_cols))
                total_dynamic = B_active @ theta
                constraints = []
                if len(cross_pos) > 0 and self.enforce_effect_constraints:
                    cross_dynamic = B_active[:, cross_pos] @ theta[cross_pos]
                    constrained_cross = (
                        cross_dynamic[1:] if n > 1 else cross_dynamic
                    )
                    if direction == "self_above_total":
                        constraints.extend(
                            [
                                constrained_cross <= -gap_min,
                                cp.sum(-constrained_cross)
                                / max(n - 1, 1)
                                >= mean_gap_min,
                            ]
                        )
                    else:
                        constraints.extend(
                            [
                                constrained_cross >= gap_min,
                                cp.sum(constrained_cross)
                                / max(n - 1, 1)
                                >= mean_gap_min,
                            ]
                        )
                elif self.enforce_effect_constraints:
                    # Unreachable when cross_pos is non-empty (caught above); when it is
                    # empty there is nothing to constrain.
                    pass
                if self.nonneg_self:
                    self_dynamic_expr = B_active[:, self_pos] @ theta[self_pos]
                    constraints.append(y0_arr[j] + self_dynamic_expr >= 0.0)

                reg_terms = [ridge * cp.sum_squares(theta)]
                if len(cross_pos) > 0:
                    reg_terms.append(
                        l1_scale
                        * self.cross_weight_ratio
                        * cp.norm1(theta[cross_pos])
                    )
                objective = cp.Minimize(
                    cp.sum_squares(y_adj - total_dynamic) + sum(reg_terms)
                )
                problem = cp.Problem(objective, constraints)
                try:
                    problem.solve(solver=cp.CLARABEL, verbose=False)
                except cp.error.SolverError:
                    problem.solve(solver=cp.SCS, verbose=False)
                if theta.value is None or problem.status not in {
                    cp.OPTIMAL,
                    cp.OPTIMAL_INACCURATE,
                }:
                    continue

                theta_val = np.asarray(theta.value, dtype=float).reshape(-1)
                candidate_W = np.zeros((p, 1), dtype=float)
                candidate_W[0, 0] = y0_arr[j]
                candidate_W[1 + active_cols, 0] = theta_val
                valid, _, diag_rows = self._check_effect_constraints(
                    Xv_raw,
                    candidate_W,
                    groups,
                    [target_name],
                    [j],
                )
                if self.enforce_effect_constraints and not valid:
                    continue
                pred = Xv_raw @ candidate_W[:, 0]
                obj_val = float(np.sum((Y[:, j] - pred) ** 2))
                if obj_val < best_obj:
                    best_obj = obj_val
                    best_theta = candidate_W[:, 0]
                    best_direction = direction
                    best_diag = diag_rows[0]

            if best_theta is None or best_diag is None:
                raise ValueError(
                    f"No feasible TIGER decomposition for target {target_name!r}. "
                    "Please lower thresholds or increase max_order/max_interactions."
                )
            W[:, j] = best_theta
            target_directions[target_name] = best_direction
            target_diagnostics.append(best_diag)

        rss_final = float(np.sum((Y - Xv_raw @ W) ** 2))
        df_final = int(np.sum(np.abs(W) > 1e-10))
        rss_floor = 1e-12 * n * n_targets
        bic_final = n * n_targets * np.log(
            max(rss_final, rss_floor) / max(n * n_targets, 1)
        ) + df_final * np.log(n)

        return {
            "X_columns": X.columns,
            "Y_columns": quasi_dynamic_df.columns,
            "W": W,
            "Xv_raw": Xv_raw,
            "Y": Y,
            "rss": rss_final,
            "df": df_final,
            "bic": bic_final,
            "alpha": self.alpha,
            "mix": self.mix,
            "alpha_rows": alpha_rows,
            "n": n,
            "n_targets": n_targets,
            "effect_constraint_directions": target_directions,
            "effect_constraint_diagnostics": target_diagnostics,
        }

    def _fit_asgl_bic(
        self,
        power_function_sample_df: pd.DataFrame,
        quasi_dynamic_df: pd.DataFrame,
        intercept_values: np.ndarray | None = None,
    ) -> "IDOPRegressor":
        """Single-stage self-dominant ASGL with an outer BIC search over max_order."""
        user_max_order = self.max_order
        upper = min(user_max_order, 20)
        lower = 1
        if upper < lower:
            upper = lower

        order_rows: list[dict[str, float | int | bool]] = []
        alpha_rows_all: list[dict[str, float | int | bool]] = []
        best_artifacts: dict | None = None
        best_order_bic = np.inf
        best_order = lower
        infeasible_messages: list[str] = []

        for order_c in range(lower, upper + 1):
            self.max_order = order_c
            try:
                res = self._fit_one_order(
                    power_function_sample_df, quasi_dynamic_df, intercept_values
                )
            except (np.linalg.LinAlgError, ValueError) as exc:
                infeasible_messages.append(f"max_order={order_c}: {exc}")
                order_rows.append(
                    {
                        "max_order": order_c,
                        "bic": np.nan,
                        "rss": np.nan,
                        "df": np.nan,
                        "n_obs": np.nan,
                        "n_targets": np.nan,
                    }
                )
                continue

            order_rows.append(
                {
                    "max_order": order_c,
                    "bic": float(res["bic"]),
                    "rss": float(res["rss"]),
                    "df": int(res["df"]),
                    "n_obs": int(res["n"]),
                    "n_targets": int(res["n_targets"]),
                }
            )
            for row in res["alpha_rows"]:
                row_with_order = dict(row)
                row_with_order["max_order"] = order_c
                alpha_rows_all.append(row_with_order)

            if res["bic"] < best_order_bic:
                best_order_bic = res["bic"]
                best_order = order_c
                best_artifacts = res

        if best_artifacts is None:
            detail = "; ".join(infeasible_messages[-3:])
            raise ValueError(
                "No feasible effect-decomposition solution satisfies the hard "
                "constraints. Please lower thresholds or increase max_order."
                + (f" Recent failures: {detail}" if detail else "")
            )

        self.max_order = best_order
        self.alpha = best_artifacts["alpha"]
        self.mix = best_artifacts["mix"]

        self.bic_order_path_ = pd.DataFrame(order_rows)
        if not self.bic_order_path_.empty:
            self.bic_order_path_["selected"] = (
                self.bic_order_path_["max_order"] == self.max_order
            )
        self.bic_alpha_path_ = pd.DataFrame(alpha_rows_all)
        if not self.bic_alpha_path_.empty:
            self.bic_alpha_path_["selected"] = (
                (self.bic_alpha_path_["max_order"] == self.max_order)
                & (self.bic_alpha_path_["alpha"] == self.alpha)
                & (self.bic_alpha_path_["mix"] == self.mix)
            )

        W = best_artifacts["W"]
        self.coef_ = pd.DataFrame(
            W,
            index=best_artifacts["X_columns"],
            columns=best_artifacts["Y_columns"],
        )
        self.mse_ = float(
            np.mean((best_artifacts["Y"] - best_artifacts["Xv_raw"] @ W) ** 2)
        )
        self.effect_constraint_directions_ = pd.Series(
            best_artifacts["effect_constraint_directions"],
            name="direction",
        )
        self.effect_constraint_diagnostics_ = pd.DataFrame(
            best_artifacts["effect_constraint_diagnostics"]
        )
        return self

    def fit(
        self,
        power_function_sample_df: pd.DataFrame,
        quasi_dynamic_df: pd.DataFrame,
        *,
        power_function_params: pd.DataFrame,
        intercept_values: np.ndarray | None = None,
    ) -> "IDOPRegressor":
        self.bic_order_path_ = None
        self.bic_alpha_path_ = None
        self.effect_constraint_directions_ = None
        self.effect_constraint_diagnostics_ = None
        self.power_function_params_ = power_function_params
        return self._fit_asgl_bic(
            power_function_sample_df, quasi_dynamic_df, intercept_values
        )

    def predict(self, power_function_sample_df: pd.DataFrame) -> pd.DataFrame:
        X = self._design(power_function_sample_df)
        return pd.DataFrame(
            X.values.astype(float) @ self.coef_.values,
            index=X.index,
            columns=self.coef_.columns,
        )

    def effect(self, power_function_sample_df: pd.DataFrame) -> list[pd.DataFrame]:
        """Aggregate integrated basis x coefficients into per-source effects (no intercept)."""
        if self.coef_ is None:
            raise RuntimeError("call fit before effect")
        X = self._design(power_function_sample_df)
        basis_int_df = X.drop(columns=["intercept"])
        n_feature = power_function_sample_df.shape[1]
        group_size = self._group_size()
        feature_names = list(power_function_sample_df.columns)
        effect_df_list: list[pd.DataFrame] = []
        for target in self.coef_.columns:
            coef_row = self.coef_.loc[basis_int_df.columns, target]
            weighted = basis_int_df.multiply(coef_row, axis=1)
            cols: list[pd.Series] = []
            for j in range(n_feature):
                cols.append(
                    weighted.iloc[:, j * group_size : (j + 1) * group_size].sum(
                        axis=1
                    )
                )
            collapsed = pd.concat(cols, axis=1)
            collapsed.columns = feature_names
            effect_df_list.append(collapsed)
        return effect_df_list

    def adjacency_matrix(
        self,
        power_function_sample_df: pd.DataFrame,
        aggregation: str = "mean",
    ) -> pd.DataFrame:
        """
        Adjacency matrix, aggregating basis columns by mean or by integral.

        ``power_function_sample_df`` is the sampled power-function matrix.
        ``aggregation`` is either ``"mean"`` (arithmetic mean over discrete samples) or
        ``"integral"`` (trapezoid integral treating the index as the independent
        variable).
        """
        if self.coef_ is None:
            raise RuntimeError("call fit before adjacency_matrix")
        if aggregation not in ("mean", "integral"):
            raise ValueError("aggregation must be 'mean' or 'integral'")
        X = self._design(power_function_sample_df)
        basis_int_df = X.drop(columns=["intercept"])
        m = power_function_sample_df.shape[1]
        group_size = self._group_size()
        targets = list(self.coef_.columns)
        names = list(power_function_sample_df.columns)
        x_axis = basis_int_df.index.to_numpy(dtype=float, copy=False)
        G = np.zeros((m, m), dtype=float)
        for r in range(group_size):
            theta_r = np.zeros((m, m), dtype=float)
            psi = np.zeros(m, dtype=float)
            for k in range(m):
                col_idx = k * group_size + r
                col_name = basis_int_df.columns[col_idx]
                theta_r[:, k] = self.coef_.loc[col_name, targets].values.astype(float)
                col_values = basis_int_df.iloc[:, col_idx].to_numpy(dtype=float, copy=False)
                if aggregation == "integral":
                    psi[k] = float(np.trapezoid(col_values, x=x_axis))
                else:
                    psi[k] = float(np.mean(col_values))
            G += theta_r * psi[np.newaxis, :]
        return pd.DataFrame(G.T, index=names, columns=targets)




from typing import Any


from NetworkReconstructor import IDOPRegressor

__all__ = [
    "StaticIDOPRegressor",
    "fit_static_idop_network",
    "select_edges_lasso",
]


def select_edges_lasso(
    data_ecg: pd.DataFrame,
    alpha: float = 0.1,
    k: int = 10,
    threshold: float = 0.9,
    standardize_y: bool = True,
) -> pd.DataFrame:
    """
    Multi-window LASSO: recover a sparse directed support by cross-window frequency.

    The samples are split into ``k`` windows and a z-scored LASSO is fitted for each
    target in each window. After the coefficients are mapped back to the original
    scale, a source enters the target support only if it is non-zero in more than
    ``threshold`` of the windows. ``standardize_y=True`` (the static idop default)
    puts the LASSO objective on an O(1) scale; ``False`` matches
    ``idopECG.edge_select``, which scales ``alpha`` by the y magnitude and therefore
    changes the selected support, so the dynamic pipeline must pass ``False``.
    Returns a ``["target", "source"]`` frame whose ``source`` looks like
    ``"{a,b}"``.
    """
    lead_columns = [column for column in data_ecg.columns if column != "time"]
    if len(lead_columns) < 2:
        raise ValueError("data_ecg must contain at least two lead columns")
    if alpha <= 0:
        raise ValueError("alpha must be greater than 0")
    if k < 1:
        raise ValueError("k must be at least 1")
    if k > len(data_ecg):
        raise ValueError("k cannot exceed the number of time points")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")

    windows = np.array_split(data_ecg[lead_columns].to_numpy(dtype=float), k, axis=0)
    edge_counts = {
        target: {source: 0 for source in lead_columns if source != target}
        for target in lead_columns
    }

    for window in windows:
        for target_idx, target in enumerate(lead_columns):
            source_indices = [i for i in range(len(lead_columns)) if i != target_idx]
            x = window[:, source_indices]
            y = window[:, target_idx]

            # Column-wise standardisation of x puts the LASSO objective on an O(1)
            x_mean = x.mean(axis=0)
            x_scale = x.std(axis=0)
            x_scale[x_scale == 0] = 1.0
            x_std = (x - x_mean) / x_scale

            if standardize_y:
                y_mean = float(y.mean())
                y_scale = float(y.std())
                if y_scale == 0.0:
                    y_scale = 1.0
                y_target = (y - y_mean) / y_scale
            else:
                # Do not standardise y: matches idopECG.edge_select.
                y_scale = 1.0
                y_target = y

            model = Lasso(
                alpha=alpha, fit_intercept=True, max_iter=100_000, tol=1e-5,
            )
            model.fit(x_std, y_target)

            # Map the coefficients back: beta_orig = beta_std * y_scale / x_scale.
            coefficients = model.coef_ * (y_scale / x_scale)
            for local_idx, coefficient in enumerate(coefficients):
                if coefficient != 0.0:
                    source = lead_columns[source_indices[local_idx]]
                    edge_counts[target][source] += 1

    rows = []
    for target in lead_columns:
        selected = [
            source
            for source in lead_columns
            if source != target and edge_counts[target][source] / k > threshold
        ]
        rows.append({"target": target, "source": "{" + ",".join(selected) + "}"})

    return pd.DataFrame(rows, columns=["target", "source"])


def _parse_support_sets(
    edge_supports: pd.DataFrame,
    features: list[str],
) -> dict[str, set[str]]:
    """``{"a,b"}`` -> ``{target: {source, ...}}``, keeping known features only."""
    support_sets: dict[str, set[str]] = {}
    for _, row in edge_supports.iterrows():
        target = str(row["target"])
        inner = str(row["source"]).strip().strip("{}")
        support_sets[target] = {
            s.strip() for s in inner.split(",") if s.strip() and s.strip() in features
        }
    return support_sets


class StaticIDOPRegressor(IDOPRegressor):
    """
    The same network-fitting pipeline as :class:`IDOPRegressor`.

    Only the cross-edge selector is replaced (``_select_lasso_cross_support`` uses
    :func:`select_edges_lasso`, a multi-window LASSO with a frequency threshold on
    the raw data); the basis functions, cvxpy constraint solve, effect-constraint
    checks and ``max_order`` BIC grid are inherited unchanged.
    """

    def __init__(
        self,
        *,
        lasso_alpha: float = 0.1,
        lasso_windows: int = 1,
        lasso_threshold: float = 0.5,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.lasso_alpha = float(lasso_alpha)
        self.lasso_windows = int(lasso_windows)
        self.lasso_threshold = float(lasso_threshold)
        self.edge_supports_: pd.DataFrame | None = None
        self._support_names_: dict[str, set[str]] = {}
        self._feature_names_: list[str] = []
        self._target_names_: list[str] = []

    def fit(
        self,
        power_function_sample_df: pd.DataFrame,
        quasi_dynamic_df: pd.DataFrame,
        *,
        power_function_params: pd.DataFrame,
        intercept_values: np.ndarray | None = None,
    ) -> "StaticIDOPRegressor":
        """Run the new edge selector, then the parent-class fitting pipeline."""
        features = [str(c) for c in power_function_sample_df.columns]
        targets = [str(c) for c in quasi_dynamic_df.columns]

        self._feature_names_ = features
        self._target_names_ = targets
        self.edge_supports_ = select_edges_lasso(
            quasi_dynamic_df,
            alpha=self.lasso_alpha,
            k=self.lasso_windows,
            threshold=self.lasso_threshold,
        )
        self._support_names_ = _parse_support_sets(self.edge_supports_, features)

        return super().fit(
            power_function_sample_df,
            quasi_dynamic_df,
            power_function_params=power_function_params,
            intercept_values=intercept_values,
        )

    def _select_lasso_cross_support(
        self,
        basis: np.ndarray,
        y_adj: np.ndarray,
        groups: list[np.ndarray],
        target_idx: int,
    ) -> list[int]:
        """
        Replace the parent-class alpha-path LASSO by the new edge selector.

        ``groups[k + 1]`` holds feature ``k`` (``groups[0]`` is the intercept), so the
        group index of a feature is its column index plus one.
        """
        if target_idx >= len(self._target_names_):
            return []
        target = self._target_names_[target_idx]
        wanted = self._support_names_.get(target, set())
        if not wanted:
            return []

        selected = [
            index + 1
            for index, name in enumerate(self._feature_names_)
            if name != target and name in wanted
        ]
        if not selected:
            return []

        # Same Top-K rule as the parent: rank by correlation with the residual.
        if self.max_interactions > 0 and len(selected) > self.max_interactions:
            residual = y_adj - float(np.mean(y_adj))
            scored = [
                (gi, float(np.linalg.norm(basis[:, groups[gi] - 1].T @ residual)))
                for gi in selected
            ]
            scored.sort(key=lambda item: item[1], reverse=True)
            selected = [gi for gi, _ in scored[: int(self.max_interactions)]]

        return selected


def fit_static_idop_network(
    curve_sample_df: pd.DataFrame,
    response_df: pd.DataFrame,
    *,
    max_order: int,
    nonneg_self: bool,
    max_interactions: int,
    adjacency_aggregation: str,
    power_function_params: pd.DataFrame,
    lasso_alpha: float = 0.1,
    lasso_windows: int = 1,
    lasso_threshold: float = 0.5,
) -> dict:
    """
    Produce the network dictionary expected by ``_fit_idop_network_from_curve_sample``.

    The only difference from the repository route is the
    :class:`StaticIDOPRegressor` model; ``fit`` / ``predict`` / ``effect`` /
    ``adjacency_matrix`` / ``_design`` are unchanged, so the shape of the effect
    decomposition is identical.
    """
    if curve_sample_df.shape[1] < 2:
        raise ValueError("At least two curves are required to build an interaction network.")
    features = [str(c) for c in curve_sample_df.columns]
    response_df = response_df.loc[:, features]

    from NetworkReconstructor import align_response_to_design

    model = StaticIDOPRegressor(
        max_order=int(max_order),
        mix=0.5,
        fix_mix=False,
        nonneg_self=bool(nonneg_self),
        max_interactions=int(max_interactions),
        adaptive_weights=False,
        lasso_alpha=float(lasso_alpha),
        lasso_windows=int(lasso_windows),
        lasso_threshold=float(lasso_threshold),
    )
    model.fit(
        curve_sample_df,
        response_df,
        power_function_params=power_function_params,
    )

    design_X = model._design(curve_sample_df)
    return {
        "model": model,
        "quasi_dynamic_df": response_df,
        "curve_sample_df": curve_sample_df,
        "design_X": design_X,
        "response_Y": align_response_to_design(response_df, design_X.index),
        "predicted_df": model.predict(curve_sample_df),
        "effect_df_list": model.effect(curve_sample_df),
        "adj_df": model.adjacency_matrix(
            curve_sample_df, aggregation=str(adjacency_aggregation)
        ),
        "adjacency_aggregation": str(adjacency_aggregation),
    }


from typing import Any

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.special import eval_legendre

from NetworkReconstructor import select_edges_lasso

__all__ = [
    "DynamicIDOPRegressor",
    "fit_dynamic_idop_network",
    "observed_for_plot",
    "solve_dynamic_core",
]

# ECG sampling: default window 2.5 s at 100 Hz = 250 points.
DEFAULT_FS = 100.0
DEFAULT_WINDOW = 250
DEFAULT_N_FOURIER = 30
DEFAULT_R_LEGENDRE = 3



def _fourier_basis(t: np.ndarray, n_fourier: int, t_end: float) -> np.ndarray:
    """
    Fourier basis normalised to ``[-1, 1]`` with frequencies ``n / t_end`` Hz.

    Column 0 is the constant term ``0.5``; cos and sin columns alternate after it.
    """
    tau = 2.0 * t / t_end - 1.0
    phi = np.zeros((len(t), 2 * n_fourier + 1))
    phi[:, 0] = 0.5
    for n in range(1, n_fourier + 1):
        phi[:, 2 * n - 1] = np.cos(n * np.pi * tau)
        phi[:, 2 * n] = np.sin(n * np.pi * tau)
    return phi


def _legendre_basis(t: np.ndarray, r_max: int, t_end: float) -> np.ndarray:
    """``P_0 .. P_{r_max}`` evaluated at ``tau = 2t/t_end - 1``."""
    tau = 2.0 * t / t_end - 1.0
    return np.column_stack([eval_legendre(r, tau) for r in range(r_max + 1)])


def _integrate_columns(matrix: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Column-wise cumulative trapezoid integration (each column starts at 0)."""
    matrix_int = np.zeros_like(matrix, dtype=float)
    for col in range(matrix.shape[1]):
        matrix_int[:, col] = cumulative_trapezoid(
            matrix[:, col].astype(float), t, initial=0.0
        )
    return matrix_int


def _r2_score(y: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination ``R^2 = 1 - SS_res / SS_tot``."""
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _solve_window_two_stage(
    y: np.ndarray,
    psi_self: np.ndarray,
    cross_cols: list[np.ndarray],
) -> dict:
    """
    Two-stage orthogonalised fit for a single window.

    Stage 1 fits ``beta_self = lstsq(Psi_self, y)`` and forms the residual
    ``r = y - Psi_self beta_self``; stage 2 projects the cross columns onto the
    orthogonal complement of ``Psi_self`` (via ``Q = QR(Psi_self)``) and fits
    ``beta_cross = lstsq(Psi_cross_perp, r)``.
    """
    beta_self, *_ = np.linalg.lstsq(psi_self, y, rcond=None)
    y_self = psi_self @ beta_self
    residual = y - y_self

    if cross_cols:
        psi_cross = np.hstack(cross_cols)
        q_self, _ = np.linalg.qr(psi_self)
        psi_cross_perp = psi_cross - q_self @ (q_self.T @ psi_cross)
        beta_cross, *_ = np.linalg.lstsq(psi_cross_perp, residual, rcond=None)
    else:
        psi_cross_perp = np.zeros((len(y), 0))
        beta_cross = np.zeros(0)

    y_cross_list, offset = [], 0
    for block in cross_cols:
        n_cols = block.shape[1]
        y_cross_list.append(
            psi_cross_perp[:, offset:offset + n_cols]
            @ beta_cross[offset:offset + n_cols]
        )
        offset += n_cols

    y_pred = y_self + (sum(y_cross_list) if y_cross_list else 0.0)
    return {
        "y_self": y_self,
        "y_cross_list": y_cross_list,
        "y_pred": y_pred,
        "beta_self": np.asarray(beta_self, dtype=float),
        "r2": _r2_score(y, y_pred),
    }


def _parse_sources(source_str: Any) -> list[str]:
    """``'{II,aVR}'`` -> ``['II', 'aVR']``."""
    if not isinstance(source_str, str):
        return []
    return [s.strip() for s in source_str.strip("{}").split(",") if s.strip()]



def solve_dynamic_core(
    data_fit: pd.DataFrame,
    edge_supports: pd.DataFrame,
    *,
    n_fourier: int = DEFAULT_N_FOURIER,
    r_legendre: int = DEFAULT_R_LEGENDRE,
    window: int = DEFAULT_WINDOW,
    fs: float = DEFAULT_FS,
    auto_adapt: bool = True,
) -> dict:
    """
    ODE solver core for dynamic data (pure computation, no I/O or plotting).

    ``data_fit`` is the denoised waveform, ``(n_timepoints, n_channels)``;
    ``edge_supports`` is the edge-selector output with columns
    ``["target", "source"]`` where ``source`` looks like ``"{II,aVR}"``.
    ``n_fourier`` / ``r_legendre`` are the self-effect Fourier order and the
    cross-effect Legendre order; ``window`` / ``fs`` give the window length in
    samples and the sampling rate (window duration ``window / fs`` seconds).
    ``auto_adapt`` shrinks the window or the Fourier order instead of raising when
    the data are too short, recording the reason in the returned ``notes``. The
    returned dict holds ``leads``, ``support_sets``, ``t_win``, ``t_end``,
    ``n_windows``, ``n_edges``, the per-window arrays, the cross-window averages,
    ``edge_rows``, ``r2_mean`` and the effective ``window`` / ``n_fourier`` /
    ``adapted`` / ``notes``.
    """
    leads = [str(c) for c in data_fit.columns]
    x_all = data_fit[leads].to_numpy(dtype=float)
    n_samples = x_all.shape[0]

    requested_window = int(window)
    requested_n_fourier = int(n_fourier)
    if requested_window < 2:
        raise ValueError("window must be at least 2.")
    if n_samples < 2:
        raise ValueError("fewer than two time points; nothing to solve.")

    notes: list[str] = []

    # 1) the window cannot exceed the series length
    if n_samples < requested_window:
        if not auto_adapt:
            raise ValueError(
                f"{n_samples} time points are fewer than one window of {requested_window}; "
            f"windowing is impossible.",
            )
        window = max(2, n_samples)
        notes.append(
            f"Only {n_samples} time points, fewer than the requested window "
            f"{requested_window}; automatically reduced to {window} (a single window). "
            "If this is unexpected, check that the curve-fitting page is in **Dynamic** "
            "mode and that a full time series (not power-law fit samples) was uploaded.",

        )
    else:
        window = requested_window

    # 2) 2N+1 Fourier columns must stay below half the window or least squares
    max_fourier = max(1, (window // 2 - 1) // 2)
    if requested_n_fourier > max_fourier:
        n_fourier = max_fourier
        notes.append(
            f"Fourier order {requested_n_fourier} is too large for a window of "
            f"{window} (the basis would exceed half the window); reduced to {n_fourier}.",

        )
    else:
        n_fourier = requested_n_fourier

    t_win = np.arange(window) / float(fs)
    t_end = window / float(fs)

    support_sets = {
        str(row["target"]): _parse_sources(row["source"])
        for _, row in edge_supports.iterrows()
    }
    n_edges = sum(len(v) for v in support_sets.values())

    # would be underdetermined; once the design columns reach the window length
    # the unregularised fit interpolates perfectly (R2 ~ 1) without meaning.
    max_support = max((len(v) for v in support_sets.values()), default=0)
    max_columns = (2 * n_fourier + 1) + max_support * (int(r_legendre) + 1)
    if max_columns >= window:
        notes.append(
            f"Unreliable result: the maximum design has {max_columns} columns "
            f"(self effect {2 * n_fourier + 1} columns + cross {max_support} sources "
            f"x {int(r_legendre) + 1} orders), which reaches or exceeds the window "
            f"length {window}. Unregularised least squares then interpolates perfectly "
            "(R2 ~ 1, MSE ~ 0) without any statistical meaning. Upload a full time "
            "series in Dynamic mode, or reduce the Fourier / Legendre orders.",

        )

    n_windows = n_samples // window
    windows = [x_all[i * window:(i + 1) * window] for i in range(n_windows)]

    # The basis is shared by all windows (the in-window grid is fixed).
    psi_self_int = _integrate_columns(_fourier_basis(t_win, n_fourier, t_end), t_win)
    legendre_t = _legendre_basis(t_win, r_legendre, t_end)

    window_self: dict[str, list[np.ndarray]] = {c: [] for c in leads}
    window_cross: dict[str, list[dict[str, np.ndarray]]] = {c: [] for c in leads}
    window_r2: dict[str, list[float]] = {c: [] for c in leads}
    window_beta_self: dict[str, list[np.ndarray]] = {c: [] for c in leads}
    window_obs: dict[str, list[np.ndarray]] = {c: [] for c in leads}
    window_x0: dict[str, list[float]] = {c: [] for c in leads}

    for x_w in windows:
        x0 = x_w[0]
        for j, lead_j in enumerate(leads):
            y = x_w[:, j] - x0[j]
            window_x0[lead_j].append(float(x0[j]))
            cross_cols, cross_names = [], []
            for k_name in support_sets.get(lead_j, []):
                if k_name in leads:
                    k_idx = leads.index(k_name)
                    cols = x_w[:, k_idx:k_idx + 1] * legendre_t
                    cross_cols.append(_integrate_columns(cols, t_win))
                    cross_names.append(k_name)
            result = _solve_window_two_stage(y, psi_self_int, cross_cols)
            window_self[lead_j].append(result["y_self"])
            window_cross[lead_j].append(
                {k: result["y_cross_list"][i] for i, k in enumerate(cross_names)}
            )
            window_r2[lead_j].append(result["r2"])
            window_beta_self[lead_j].append(result["beta_self"])
            window_obs[lead_j].append(y)

    # ---- aggregation: cross-window mean effect curves and edge weights ----
    avg_self: dict[str, np.ndarray] = {}
    avg_cross: dict[str, dict[str, np.ndarray]] = {c: {} for c in leads}
    avg_obs: dict[str, np.ndarray] = {}
    avg_x0: dict[str, float] = {}
    r2_mean: dict[str, float] = {}
    edge_rows: list[dict[str, Any]] = []

    for lead_j in leads:
        avg_self[lead_j] = np.stack(window_self[lead_j]).mean(axis=0)
        avg_obs[lead_j] = np.stack(window_obs[lead_j]).mean(axis=0)
        # Intercept = cross-window mean of x(0); effect curves are centred on it.
        avg_x0[lead_j] = float(np.mean(window_x0[lead_j]))
        r2_mean[lead_j] = float(np.mean(np.array(window_r2[lead_j])))

        for k_name in support_sets.get(lead_j, []):
            if k_name not in window_cross[lead_j][0]:
                continue
            curves = np.stack([d[k_name] for d in window_cross[lead_j]])
            avg_cross[lead_j][k_name] = curves.mean(axis=0)
            weight = float(curves.mean())
            edge_rows.append({
                "source": k_name,
                "target": lead_j,
                "weight": round(weight, 8),
                "sign": "positive" if weight >= 0.0 else "negative",
            })

    return {
        "leads": leads,
        "support_sets": support_sets,
        "t_win": t_win,
        "t_end": t_end,
        "n_windows": n_windows,
        "n_edges": n_edges,
        "window_self": window_self,
        "window_cross": window_cross,
        "window_r2": window_r2,
        "window_beta_self": window_beta_self,
        "window_x0": window_x0,
        "window_obs": window_obs,
        "avg_self": avg_self,
        "avg_cross": avg_cross,
        "avg_obs": avg_obs,
        "avg_x0": avg_x0,
        "edge_rows": edge_rows,
        "r2_mean": r2_mean,
        # Effective basis / window plus the auto-adaptation notes.
        "window": window,
        "n_fourier": n_fourier,
        "r_legendre": int(r_legendre),
        "adapted": bool(notes),
        "notes": notes,
    }



class DynamicIDOPRegressor:
    """
    Dynamic-data solver exposing the same downstream interface as :class:`IDOPRegressor`.

    The design input is the SWT-denoised waveform and the response comes from the
    same source, matching ``idopECG.solve_core``. Effect curves live on the window
    grid (``0 .. window/fs`` seconds) rather than on the full time axis. Of the
    parameters the export logic reads, ``max_order`` maps to the Legendre order and
    ``alpha`` to the LASSO strength; the rest are recorded for provenance.
    """

    def __init__(
        self,
        *,
        n_fourier: int = DEFAULT_N_FOURIER,
        r_legendre: int = DEFAULT_R_LEGENDRE,
        window: int = DEFAULT_WINDOW,
        fs: float = DEFAULT_FS,
        lasso_alpha: float = 0.01,
        lasso_windows: int = 10,
        lasso_threshold: float = 0.4,
        mix: float = 0.5,
        nonneg_self: bool = True,
        max_interactions: int = 0,
        basis_type: str = "fourier_legendre",
        ebic_gamma: float = 0.0,
        enforce_effect_constraints: bool = False,
        window_index: int | None = None,
    ):
        self.n_fourier = int(n_fourier)
        self.r_legendre = int(r_legendre)
        self.window = int(window)
        self.fs = float(fs)
        self.lasso_alpha = float(lasso_alpha)
        self.lasso_windows = int(lasso_windows)
        self.lasso_threshold = float(lasso_threshold)
        # Plotting window; None selects the window with the largest sum of R2.
        self.window_index = None if window_index is None else int(window_index)

        # Compatibility attributes read by the export / diagnostics path.
        self.max_order = int(r_legendre)
        self.alpha = float(lasso_alpha)
        self.mix = float(mix)
        self.nonneg_self = bool(nonneg_self)
        self.max_interactions = int(max_interactions)
        self.basis_type = str(basis_type)
        self.ebic_gamma = float(ebic_gamma)
        self.enforce_effect_constraints = bool(enforce_effect_constraints)

        self.coef_: pd.DataFrame | None = None
        self.mse_: float | None = None
        self.edge_supports_: pd.DataFrame | None = None
        self._core: dict | None = None
        self._intercepts: np.ndarray | None = None
        self.adaptation_notes_: list[str] = []
        # Plotting convention: the window with the largest sum of R2.
        self.plot_window_: int = 0
        self.n_windows_: int = 0
        self.window_r2_: dict[str, list[float]] = {}
        self.r2_mean_: dict[str, float] = {}

    def fit(
        self,
        power_function_sample_df: pd.DataFrame,
        quasi_dynamic_df: pd.DataFrame | None = None,
        *,
        power_function_params: pd.DataFrame | None = None,
    ) -> "DynamicIDOPRegressor":
        signal = power_function_sample_df.apply(pd.to_numeric, errors="coerce")
        leads = [str(c) for c in signal.columns]

        self.edge_supports_ = select_edges_lasso(
            signal,
            alpha=self.lasso_alpha,
            k=self.lasso_windows,
            threshold=self.lasso_threshold,
            standardize_y=False,  # matches idopECG.edge_select (y is not standardised)
        )
        core = solve_dynamic_core(
            signal,
            self.edge_supports_,
            n_fourier=self.n_fourier,
            r_legendre=self.r_legendre,
            window=self.window,
            fs=self.fs,
        )
        self._core = core
        # Effective parameters may differ from the requested ones after adaptation.
        self.n_fourier = int(core["n_fourier"])
        self.window = int(core["window"])
        self.adaptation_notes_ = list(core.get("notes", []))

        # ---- plotting convention ----
        # Observed, predicted and effect curves all use one window, by default the
        # one with the largest sum of R2: averaging across windows flattens the
        # waveform and looks like a poor fit. Edge weights stay cross-window averaged.
        self.n_windows_ = int(core["n_windows"])
        self.window_r2_ = {lead: list(core["window_r2"][lead]) for lead in leads}
        self.r2_mean_ = dict(core["r2_mean"])
        if self.window_index is None:
            scores = [
                float(sum(core["window_r2"][lead][iw] for lead in leads))
                for iw in range(self.n_windows_)
            ]
            self.plot_window_ = int(np.argmax(scores)) if scores else 0
        else:
            self.plot_window_ = int(
                min(max(self.window_index, 0), max(self.n_windows_ - 1, 0))
            )

        window_used = self.plot_window_
        # Intercept = x(0) of this window; effect curves are centred on it.
        self._intercepts = np.array(
            [float(core["window_x0"][lead][window_used]) for lead in leads]
        )

        # mse: observed vs predicted on the plotting window.
        errors = []
        for lead in leads:
            per_window = core["window_cross"][lead][window_used]
            prediction = core["window_self"][lead][window_used] + (
                sum(per_window.values()) if per_window else 0.0
            )
            errors.append(
                float(np.mean((core["window_obs"][lead][window_used] - prediction) ** 2))
            )
        self.mse_ = float(np.mean(errors)) if errors else float("nan")

        # coef_: intercept row (cross-window mean x(0)) plus window-mean self-effect
        # Fourier coefficients; cross effects are curves returned by effect().
        rows: dict[str, dict[str, float]] = {
            "intercept": {
                lead: float(self._intercepts[j]) for j, lead in enumerate(leads)
            }
        }
        beta_stack = {
            lead: np.stack(core["window_beta_self"][lead]).mean(axis=0)
            for lead in leads
        }
        n_basis = len(next(iter(beta_stack.values())))
        for i in range(n_basis):
            name = f"self::F{i}"
            rows[name] = {lead: float(beta_stack[lead][i]) for lead in leads}
        self.coef_ = pd.DataFrame(rows).T.reindex(columns=leads)
        return self

    def _design(self, power_function_sample_df: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        Design matrix on the window grid: intercept + integrated Fourier + Legendre bases.

        The cross columns are only a row reference here, because each target has its own
        support and the per-target design is built inside ``solve_dynamic_core``.
        """
        if self._core is None:
            raise RuntimeError("call fit before _design")
        core = self._core
        t_win, t_end = core["t_win"], core["t_end"]

        data: dict[str, np.ndarray] = {"intercept": np.ones(len(t_win))}
        self_basis = _integrate_columns(
            _fourier_basis(t_win, self.n_fourier, t_end), t_win
        )
        for i in range(self_basis.shape[1]):
            data[f"self::F{i}"] = self_basis[:, i]
        legendre = _legendre_basis(t_win, self.r_legendre, t_end)
        for i in range(legendre.shape[1]):
            data[f"cross::P{i}"] = legendre[:, i]
        return pd.DataFrame(data, index=pd.Index(t_win, name="time"))

    def predict(self, power_function_sample_df: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        Absolute prediction (``x(0) + self + cross``) on the window time grid.

        Uses the plotting window (:attr:`plot_window_`) so it shares window and scale
        with :meth:`observed` and :meth:`effect`.
        """
        if self._core is None:
            raise RuntimeError("call fit before predict")
        core = self._core
        leads = core["leads"]
        t_win = core["t_win"]
        iw = self.plot_window_
        columns = {}
        for j, lead in enumerate(leads):
            per_window = core["window_cross"][lead][iw]
            cross = sum(per_window.values()) if per_window else 0.0
            columns[lead] = self._intercepts[j] + core["window_self"][lead][iw] + cross
        return pd.DataFrame(columns, index=pd.Index(t_win, name="time"))

    def observed(self) -> pd.DataFrame:
        """
        Absolute observations of the single plotting window (``x(0) + centred data``).

        Same window, grid and scale as :meth:`predict` / :meth:`effect`, so the curves
        can be overlaid. A single window is used rather than a cross-window average
        because different windows have different beat phases and averaging flattens the
        waveform, which looks like a poor fit; ``idopECG._plot_ode_lead`` was changed
        for the same reason. Edge weights still use the cross-window average, which is
        the statistically stable estimate.
        """
        if self._core is None:
            raise RuntimeError("call fit before observed")
        core = self._core
        iw = self.plot_window_
        columns = {
            lead: core["window_x0"][lead][iw] + core["window_obs"][lead][iw]
            for lead in core["leads"]
        }
        return pd.DataFrame(columns, index=pd.Index(core["t_win"], name="time"))

    def effect(self, power_function_sample_df: pd.DataFrame | None = None) -> list[pd.DataFrame]:
        """
        One DataFrame per target: index = window grid, columns = all channels.

        The own column is the Fourier self effect, the others are Legendre cross effects
        and unselected sources are 0, matching the layout of ``IDOPRegressor.effect``
        and likewise reporting centred contributions (the intercept lives in
        ``coef_.loc["intercept"]``). Uses the plotting window.
        """
        if self._core is None:
            raise RuntimeError("call fit before effect")
        core = self._core
        leads = core["leads"]
        iw = self.plot_window_
        index = pd.Index(core["t_win"], name="time")

        effect_df_list: list[pd.DataFrame] = []
        for lead_j in leads:
            per_window = core["window_cross"][lead_j][iw]
            columns = {}
            for name in leads:
                if name == lead_j:
                    columns[name] = core["window_self"][lead_j][iw]
                elif name in per_window:
                    columns[name] = per_window[name]
                else:
                    columns[name] = np.zeros(len(core["t_win"]))
            effect_df_list.append(pd.DataFrame(columns, index=index))
        return effect_df_list

    def adjacency_matrix(
        self,
        power_function_sample_df: pd.DataFrame | None = None,
        aggregation: str = "mean",
    ) -> pd.DataFrame:
        """Adjacency matrix; ``adj.loc[source, target]`` is the source -> target weight."""
        if aggregation not in ("mean", "integral"):
            raise ValueError("aggregation must be 'mean' or 'integral'")
        if self._core is None:
            raise RuntimeError("call fit before adjacency_matrix")
        core = self._core
        leads = core["leads"]
        t_win = core["t_win"]

        matrix = np.zeros((len(leads), len(leads)), dtype=float)
        for target in leads:
            for source, curve in core["avg_cross"][target].items():
                if aggregation == "integral":
                    value = float(np.trapezoid(curve, x=t_win))
                else:
                    value = float(np.mean(curve))
                matrix[leads.index(source), leads.index(target)] = value
        return pd.DataFrame(matrix, index=leads, columns=leads)


def observed_for_plot(model: Any, fallback: pd.DataFrame) -> pd.DataFrame:
    """
    Observations on the same grid as the prediction, for plotting.

    Static predictions live on the design-matrix index and can be compared with the
    response directly, so ``fallback`` is returned unchanged. The dynamic route is
    different: its predictions and effects live on the window time grid (e.g. 250
    points over 0-2.5 s) while the response is the full series (e.g. 1000 points
    over 0-10 s), and plotting both would stretch the observations over the whole
    axis while the prediction occupies a small left-hand segment. In that case the
    model's own window-aligned absolute observations are returned.
    """
    observed = getattr(model, "observed", None)
    if callable(observed):
        return observed()
    return fallback


def fit_dynamic_idop_network(
    curve_sample_df: pd.DataFrame,
    response_df: pd.DataFrame | None = None,
    *,
    max_order: int = DEFAULT_R_LEGENDRE,
    nonneg_self: bool = True,
    max_interactions: int = 0,
    adjacency_aggregation: str = "mean",
    n_fourier: int = DEFAULT_N_FOURIER,
    window: int = DEFAULT_WINDOW,
    fs: float = DEFAULT_FS,
    lasso_alpha: float = 0.01,
    lasso_windows: int = 10,
    lasso_threshold: float = 0.4,
    power_function_params: pd.DataFrame | None = None,
) -> dict:
    """
    Produce the network dictionary expected by ``_fit_idop_network_from_curve_sample``.

    As in the static route it returns nine keys, so downstream export, effect
    decomposition and plotting need no branching. ``response_df`` is kept only for
    interface compatibility: the dynamic observations are the model's window-aligned
    absolute observations, because the full time series and the window-grid
    prediction do not share a horizontal axis.
    """
    from NetworkReconstructor import align_response_to_design

    model = DynamicIDOPRegressor(
        n_fourier=int(n_fourier),
        r_legendre=int(max_order),
        window=int(window),
        fs=float(fs),
        lasso_alpha=float(lasso_alpha),
        lasso_windows=int(lasso_windows),
        lasso_threshold=float(lasso_threshold),
        mix=0.5,
        nonneg_self=bool(nonneg_self),
        max_interactions=int(max_interactions),
    )
    model.fit(curve_sample_df)

    design_X = model._design()
    # The observations must share the window time grid with predict / effect,
    # otherwise the fit and effect figures are misaligned (a full series of
    # 1000 points against a 250-point window).
    response = model.observed()
    return {
        "model": model,
        "quasi_dynamic_df": response,
        "curve_sample_df": curve_sample_df,
        "design_X": design_X,
        "response_Y": align_response_to_design(response, design_X.index),
        "predicted_df": model.predict(),
        "effect_df_list": model.effect(),
        "adj_df": model.adjacency_matrix(aggregation=str(adjacency_aggregation)),
        "adjacency_aggregation": str(adjacency_aggregation),
    }




import hashlib
import json
import os
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


N_CENTER_HUBS = 3

LAYOUT_METRIC = "weighted_out_abs"


CENTER_HUB_RADIUS = 2.70

FIRST_NONHUB_RING_RADIUS = 6.50

SECOND_NONHUB_RING_RADIUS = 10.20


RING_GAP_BASE = 2.25

RING_GAP_GROWTH = 0.20


RING_CAPACITY_START = 8

RING_CAPACITY_STEP = 4

RING_CAPACITY_MAX = 40


RING_ROTATION_DEG = 23.0

EVEN_RING_HALF_STEP = True


LAYOUT_SEED = 20260903

ANGLE_JITTER_FRACTION = 0.004

RADIUS_JITTER = 0.0015


# Nodes

HUB_NODE_DIAMETER = 35.0

FIRST_RING_DIAMETER = 28.0

RING_DIAMETER_DECAY = 4.6

MIN_NODE_DIAMETER = 7.5


RING_COLOR_PALETTE = [

    "#D73027",
    "#F46D43",
    "#FDAE61",
    "#FEE08B",
    "#D9EF8B",
    "#A6D96A",
    "#66C2A5",
    "#74ADD1",
    "#4575B4",
    "#7B68EE",
    "#9E9AC8",

]


NODE_EDGE_COLOR = "#555555"

TEXT_COLOR = "#202020"


# Edges

POSITIVE_EDGE_COLOR = "#F08A5D"

NEGATIVE_EDGE_COLOR = "#4EA3F1"


TOPK_PER_TARGET = 8

TOPK_PER_SOURCE = 5

EDGE_QUANTILE = 0.60


MIN_DISPLAY_EDGES = 260

MAX_DISPLAY_EDGES = 700


EDGE_WIDTH_STRONG = 1.55

EDGE_WIDTH_MEDIUM = 0.95

EDGE_WIDTH_WEAK = 0.40


RECIPROCAL_CURVATURE = 0.085

CURVATURE_BASE = 0.018

CURVATURE_STEP = 0.008


# HTML display defaults

HTML_HEIGHT = 950

POSITION_SCALE = 78.0


DEFAULT_NODE_SCALE = 1.5

DEFAULT_LABEL_SCALE = 1.5

DEFAULT_EDGE_SCALE = 1.8

DEFAULT_EDGE_OPACITY = 1.0

# Edge highlight uses min(1.0, edgeOpacity + 0.30), so at the default 1.0 the
# edges are already opaque and hovering cannot brighten them; lower the
# default (e.g. 0.85) if visible hover feedback is needed.

DEFAULT_EDGE_PERCENTILE = 0



DEGREE_PANEL_HEIGHT = 950

IN_DEGREE_COLOR = "#4EA3F1"

OUT_DEGREE_COLOR = "#F08A5D"


# Effect figures are not embedded in the HTML body; they are written as
# PNG/PDF after fitting and offered as downloads.
SAVE_EFFECT_PNG = True

SAVE_EFFECT_PDF = True

EFFECT_DPI = 300


# Runtime module list
# MODULES is filled in by plot_network() from the node set of the input
# (source | target) instead of from the labels data of the original script.
#
#

MODULES = []


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D

font_prop = None  # Set by application layer for CJK font support


# 1. Utilities: adjacency-matrix validation and normalisation

def _normalize_adjacency_matrix(adj_df: pd.DataFrame) -> pd.DataFrame:
    """Normalise an adjacency matrix: string index/columns, same square order."""
    if not isinstance(adj_df, pd.DataFrame):
        raise TypeError("adj_df must be a pandas DataFrame.")

    if adj_df.shape[0] != adj_df.shape[1]:
        raise ValueError("adj_df must be a square matrix.")

    out = adj_df.copy()
    out.index = out.index.astype(str)
    out.columns = out.columns.astype(str)

    if set(out.index) != set(out.columns):
        raise ValueError("adj_df.index and adj_df.columns must contain the same node names.")

    # Keep the index order and align the columns to it.
    out = out.loc[list(out.index), list(out.index)]
    out = out.apply(pd.to_numeric, errors="raise")

    if out.isna().values.any():
        raise ValueError("adj_df contains missing values.")

    return out


def _degree_table(
    adj_df: pd.DataFrame,
    *,
    edge_threshold: float = 0.0,
    include_self_edges: bool = False,
    sort_by: str = "total_degree",
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Node in/out-degree table.

    Direction rule: ``adj_df.loc[source, target]`` is ``source -> target``, so the
    out-degree counts non-zeros per row and the in-degree counts non-zeros per
    column.
    """
    adj_df = _normalize_adjacency_matrix(adj_df)
    edge_arr = adj_df.to_numpy(dtype=float, copy=True)

    if not include_self_edges:
        np.fill_diagonal(edge_arr, 0.0)

    edge_mat = pd.DataFrame(
        edge_arr,
        index=adj_df.index,
        columns=adj_df.columns,
    )

    active = edge_mat.abs() > float(edge_threshold)

    outdegree = active.sum(axis=1).astype(int)
    indegree = active.sum(axis=0).astype(int)

    weighted_outdegree = edge_mat.abs().sum(axis=1)
    weighted_indegree = edge_mat.abs().sum(axis=0)

    degree_df = pd.DataFrame(
        {
            "node": list(adj_df.index),
            "outdegree": outdegree.reindex(adj_df.index).values,
            "indegree": indegree.reindex(adj_df.index).values,
            "total_degree": (
                outdegree.reindex(adj_df.index).values
                + indegree.reindex(adj_df.index).values
            ),
            "weighted_outdegree": weighted_outdegree.reindex(adj_df.index).values,
            "weighted_indegree": weighted_indegree.reindex(adj_df.index).values,
            "weighted_total_degree": (
                weighted_outdegree.reindex(adj_df.index).values
                + weighted_indegree.reindex(adj_df.index).values
            ),
            "node_weight": np.diag(adj_df.values),
        }
    )

    allowed = [
        "node",
        "outdegree",
        "indegree",
        "total_degree",
        "weighted_outdegree",
        "weighted_indegree",
        "weighted_total_degree",
        "node_weight",
    ]

    if sort_by not in allowed:
        raise ValueError(f"sort_by must be one of {allowed}")

    degree_df = degree_df.sort_values(
        sort_by,
        ascending=ascending,
    ).reset_index(drop=True)

    return degree_df


def _get_node_order(
    adj_df: pd.DataFrame,
    *,
    order_by: str = "outdegree",
    edge_threshold: float = 0.0,
    include_self_edges: bool = False,
) -> list[str]:
    """Node order derived from the node degrees."""
    adj_df = _normalize_adjacency_matrix(adj_df)

    if order_by == "original":
        return list(adj_df.index.astype(str))

    degree_df = _degree_table(
        adj_df,
        edge_threshold=edge_threshold,
        include_self_edges=include_self_edges,
        sort_by=order_by,
        ascending=False,
    )

    return degree_df["node"].astype(str).tolist()


def _select_hub_nodes(
    adj_df: pd.DataFrame,
    *,
    hub_n: int = 3,
    hub_by: str = "outdegree",
    edge_threshold: float = 0.0,
    include_self_edges: bool = False,
) -> list[str]:
    """Select the hub nodes."""
    adj_df = _normalize_adjacency_matrix(adj_df)

    hub_n = int(hub_n)
    if hub_n <= 0:
        return []

    degree_df = _degree_table(
        adj_df,
        edge_threshold=edge_threshold,
        include_self_edges=include_self_edges,
        sort_by=hub_by,
        ascending=False,
    )

    return degree_df["node"].astype(str).head(min(hub_n, len(degree_df))).tolist()


# 2. Network construction and layout

def _build_digraph(
    adj_df: pd.DataFrame,
    *,
    edge_threshold: float = 0.0,
    top_edges: int | None = None,
    include_self_edges: bool = False,
) -> nx.DiGraph:
    """
    Build the NetworkX directed graph from the adjacency matrix.

    Direction rule: ``adj_df.loc[source, target]`` is ``source -> target``.
    """
    adj_df = _normalize_adjacency_matrix(adj_df)

    G = nx.DiGraph()

    for node in adj_df.index:
        node_str = str(node)
        G.add_node(node_str, node_weight=float(adj_df.loc[node, node]))

    edges: list[tuple[str, str, float]] = []

    for source in adj_df.index:
        for target in adj_df.columns:
            if source == target and not include_self_edges:
                continue

            w = float(adj_df.loc[source, target])

            if abs(w) <= float(edge_threshold):
                continue

            edges.append((str(source), str(target), w))

    if top_edges is not None:
        top_edges = int(top_edges)
        if top_edges < 0:
            raise ValueError("top_edges must be non-negative or None.")
        edges = sorted(edges, key=lambda x: abs(x[2]), reverse=True)[:top_edges]

    for source, target, w in edges:
        G.add_edge(
            source,
            target,
            weight=w,
            abs_weight=abs(w),
            sign=1 if w > 0 else -1,
        )

    return G


def _fixed_circular_layout(
    node_order: list[str],
    *,
    radius: float = 1.0,
    start_angle: float = np.pi / 2,
    clockwise: bool = True,
) -> dict[str, np.ndarray]:
    """Fixed circular layout with the first node at 12 o'clock."""
    node_order = [str(x) for x in node_order]
    n = len(node_order)

    if n == 0:
        return {}

    pos: dict[str, np.ndarray] = {}
    direction = -1 if clockwise else 1

    for i, node in enumerate(node_order):
        theta = start_angle + direction * 2 * np.pi * i / n
        pos[node] = np.array([radius * np.cos(theta), radius * np.sin(theta)])

    return pos


def _hub_circular_layout(
    *,
    node_order: list[str],
    hub_nodes: list[str],
    layout_scale: float = 1.0,
    hub_inner_radius: float = 0.28,
    hub_outer_radius: float = 1.0,
    start_angle: float = np.pi / 2,
    clockwise: bool = True,
) -> dict[str, np.ndarray]:
    """Hub nodes in the centre, non-hub nodes on the surrounding ring."""
    node_order = [str(x) for x in node_order]
    hub_nodes = [str(x) for x in hub_nodes if str(x) in node_order]

    hub_set = set(hub_nodes)
    outer_nodes = [node for node in node_order if node not in hub_set]

    pos: dict[str, np.ndarray] = {}
    direction = -1 if clockwise else 1

    hub_count = len(hub_nodes)
    outer_count = len(outer_nodes)

    if hub_count == 1:
        pos[hub_nodes[0]] = np.array([0.0, 0.0])

    elif hub_count > 1:
        for i, node in enumerate(hub_nodes):
            theta = start_angle + direction * 2 * np.pi * i / hub_count
            pos[node] = (
                np.array(
                    [
                        hub_inner_radius * np.cos(theta),
                        hub_inner_radius * np.sin(theta),
                    ]
                )
                * layout_scale
            )

    if outer_count > 0:
        for i, node in enumerate(outer_nodes):
            theta = start_angle + direction * 2 * np.pi * i / outer_count
            pos[node] = (
                np.array(
                    [
                        hub_outer_radius * np.cos(theta),
                        hub_outer_radius * np.sin(theta),
                    ]
                )
                * layout_scale
            )

    return pos


def _get_layout(
    G: nx.DiGraph,
    *,
    layout: str = "hub_circular",
    seed: int = 123,
    layout_scale: float = 1.0,
    node_order: list[str] | None = None,
    hub_nodes: list[str] | None = None,
    hub_inner_radius: float = 0.28,
    hub_outer_radius: float = 1.0,
    start_angle: float = np.pi / 2,
    clockwise: bool = True,
) -> dict[str, np.ndarray]:
    """Build the network layout."""
    layout = str(layout).lower()

    nodes = [str(x) for x in G.nodes()]
    if not nodes:
        return {}

    if node_order is None:
        node_order = nodes
    else:
        node_order = [str(x) for x in node_order if str(x) in nodes]
        node_order += [x for x in nodes if x not in node_order]

    if hub_nodes is None:
        hub_nodes = []
    else:
        hub_nodes = [str(x) for x in hub_nodes if str(x) in nodes]

    if layout == "fixed_circular":
        return _fixed_circular_layout(
            node_order=node_order,
            radius=layout_scale,
            start_angle=start_angle,
            clockwise=clockwise,
        )

    if layout == "hub_circular":
        return _hub_circular_layout(
            node_order=node_order,
            hub_nodes=hub_nodes,
            layout_scale=layout_scale,
            hub_inner_radius=hub_inner_radius,
            hub_outer_radius=hub_outer_radius,
            start_angle=start_angle,
            clockwise=clockwise,
        )

    if layout == "circular":
        return nx.circular_layout(G, scale=layout_scale)

    if layout == "spring":
        return nx.spring_layout(G, seed=seed, scale=layout_scale)

    if layout == "shell":
        return nx.shell_layout(G, scale=layout_scale)

    if layout == "kamada_kawai":
        return nx.kamada_kawai_layout(G, scale=layout_scale)

    if layout == "spectral":
        return nx.spectral_layout(G, scale=layout_scale)

    raise ValueError(
        "layout must be one of: "
        "'hub_circular', 'fixed_circular', 'circular', 'spring', "
        "'shell', 'kamada_kawai', 'spectral'."
    )


# 3. Plotting helpers

def _compute_node_sizes(
    adj_df: pd.DataFrame,
    nodes: list[str],
    *,
    uniform_node_size: bool = True,
    node_size: float = 850,
    node_size_min: float = 450,
    node_size_max: float = 1800,
    node_size_scale: float = 1.0,
    edge_threshold: float = 0.0,
    include_self_edges: bool = False,
) -> dict[str, float]:
    """Node sizes; equal by default, optionally driven by the out-degree."""
    adj_df = _normalize_adjacency_matrix(adj_df)
    nodes = [str(x) for x in nodes]

    if uniform_node_size:
        return {node: float(node_size) * float(node_size_scale) for node in nodes}

    degree_df = _degree_table(
        adj_df,
        edge_threshold=edge_threshold,
        include_self_edges=include_self_edges,
        sort_by="node",
        ascending=True,
    ).set_index("node")

    out_values = np.array(
        [degree_df.loc[node, "outdegree"] if node in degree_df.index else 0 for node in nodes],
        dtype=float,
    )

    if len(out_values) > 0 and out_values.max() > out_values.min():
        sizes = node_size_min + (
            (out_values - out_values.min())
            / (out_values.max() - out_values.min())
        ) * (node_size_max - node_size_min)
    else:
        sizes = np.repeat(node_size, len(nodes))

    sizes = sizes * float(node_size_scale)

    return {node: float(size) for node, size in zip(nodes, sizes)}


def _draw_labels_with_font(
    ax: plt.Axes,
    pos: dict[str, np.ndarray],
    nodes: list[str],
    *,
    font_size: int = 9,
    font_color: str = "black",
    font_weight: str = "normal",
) -> None:
    """Draw node labels manually with font_prop to avoid CJK glyph problems."""
    for node in nodes:
        if node not in pos:
            continue
        x, y = pos[node]
        ax.text(
            x,
            y,
            str(node),
            ha="center",
            va="center",
            fontsize=font_size,
            color=font_color,
            fontweight=font_weight,
            fontproperties=font_prop,
            zorder=5,
        )

def _center_network_axis(
    ax: plt.Axes,
    pos: dict[str, np.ndarray],
    *,
    pad_ratio: float = 0.30,
) -> None:
    """Centre the network drawing in its axis and keep it fully visible."""
    if not pos:
        return

    xy = np.array(list(pos.values()), dtype=float)
    xs = xy[:, 0]
    ys = xy[:, 1]

    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()

    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0

    half_span = max((xmax - xmin) / 2.0, (ymax - ymin) / 2.0, 0.8)
    half_span = half_span * (1.0 + pad_ratio)

    ax.set_xlim(cx - half_span, cx + half_span)
    ax.set_ylim(cy - half_span, cy + half_span)

    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor("C")
    
def _draw_signed_edges(
    G: nx.DiGraph,
    pos: dict[str, np.ndarray],
    ax: plt.Axes,
    *,
    positive_edge_color: str = "#E74C3C",
    negative_edge_color: str = "#2F7ED8",
    arrowsize: int = 18,
    edge_alpha: float = 0.92,
    edge_width_min: float = 1.2,
    edge_width_max_add: float = 4.5,
    reciprocal_rad: float = 0.45,
    single_edge_rad: float = 0.08,
) -> None:
    """
    Draw the directed edges one by one.

    NetworkX ``arc3`` rad is relative to the edge direction, so giving A->B and B->A
    opposite signs would not separate them reliably; using the same rad for both
    directions is more robust.
    """
    edges = list(G.edges(data=True))
    if not edges:
        return

    max_abs_w = max(abs(float(d.get("weight", 1.0))) for _, _, d in edges)
    max_abs_w = max(max_abs_w, 1e-12)

    edge_set = {(str(u), str(v)) for u, v, _ in edges}

    for u, v, d in edges:
        u = str(u)
        v = str(v)

        w = float(d.get("weight", 1.0))
        aw = abs(w)

        width = edge_width_min + edge_width_max_add * aw / max_abs_w
        color = positive_edge_color if w >= 0 else negative_edge_color

        # Key fix:
        # when both directions exist, A->B and B->A use the same rad; opposite
        # edge directions then place the arcs on physically opposite sides.
        if u != v and (v, u) in edge_set:
            rad = float(reciprocal_rad)
        else:
            rad = float(single_edge_rad)

        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v)],
            width=width,
            edge_color=color,
            alpha=edge_alpha,
            arrows=True,
            arrowsize=arrowsize,
            arrowstyle="-|>",
            connectionstyle=f"arc3,rad={rad}",
            min_source_margin=26,
            min_target_margin=26,
            ax=ax,
        )


def _mirror_barh_degree(
    ax: plt.Axes,
    degree_df: pd.DataFrame,
    *,
    node_order: list[str],
    title: str = "",
    bg_color: str = "white",
    bar_height: float | None = None,
) -> None:
    """
    Mirrored horizontal bar chart: in-degree to the left, out-degree to the right.

    Bars and fonts are scaled down as the node count grows so the panel stays
    compact.
    """
    ax.set_facecolor(bg_color)

    plot_df = degree_df.copy()
    plot_df["node"] = plot_df["node"].astype(str)
    node_order = [str(x) for x in node_order]

    plot_df["node"] = pd.Categorical(
        plot_df["node"],
        categories=node_order,
        ordered=True,
    )
    plot_df = plot_df.sort_values("node").reset_index(drop=True)

    n = max(1, len(plot_df))

    # adaptive bar thickness
    if bar_height is None:
        if n <= 10:
            bar_height = 0.72
        elif n <= 20:
            bar_height = 0.58
        elif n <= 35:
            bar_height = 0.44
        elif n <= 60:
            bar_height = 0.32
        else:
            bar_height = 0.24

    # adaptive label font size
    if n <= 10:
        ytick_fs = 11
    elif n <= 20:
        ytick_fs = 9
    elif n <= 40:
        ytick_fs = 8
    else:
        ytick_fs = 7

    if n <= 20:
        xtick_fs = 10
    elif n <= 40:
        xtick_fs = 9
    else:
        xtick_fs = 8

    y = np.arange(n)

    indeg = plot_df["indegree"].to_numpy(dtype=float)
    outdeg = plot_df["outdegree"].to_numpy(dtype=float)

    # Draw the negative sign on the left to mirror the axis.
    ax.barh(
        y,
        -indeg,
        height=bar_height,
        color="#5B8FD1",
        edgecolor="none",
        alpha=0.92,
        label="In-degree",
    )
    ax.barh(
        y,
        outdeg,
        height=bar_height,
        color="#E07A5F",
        edgecolor="none",
        alpha=0.92,
        label="Out-degree",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["node"].tolist(), fontproperties=font_prop, fontsize=ytick_fs)
    ax.invert_yaxis()

    max_deg = max(
        float(np.max(indeg)) if len(indeg) else 0.0,
        float(np.max(outdeg)) if len(outdeg) else 0.0,
        1.0,
    )
    xpad = max(0.5, 0.08 * max_deg)

    ax.set_xlim(-(max_deg + xpad), max_deg + xpad)

    xticks = np.arange(-int(np.ceil(max_deg)), int(np.ceil(max_deg)) + 1, 2)
    if len(xticks) > 0:
        ax.set_xticks(xticks)
        ax.set_xticklabels(
            [str(abs(int(t))) if t != 0 else "0" for t in xticks],
            fontsize=xtick_fs,
        )

    ax.axvline(0, color="#9E9E9E", linestyle=":", linewidth=1.0)

    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.30)
    ax.grid(axis="y", visible=False)

    ax.set_xlabel("Degree", fontsize=10, fontproperties=font_prop)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", fontproperties=font_prop)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.legend(loc="upper right", frameon=False, prop=font_prop)


# 4. Effect decomposition figure
def plot_adjusted_matrix_heatmap(
    adj_df: pd.DataFrame,
    *,
    title: str = "Adjusted Matrix Heatmap",
    show_values: bool = False,
    zero_eps: float = 1e-12,
    normalize: bool = True,
) -> None:
    """
    Draw the adjacency-matrix heatmap.

    Direction rule: ``adj_df.loc[source, target]`` is ``source -> target``. Off the
    diagonal the colour encodes the edge weight (positive red, negative blue, deeper
    for larger magnitude, white/blank for 0, scaled to [-1, 1] by the largest
    off-diagonal magnitude when ``normalize=True``); on the diagonal it encodes the
    absolute self effect in greyscale, scaled to [0, 1] when ``normalize=True``.
    """
    try:
        adj_df = _normalize_adjacency_matrix(adj_df)
    except Exception as e:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=220)
        ax.text(
            0.5,
            0.5,
            f"Adjacency matrix error: {e}",
            ha="center",
            va="center",
            fontsize=12,
            color="gray",
            transform=ax.transAxes,
            fontproperties=font_prop,
        )
        ax.axis("off")
        return fig

    raw = adj_df.to_numpy(dtype=float)
    n = raw.shape[0]

    diag_mask = np.eye(n, dtype=bool)
    offdiag_mask = ~diag_mask

    # 1. Off-diagonal: edge weights, positive red and negative blue
    offdiag_raw = np.where(offdiag_mask, raw, np.nan)
    offdiag_raw = np.where(np.abs(offdiag_raw) > zero_eps, offdiag_raw, np.nan)

    if np.isfinite(offdiag_raw).any():
        offdiag_max_abs = float(np.nanmax(np.abs(offdiag_raw)))
        if offdiag_max_abs < 1e-12:
            offdiag_max_abs = 1.0
    else:
        offdiag_max_abs = 1.0

    if normalize:
        offdiag_plot = offdiag_raw / offdiag_max_abs
        offdiag_vmin = -1.0
        offdiag_vmax = 1.0
    else:
        offdiag_plot = offdiag_raw
        offdiag_vmin = -offdiag_max_abs
        offdiag_vmax = offdiag_max_abs

    offdiag_ma = np.ma.masked_invalid(offdiag_plot)

    cmap_off = plt.get_cmap("bwr").copy()  # blue-white-red
    cmap_off.set_bad(color=(1, 1, 1, 0))

    # 2. Diagonal: absolute self effect in greyscale
    diag_abs = np.where(diag_mask, np.abs(raw), np.nan)
    diag_abs = np.where(np.abs(diag_abs) > zero_eps, diag_abs, np.nan)

    if np.isfinite(diag_abs).any():
        diag_max = float(np.nanmax(diag_abs))
        if diag_max < 1e-12:
            diag_max = 1.0
    else:
        diag_max = 1.0

    if normalize:
        diag_plot = diag_abs / diag_max
        diag_vmin = 0.0
        diag_vmax = 1.0
    else:
        diag_plot = diag_abs
        diag_vmin = 0.0
        diag_vmax = diag_max

    diag_ma = np.ma.masked_invalid(diag_plot)

    cmap_diag = plt.get_cmap("Greys").copy()
    cmap_diag.set_bad(color=(1, 1, 1, 0))

    # 3. Draw
    fig_size = max(6.5, min(14.0, 0.42 * n + 4.2))
    fig = plt.figure(
        figsize=(fig_size + 2.2, fig_size),
        dpi=220,
        facecolor="white",
    )

    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[18, 0.7, 0.7],
        wspace=0.30,
    )

    ax = fig.add_subplot(gs[0, 0])
    cax_off = fig.add_subplot(gs[0, 1])
    cax_diag = fig.add_subplot(gs[0, 2])

    ax.set_facecolor("#FAFAFA")

    # Off-diagonal edge weights first,
    im_off = ax.imshow(
        offdiag_ma,
        cmap=cmap_off,
        vmin=offdiag_vmin,
        vmax=offdiag_vmax,
        aspect="equal",
        interpolation="none",
    )

    # then the diagonal self effects on top.
    im_diag = ax.imshow(
        diag_ma,
        cmap=cmap_diag,
        vmin=diag_vmin,
        vmax=diag_vmax,
        aspect="equal",
        interpolation="none",
    )

    source_names = adj_df.index.astype(str).tolist()
    target_names = adj_df.columns.astype(str).tolist()

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))

    label_fs = 9 if n <= 15 else 7 if n <= 35 else 6

    ax.set_xticklabels(
        target_names,
        rotation=90,
        fontsize=label_fs,
        fontproperties=font_prop,
    )
    ax.set_yticklabels(
        source_names,
        fontsize=label_fs,
        fontproperties=font_prop,
    )

    ax.set_xlabel("Target node", fontsize=11, fontproperties=font_prop)
    ax.set_ylabel("Source node", fontsize=11, fontproperties=font_prop)

    ax.set_title(
        title,
        fontsize=14,
        fontweight="bold",
        pad=12,
        fontproperties=font_prop,
    )

    # Grid lines
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="#D9D9D9", linestyle="-", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)

    # 4. Optional raw annotations
    if show_values and n <= 20:
        for i in range(n):
            for j in range(n):
                val = raw[i, j]
                if abs(val) <= zero_eps:
                    continue

                if i == j:
                    norm_v = abs(val) / diag_max if diag_max > 0 else 0.0
                else:
                    norm_v = abs(val) / offdiag_max_abs if offdiag_max_abs > 0 else 0.0

                txt_color = "white" if norm_v > 0.55 else "black"

                ax.text(
                    j,
                    i,
                    f"{val:.2g}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=txt_color,
                    fontproperties=font_prop,
                )

    # 5. Colorbar
    cb_off = fig.colorbar(im_off, cax=cax_off)
    if normalize:
        cb_off.set_label(
            f"Edge weight / {offdiag_max_abs:.2g}",
            fontsize=9,
            fontproperties=font_prop,
        )
    else:
        cb_off.set_label(
            "Edge weight",
            fontsize=9,
            fontproperties=font_prop,
        )
    for lab in cb_off.ax.get_yticklabels():
        lab.set_fontproperties(font_prop)

    cb_diag = fig.colorbar(im_diag, cax=cax_diag)
    if normalize:
        cb_diag.set_label(
            f"|Self effect| / {diag_max:.2g}",
            fontsize=9,
            fontproperties=font_prop,
        )
    else:
        cb_diag.set_label(
            "|Self effect|",
            fontsize=9,
            fontproperties=font_prop,
        )
    for lab in cb_diag.ax.get_yticklabels():
        lab.set_fontproperties(font_prop)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#CFCFCF")
        spine.set_linewidth(1.0)

    fig.subplots_adjust(left=0.10, right=0.92, top=0.90, bottom=0.22)

    return fig


def plot_effect(
    quasi_dynamic_df: pd.DataFrame,
    curve_df: pd.DataFrame,
    effect_df_list: list[pd.DataFrame],
    intercept: pd.Series | None = None,
    log_log: bool = False,
    plot_ncols: int = 4,
    plot_max_vars: int = 0,
) -> None:
    """
    Overlay the effects on a grid of subplots.

    Raw quasi-dynamic data are orange scatter points, model predictions a thick blue
    line, the self effect a red line that includes the intercept, and every
    non-self effect its own colour (never red). The global legend lists each
    non-self source with the colour used in the panels.
    """
    
    ncols = max(1, int(plot_ncols))

    cols = list(quasi_dynamic_df.columns)
    if plot_max_vars > 0:
        cols = cols[:plot_max_vars]

    n = len(cols)
    nrows = (n + ncols - 1) // ncols

    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(4.3 * ncols, 3.2 * nrows),
        dpi=300,
    )
    axs = np.array(axs).reshape(-1)

    # Fixed main colours
    color_self = "#D62728"   # self effect (red)
    color_pred = "#1F77B4"   # model prediction (blue)
    color_data = "#F59E0B"   # data (orange)

    # Palette for non-self effects; red is deliberately excluded.
    nonself_palette = [
        "#2CA02C",  # green
        "#9467BD",  # purple
        "#17BECF",  # cyan
        "#8C564B",  # brown
        "#7F7F7F",  # grey
        "#BCBD22",  # olive
        "#1B9E77",  # dark teal
        "#66A61E",  # grass green
        "#6A3D9A",  # dark purple
        "#A6761D",  # ochre
        "#4D908E",  # blue green
        "#577590",  # grey blue
    ]

    # One fixed non-red colour per source.
    all_sources = list(quasi_dynamic_df.columns)
    cross_color_map = {
        src: nonself_palette[i % len(nonself_palette)]
        for i, src in enumerate(all_sources)
    }

    # Global legend handles
    base_handles: list[tuple] = []
    used_cross_sources: list[str] = []
    used_cross_source_set: set[str] = set()

    for idx, c in enumerate(cols):
        ax = axs[idx]

        # raw data
        h_scatter = ax.scatter(
            quasi_dynamic_df.index,
            quasi_dynamic_df[c],
            color=color_data,
            alpha=0.72,
            s=42,
            zorder=1,
            label="Data",
        )

        # model prediction
        h_pred = ax.plot(
            curve_df.index,
            curve_df[c],
            color=color_pred,
            linestyle="-",
            linewidth=2.5,
            zorder=3,
            label="Model prediction (sum of effects)",
        )[0]

        eff_df = effect_df_list[idx]
        icpt = (
            float(intercept.loc[c])
            if intercept is not None and c in intercept.index
            else 0.0
        )

        h_self_line = None

        for src in eff_df.columns:
            y = eff_df[src].to_numpy(dtype=float)

            # self effect: the only red line
            if src == c:
                y = y + icpt
                h_self_line = ax.plot(
                    eff_df.index,
                    y,
                    color=color_self,
                    linewidth=2.8,
                    alpha=0.95,
                    zorder=4,
                    label=f"self effect: {src}",
                )[0]
            else:
                if np.allclose(y, 0.0, rtol=0.0, atol=1e-12):
                    continue

                cross_color = cross_color_map.get(src, "#2CA02C")

                ax.plot(
                    eff_df.index,
                    y,
                    color=cross_color,
                    linewidth=1.8,
                    alpha=0.88,
                    zorder=2,
                    label=f"{src} → {c}",
                )

                # Record which non-self sources actually appear in the global legend.
                if src not in used_cross_source_set:
                    used_cross_source_set.add(src)
                    used_cross_sources.append(src)

        ax.axhline(
            y=0,
            color="black",
            linestyle="--",
            linewidth=1.0,
            alpha=0.45,
            zorder=0,
        )

        if log_log:
            ax.set_xscale("log")
            ax.set_yscale("log")

        ax.set_title(c, fontproperties=font_prop, fontsize=10)
        ax.tick_params(labelsize=8)

        # Axis labels: x on every panel, y only on the leftmost column.
        ax.set_xlabel("Quasi-time (\u03c4)", fontsize=9,
                      fontproperties=font_prop)
        if idx % ncols == 0:
            ax.set_ylabel("Abundance / effect", fontsize=9,
                          fontproperties=font_prop)

        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            lab.set_fontproperties(font_prop)

        # Base legend handles are collected once.
        if idx == 0:
            base_handles = [
                (h_scatter, "Data"),
                (h_pred, "Model prediction (sum of effects)"),
            ]
            if h_self_line is not None:
                base_handles.append((h_self_line, "Self effect (+ intercept)"))

    # Hide unused panels.
    for ax in axs[n:]:
        ax.axis("off")

    plt.tight_layout()

    # Global legend: base entries plus one per non-self source.
    legend_handles = [h for h, _ in base_handles]
    legend_labels = [lab for _, lab in base_handles]

    from matplotlib.lines import Line2D

    for src in used_cross_sources:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=cross_color_map[src],
                linewidth=2.0,
            )
        )
        legend_labels.append(f"non-self effect: {src}")

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=min(5, len(legend_labels)),
        fontsize=8,
        frameon=True,
        prop=font_prop,
    )

    return fig


# Short end labels (long variable names are handled by _effect_label_mode).
_EFFECT_END_LABELS = {"total": "total", "self": "self"}


def _effect_label_mode(
    quasi_dynamic_df: pd.DataFrame,
    curve_df: pd.DataFrame,
    effect_df_list: list[pd.DataFrame],
    intercept: pd.Series | None = None,
    log_log: bool = False,
    plot_ncols: int = 3,
    plot_max_vars: int = 0,
    color_total: str = "#1F77B4",
    color_self: str = "#D62728",
    color_cross: str = "#2CA02C",
    color_data: str = "#F59E0B",
    panel_width_in: float = 5.5,
    label_fontsize: float = 7.0,
    label_pad_pt: float = 2.0,
    var_label_align: str = "left",
    dpi: int = 150,
) -> "plt.Figure":
    """
    Effect-decomposition figure with variable names at the curve ends.

    Panel width and label size come from measurements on real data, and the
    right-hand label band is reserved per panel from the longest label actually
    present, so the macro layer reserves nothing and long micro-layer names are not
    abbreviated. ``ha="right"`` aligns the end of a name with the end of its curve
    and needs no reserved band. ``dpi`` defaults to 150 because widened panels can
    reach 22 in x N, where 300 dpi would need gigabytes (vector PDF is unaffected).
    """
    ncols = max(1, int(plot_ncols))
    cols = list(quasi_dynamic_df.columns)
    if plot_max_vars > 0:
        cols = cols[:plot_max_vars]

    n = len(cols)
    nrows = (n + ncols - 1) // ncols
    fig, axs = plt.subplots(
        nrows, ncols, figsize=(panel_width_in * ncols, 3.9 * nrows), dpi=dpi,
    )
    axs = np.array(axs).reshape(-1)

    # All labels are anchored at the last x point, as in the reference.
    x_last = float(np.asarray(quasi_dynamic_df.index, dtype=float)[-1])
    texts_by_ax: dict = {}       # Per panel: (Text, y value) pairs used to de-overlap labels

    for idx, c in enumerate(cols):
        ax = axs[idx]
        eff = effect_df_list[idx]
        icpt = (float(intercept.loc[c])
                if intercept is not None and c in intercept.index else 0.0)

        # 1) raw quasi-dynamic scatter (open circles), drawn first
        ax.scatter(quasi_dynamic_df.index, quasi_dynamic_df[c],
                   s=22, marker="o", facecolors="none",
                   edgecolors=color_data, linewidths=0.9, alpha=0.75, zorder=1)

        # 2) per-source effect curves: self red, others green, named at the end
        for src in eff.columns:
            y = eff[src].to_numpy(dtype=float)
            is_self = (src == c)
            if is_self:
                y = y + icpt
            if np.allclose(y, 0.0, rtol=0.0, atol=1e-12):
                continue
            col = color_self if is_self else color_cross
            ax.plot(eff.index, y, color=col,
                    lw=3.0 if is_self else 2.0, alpha=0.92, zorder=2)
            # self keeps ha="left"; other names default to ha="right" so the name end
            # aligns with the curve end and no right-hand band is needed.
            if is_self:
                txt, ha, dx = " self", "left", 0.0
            elif var_label_align == "right":
                txt, ha, dx = f"{src} ", "right", -1.0
            else:
                txt, ha, dx = f" {src}", "left", 0.0
            t = ax.text(x_last + dx, float(y[-1]), txt,
                        fontsize=label_fontsize, va="center", ha=ha,
                        color=col, zorder=6, fontproperties=font_prop,
                        clip_on=False)
            texts_by_ax.setdefault(ax, []).append((t, float(y[-1])))

        # 3) total effect (blue dashed) with an end label
        yc = curve_df[c].to_numpy(dtype=float)
        ax.plot(curve_df.index, yc, color=color_total, lw=3.0,
                ls="--", alpha=0.95, zorder=3)
        t = ax.text(x_last, float(yc[-1]), " total",
                    fontsize=label_fontsize, va="center", ha="left",
                    color=color_total, zorder=6, fontproperties=font_prop,
                    clip_on=False)
        texts_by_ax.setdefault(ax, []).append((t, float(yc[-1])))

        # zero reference line and grid
        ax.axhline(0, color="black", linestyle="--", linewidth=1.0,
                   alpha=0.6, zorder=0)
        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.4)

        if log_log:
            ax.set_xscale("log")
            ax.set_yscale("log")

        ax.set_title(c, fontproperties=font_prop, fontsize=10)
        ax.tick_params(labelsize=8)
        ax.set_xlabel("Quasi-time (\u03c4)", fontsize=9, fontproperties=font_prop)
        if idx % ncols == 0:
            ax.set_ylabel("Abundance / effect", fontsize=9,
                          fontproperties=font_prop)
        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            lab.set_fontproperties(font_prop)

    for ax in axs[n:]:
        ax.axis("off")

    plt.tight_layout()

    # After rendering: (a) widen each x axis by its true longest label and
    # (b) spread vertically overlapping labels apart.
    try:
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()

        for idx, c in enumerate(cols):
            ax = axs[idx]
            if var_label_align == "right":
                # Right-aligned names end at the curve, so no right-hand band is needed;
                # a 3% margin keeps the curve off the frame.
                x_lo, x_hi = ax.get_xlim()
                span = (x_hi - x_lo) if x_hi > x_lo else 1.0
                ax.set_xlim(x_lo, x_hi + 0.03 * span)
                continue
            # Left-aligned mode measures the true width of this panel's longest label,
            # reserving only what is needed.
            cand = [" total", " self"] + \
                   [f" {s}" for s in effect_df_list[idx].columns if s != c]
            probes = [ax.text(0, -1e9, t, fontsize=label_fontsize,
                              fontproperties=font_prop) for t in cand]
            fig.canvas.draw()
            need_in = max(p.get_window_extent(renderer=rend).width / fig.dpi
                          for p in probes)
            for p in probes:
                p.remove()
            ax_w_in = ax.get_window_extent(renderer=rend).width / fig.dpi
            x_lo, x_hi = ax.get_xlim()
            span = (x_hi - x_lo) if x_hi > x_lo else 1.0
            # Only a 2% safety margin: the measurement already includes the glyph box.
            frac = float(np.clip(need_in * 1.02 / max(ax_w_in, 1e-6), 0.02, 0.40))
            ax.set_xlim(x_lo, x_hi + frac * span)

        # (b) de-overlap: sort by y and enforce a minimum gap bottom-up
        fig.canvas.draw()
        for ax, items in texts_by_ax.items():
            if len(items) < 2:
                continue
            y_lo, y_hi = ax.get_ylim()
            yspan = (y_hi - y_lo) if y_hi > y_lo else 1.0
            ax_h_px = ax.get_window_extent(renderer=rend).height
            # One text line is about 1.35 font sizes, converted to data units.
            gap_data = (label_fontsize * 1.35 / 72.0) * (yspan / (ax_h_px / fig.dpi))
            items.sort(key=lambda t: t[1])
            for k in range(1, len(items)):
                t_prev, y_prev = items[k - 1]
                t_cur, y_cur = items[k]
                if y_cur - y_prev < gap_data:
                    y_new = y_prev + gap_data
                    items[k] = (t_cur, y_new)
                    t_cur.set_position((t_cur.get_position()[0], y_new))
        fig.canvas.draw()
    except Exception as exc:
        print(f"    [note] label layout skipped: {type(exc).__name__}: {exc}")

    # The legend explains the three curve colours only; names are on the curves.
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color=color_total, linewidth=3.0, linestyle="--"),
        Line2D([0], [0], color=color_self, linewidth=3.0),
        Line2D([0], [0], color=color_cross, linewidth=2.0),
        Line2D([0], [0], color=color_data, marker="o", linestyle="none",
               markerfacecolor="none", markeredgecolor=color_data,
               markersize=6, markeredgewidth=0.9),
    ]
    labels = ["Total effect (total)", "Self effect (self)",
              "Cross effect (variable name)", "Observed"]
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.012),
               ncol=4, fontsize=8, frameon=False, prop=font_prop)
    return fig


# 5. Network figure: called directly by the NetRecon page

def plot_network(
    adj_df: pd.DataFrame,
    target_node: str = "",
    *,
    layout: str = "hub_circular",
    node_order_by: str = "outdegree",
    hub_center: bool = True,
    hub_n: int = 3,
    hub_by: str = "outdegree",
    edge_threshold: float = 0.0,
    top_edges: int | None = None,
    include_self_edges: bool = False,
    uniform_node_size: bool = True,
    node_size: float = 850,
    node_facecolor: str = "#EADFB4",
    node_edge_color: str = "#2C4A6B",
    node_linewidth: float = 1.1,
    positive_edge_color: str = "#E74C3C",
    negative_edge_color: str = "#2F7ED8",
    show_degree_panel: bool = True,
    reciprocal_rad: float = 0.30,
    single_edge_rad: float = 0.00,
) -> None:
    """
    Draw the directed weighted network.

    Direction rule: ``adj_df.loc[source, target]`` is ``source -> target``. When
    ``target_node`` is given, only edges pointing at that node are shown.
    """

    try:
        adj_df = _normalize_adjacency_matrix(adj_df)
    except Exception as e:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
        ax.text(
            0.5,
            0.5,
            f"Adjacency matrix error: {e}",
            ha="center",
            va="center",
            fontsize=12,
            color="gray",
            transform=ax.transAxes,
            fontproperties=font_prop,
        )
        ax.axis("off")
        return fig

    # target_node filter: keep only source -> target_node edges
    if target_node:
        target_node = str(target_node)

        if target_node not in adj_df.columns:
            fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
            ax.text(
                0.5,
                0.5,
                "No edge matches the filter",
                ha="center",
                va="center",
                fontsize=14,
                color="gray",
                transform=ax.transAxes,
                fontproperties=font_prop,
            )
            ax.axis("off")
            return fig

        source_nodes = [
            source
            for source in adj_df.index
            if source != target_node
            and abs(float(adj_df.loc[source, target_node])) > float(edge_threshold)
        ]

        if not source_nodes:
            fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
            ax.text(
                0.5,
                0.5,
                "No edge matches the filter",
                ha="center",
                va="center",
                fontsize=14,
                color="gray",
                transform=ax.transAxes,
                fontproperties=font_prop,
            )
            ax.axis("off")
            return fig

        keep_nodes = source_nodes + [target_node]
        plot_adj = pd.DataFrame(
            0.0,
            index=keep_nodes,
            columns=keep_nodes,
        )

        for source in source_nodes:
            plot_adj.loc[source, target_node] = float(adj_df.loc[source, target_node])

        title = f"Interactive network pointing at {target_node}"

    else:
        plot_adj = adj_df
        title = "Directed interaction network"

    # No edge left after filtering.
    tmp_arr = plot_adj.to_numpy(dtype=float, copy=True)

    if not include_self_edges:
        np.fill_diagonal(tmp_arr, 0.0)

    if not (np.abs(tmp_arr) > float(edge_threshold)).any():
        fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
        ax.text(
            0.5,
            0.5,
            "No edge matches the filter",
            ha="center",
            va="center",
            fontsize=14,
            color="gray",
            transform=ax.transAxes,
            fontproperties=font_prop,
        )
        ax.axis("off")
        return fig

    node_order = _get_node_order(
        plot_adj,
        order_by=node_order_by,
        edge_threshold=edge_threshold,
        include_self_edges=include_self_edges,
    )

    hub_nodes = (
        _select_hub_nodes(
            plot_adj,
            hub_n=min(int(hub_n), plot_adj.shape[0]),
            hub_by=hub_by,
            edge_threshold=edge_threshold,
            include_self_edges=include_self_edges,
        )
        if hub_center or layout == "hub_circular"
        else []
    )

    layout_to_use = "hub_circular" if hub_center else layout

    G = _build_digraph(
        plot_adj,
        edge_threshold=edge_threshold,
        top_edges=top_edges,
        include_self_edges=include_self_edges,
    )

    # Keep isolated nodes visible.
    for node in plot_adj.index:
        if str(node) not in G:
            G.add_node(str(node), node_weight=float(plot_adj.loc[node, node]))

    pos = _get_layout(
        G,
        layout="hub_circular",
        seed=123,
        layout_scale=1.0,
        node_order=node_order,
        hub_nodes=hub_nodes,
        hub_inner_radius=0.45,
        hub_outer_radius=1.35,
        clockwise=True,
    )

    nodes = [str(x) for x in G.nodes()]
    draw_nodes = [node for node in node_order if node in nodes]
    draw_nodes += [node for node in nodes if node not in draw_nodes]

    size_dict = _compute_node_sizes(
        plot_adj,
        nodes=draw_nodes,
        uniform_node_size=uniform_node_size,
        node_size=node_size,
        node_size_min=450,
        node_size_max=1800,
        node_size_scale=1.0,
        edge_threshold=edge_threshold,
        include_self_edges=include_self_edges,
    )

    n_nodes = len(draw_nodes)

    # Scale the figure height with the node count.
    fig_height = max(7.5, 0.55 * n_nodes + 3.5)

    if show_degree_panel:
        n_nodes = len(node_order)

        # adaptive figure height
        if n_nodes <= 10:
            fig_height = 7.5
            degree_panel_ratio = 1.10
        elif n_nodes <= 20:
            fig_height = 8.8
            degree_panel_ratio = 1.00
        elif n_nodes <= 40:
            fig_height = 10.8
            degree_panel_ratio = 0.90
        elif n_nodes <= 70:
            fig_height = 12.8
            degree_panel_ratio = 0.82
        else:
            fig_height = min(18.0, 0.16 * n_nodes + 4.0)
            degree_panel_ratio = 0.74

        fig = plt.figure(figsize=(15.5, fig_height), dpi=180, facecolor="#F6F6F6")
        gs = fig.add_gridspec(
            1,
            3,
            width_ratios=[2.35, 0.10, degree_panel_ratio],
            wspace=0.00,
        )

        ax_net = fig.add_subplot(gs[0, 0])
        ax_gap = fig.add_subplot(gs[0, 1])
        ax_bar = fig.add_subplot(gs[0, 2])
        ax_gap.axis("off")
    else:
        fig, ax_net = plt.subplots(figsize=(9.0, fig_height), dpi=200, facecolor="#F6F6F6")
        ax_bar = None
        
    ax_net.set_facecolor("white")

    if ax_bar is not None:
        ax_bar.set_facecolor("white")

    # Frame around the network panel.
    net_border = plt.Rectangle(
        (0, 0), 1, 1,
        transform=ax_net.transAxes,
        fill=False,
        edgecolor="#D9D9D9",
        linewidth=1.0,
        zorder=10,
        clip_on=False,
    )
    ax_net.add_patch(net_border)

    # Frame around the degree panel.
    if ax_bar is not None:
        bar_border = plt.Rectangle(
            (0, 0), 1, 1,
            transform=ax_bar.transAxes,
            fill=False,
            edgecolor="#D9D9D9",
            linewidth=1.0,
            zorder=10,
            clip_on=False,
        )
        ax_bar.add_patch(bar_border)

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=draw_nodes,
        node_size=[size_dict[node] for node in draw_nodes],
        node_color=node_facecolor,
        edgecolors=node_edge_color,
        linewidths=node_linewidth,
        ax=ax_net,
    )

    _draw_signed_edges(
        G,
        pos,
        ax_net,
        positive_edge_color=positive_edge_color,
        negative_edge_color=negative_edge_color,
        arrowsize=18,
        edge_alpha=0.92,
        reciprocal_rad=reciprocal_rad,
        single_edge_rad=single_edge_rad,
    )

    _draw_labels_with_font(
        ax_net,
        pos,
        draw_nodes,
        font_size=9,
        font_color="black",
        font_weight="normal",
    )

    ax_net.set_title(
        title,
        fontsize=18,
        fontweight="bold",
        pad=16,
        fontproperties=font_prop,
    )
    ax_net.axis("off")
    _center_network_axis(ax_net, pos, pad_ratio=0.32)

    legend_elements = [
        Line2D([0], [0], color=positive_edge_color, linewidth=2.8, label="Positive (promoting)"),
        Line2D([0], [0], color=negative_edge_color, linewidth=2.8, label="Negative (inhibiting)"),
    ]

    ax_net.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=2,
        fontsize=8,
        frameon=True,
        prop=font_prop,
    )

    if show_degree_panel and ax_bar is not None:
        degree_df = _degree_table(
            plot_adj,
            edge_threshold=edge_threshold,
            include_self_edges=include_self_edges,
            sort_by="node",
            ascending=True,
        )

        _mirror_barh_degree(
            ax_bar,
            degree_df,
            node_order=node_order,
            title="In-degree / Out-degree",
            bg_color="white",
        )

    fig.subplots_adjust(left=0.04, right=0.98, top=0.92, bottom=0.10)
    return fig


def _resolve_modules(modules=None):
    """
    Return the module order used by this call.

    The original script passed state between functions through the module-level
    ``MODULES`` global. Under concurrent sessions one session would overwrite that
    list and another would build its HTML from the wrong node set, so the module list
    is now an explicit argument that only falls back to the global when omitted;
    standalone use of the original script still works.
    """
    if modules is not None:
        return list(modules)
    return list(MODULES)


def module_number(x):

    s = str(x).strip()

    if s.upper().startswith("M"):

        try:
            return int(float(s[1:]))

        except Exception:
            return 10**9

    try:
        return int(float(s))

    except Exception:
        return 10**9


def sort_modules(x):

    return sorted(
        [str(i) for i in x],
        key=module_number
    )


def make_degree(
    edges,
    modules=None,
):

    rows = []


    for module in _resolve_modules(modules):

        outgoing = edges[
            edges[
                "source"
            ]
            ==
            module
        ]


        incoming = edges[
            edges[
                "target"
            ]
            ==
            module
        ]


        rows.append(
            {
                "module":
                    module,

                "out_degree":
                    len(
                        outgoing
                    ),

                "in_degree":
                    len(
                        incoming
                    ),

                "total_degree":
                    len(
                        outgoing
                    )
                    +
                    len(
                        incoming
                    ),

                "positive_out":
                    int(
                        (
                            outgoing[
                                "weight"
                            ]
                            >
                            0
                        ).sum()
                    ),

                "negative_out":
                    int(
                        (
                            outgoing[
                                "weight"
                            ]
                            <
                            0
                        ).sum()
                    ),

                "positive_in":
                    int(
                        (
                            incoming[
                                "weight"
                            ]
                            >
                            0
                        ).sum()
                    ),

                "negative_in":
                    int(
                        (
                            incoming[
                                "weight"
                            ]
                            <
                            0
                        ).sum()
                    ),

                "weighted_out_abs":
                    float(
                        outgoing[
                            "weight"
                        ]
                        .abs()
                        .sum()
                    ),

                "weighted_in_abs":
                    float(
                        incoming[
                            "weight"
                        ]
                        .abs()
                        .sum()
                    ),
            }
        )


    return (

        pd.DataFrame(
            rows
        )

        .sort_values(

            [
                "out_degree",
                "total_degree",
                "weighted_out_abs",
            ],

            ascending=[
                False,
                False,
                False
            ]
        )

        .reset_index(
            drop=True
        )
    )


def _stable_integer(
    text,
):

    digest = hashlib.md5(
        str(text).encode(
            "utf-8"
        )
    ).hexdigest()

    return int(
        digest[
            :8
        ],
        16
    )


def _ring_capacity(
    ring_index,
):

    return int(

        min(

            RING_CAPACITY_START

            +

            ring_index
            *
            RING_CAPACITY_STEP,

            RING_CAPACITY_MAX
        )
    )


def _split_nodes_to_rings(
    nodes,
    start_ring_index=0,
):

    nodes = list(
        nodes
    )

    rings = []

    cursor = 0

    ring_index = int(
        start_ring_index
    )


    while cursor < len(
        nodes
    ):

        capacity = (
            _ring_capacity(
                ring_index
            )
        )


        current = nodes[
            cursor:
            cursor
            +
            capacity
        ]


        rings.append(
            {
                "ring_index":
                    ring_index,

                "nodes":
                    current,
            }
        )


        cursor += len(
            current
        )


        ring_index += 1


    return rings


def _ring_radius(
    ring_number,
):

    ring_number = int(
        ring_number
    )


    if ring_number == 0:

        return 0.0


    if ring_number == 1:

        return float(
            FIRST_NONHUB_RING_RADIUS
        )


    if ring_number == 2:

        return float(
            SECOND_NONHUB_RING_RADIUS
        )


    radius = float(
        SECOND_NONHUB_RING_RADIUS
    )


    for current_ring in range(
        3,
        ring_number
        +
        1
    ):

        gap = (

            RING_GAP_BASE

            +

            (
                current_ring
                -
                3
            )

            *

            RING_GAP_GROWTH
        )


        radius += gap


    return float(
        radius
    )


def _node_diameter(
    ring_number,
):

    ring_number = int(
        ring_number
    )


    if ring_number == 0:

        return float(
            HUB_NODE_DIAMETER
        )


    diameter = (

        FIRST_RING_DIAMETER

        -

        (
            ring_number
            -
            1
        )

        *

        RING_DIAMETER_DECAY
    )


    return float(

        max(
            diameter,
            MIN_NODE_DIAMETER
        )
    )


def _ring_color(
    ring_number,
):

    return RING_COLOR_PALETTE[

        int(
            ring_number
        )

        %

        len(
            RING_COLOR_PALETTE
        )
    ]


def _select_display_edges(
    edges,
):

    if edges.empty:

        return edges.copy()


    work = edges.copy()


    work[
        "abs_weight"
    ] = (
        work[
            "weight"
        ]
        .abs()
    )


    threshold = float(

        work[
            "abs_weight"
        ]
        .quantile(
            EDGE_QUANTILE
        )
    )


    selected_parts = []


    for _, g in work.groupby(
        "target",
        sort=False
    ):

        selected_parts.append(

            g
            .sort_values(
                "abs_weight",
                ascending=False
            )
            .head(
                TOPK_PER_TARGET
            )
        )


    for _, g in work.groupby(
        "source",
        sort=False
    ):

        selected_parts.append(

            g
            .sort_values(
                "abs_weight",
                ascending=False
            )
            .head(
                TOPK_PER_SOURCE
            )
        )


    selected_parts.append(

        work.loc[
            work[
                "abs_weight"
            ]
            >=
            threshold
        ]
    )


    selected = (

        pd.concat(
            selected_parts,
            ignore_index=True
        )

        .drop_duplicates(
            subset=[
                "source",
                "target",
            ]
        )
    )


    min_required = min(

        MIN_DISPLAY_EDGES,

        len(
            work
        )
    )


    if len(
        selected
    ) < min_required:

        supplement = (

            work

            .sort_values(
                "abs_weight",
                ascending=False
            )

            .head(
                min_required
            )
        )


        selected = (

            pd.concat(
                [
                    selected,
                    supplement,
                ],
                ignore_index=True
            )

            .drop_duplicates(
                subset=[
                    "source",
                    "target",
                ]
            )
        )


    selected = (

        selected

        .sort_values(
            "abs_weight",
            ascending=False
        )

        .head(
            MAX_DISPLAY_EDGES
        )

        .reset_index(
            drop=True
        )
    )


    return selected


def _build_ring_layout(
    degree_df,
    group_name,
    modules=None,
):

    work = (
        degree_df
        .copy()
    )


    work[
        "module"
    ] = (
        work[
            "module"
        ]
        .astype(str)
    )


    work = (

        work

        .set_index(
            "module"
        )

        .reindex(
            _resolve_modules(modules)
        )

        .fillna(
            0.0
        )
    )


    # As in the single-cell figure: the largest weighted out-degree nodes
    # enter the centre / inner ring first.

    nonzero_df = (

        work.loc[
            work[
                "weighted_out_abs"
            ]
            >
            0
        ]

        .copy()

        .sort_values(

            [
                "weighted_out_abs",
                "out_degree",
                "total_degree",
            ],

            ascending=[
                False,
                False,
                False,
            ]
        )
    )


    hub_nodes = (

        nonzero_df

        .head(
            N_CENTER_HUBS
        )

        .index
        .astype(str)
        .tolist()
    )


    remaining_nonzero = [

        node

        for node in (
            nonzero_df
            .index
            .astype(str)
            .tolist()
        )

        if node not in hub_nodes
    ]


    zero_nodes = (

        work.loc[
            work[
                "weighted_out_abs"
            ]
            <=
            0
        ]

        .index
        .astype(str)
        .tolist()
    )


    rng = np.random.default_rng(

        LAYOUT_SEED

        +

        (
            _stable_integer(
                group_name
            )
            %
            10000
        )
    )


    # Zero out-degree nodes go to the outermost ring.
    rng.shuffle(
        zero_nodes
    )


    nonzero_rings = (
        _split_nodes_to_rings(
            remaining_nonzero,
            start_ring_index=0,
        )
    )


    ring_records = []


    for item in nonzero_rings:

        ring_records.append(
            {
                "ring_number":
                    item[
                        "ring_index"
                    ]
                    +
                    1,

                "nodes":
                    item[
                        "nodes"
                    ],

                "zero_ring":
                    False,
            }
        )


    next_ring_index = (

        ring_records[
            -1
        ][
            "ring_number"
        ]

        if ring_records

        else 0
    )


    zero_rings = (
        _split_nodes_to_rings(
            zero_nodes,
            start_ring_index=
                next_ring_index,
        )
    )


    for item in zero_rings:

        ring_records.append(
            {
                "ring_number":
                    item[
                        "ring_index"
                    ]
                    +
                    1,

                "nodes":
                    item[
                        "nodes"
                    ],

                "zero_ring":
                    True,
            }
        )


    positions = {}

    ring_map = {}



    if len(
        hub_nodes
    ) == 1:

        node = hub_nodes[
            0
        ]


        positions[
            node
        ] = (
            0.0,
            0.0
        )


        ring_map[
            node
        ] = 0


    elif len(
        hub_nodes
    ) > 1:

        for i, node in enumerate(
            hub_nodes
        ):

            angle = (

                np.pi
                /
                2

                -

                2
                *
                np.pi
                *
                i
                /
                len(
                    hub_nodes
                )
            )


            positions[
                node
            ] = (

                CENTER_HUB_RADIUS
                *
                np.cos(
                    angle
                ),

                CENTER_HUB_RADIUS
                *
                np.sin(
                    angle
                ),
            )


            ring_map[
                node
            ] = 0



    for record in ring_records:

        ring_number = int(
            record[
                "ring_number"
            ]
        )


        nodes = list(
            record[
                "nodes"
            ]
        )


        n_nodes = len(
            nodes
        )


        if n_nodes == 0:

            continue


        radius = (
            _ring_radius(
                ring_number
            )
        )


        base_angle = (

            np.pi
            /
            2

            -

            np.deg2rad(

                RING_ROTATION_DEG

                *

                (
                    ring_number
                    -
                    1
                )
            )
        )


        if (
            EVEN_RING_HALF_STEP
            and
            ring_number
            %
            2
            ==
            0
        ):

            base_angle -= (

                np.pi

                /
                n_nodes
            )


        angle_step = (

            2
            *
            np.pi
            /
            n_nodes
        )


        for i, node in enumerate(
            nodes
        ):

            angle = (

                base_angle

                -

                i
                *
                angle_step
            )


            angle += rng.normal(

                0.0,

                ANGLE_JITTER_FRACTION
                *
                angle_step
            )


            current_radius = (

                radius

                *

                (
                    1.0

                    +

                    rng.normal(
                        0.0,
                        RADIUS_JITTER
                    )
                )
            )


            positions[
                node
            ] = (

                current_radius
                *
                np.cos(
                    angle
                ),

                current_radius
                *
                np.sin(
                    angle
                ),
            )


            ring_map[
                node
            ] = (
                ring_number
            )


    rows = []


    for module in _resolve_modules(modules):

        if module not in positions:

            # Should be unreachable; defensive fallback only.
            positions[
                module
            ] = (
                0.0,
                0.0
            )

            ring_map[
                module
            ] = 0


        rows.append(
            {
                "group":
                    group_name,

                "module":
                    module,

                "x":
                    float(
                        positions[
                            module
                        ][0]
                    ),

                "y":
                    float(
                        positions[
                            module
                        ][1]
                    ),

                "ring_number":
                    int(
                        ring_map[
                            module
                        ]
                    ),

                "is_hub":
                    bool(
                        module
                        in
                        hub_nodes
                    ),

                "weighted_out_abs":
                    float(
                        work.loc[
                            module,
                            "weighted_out_abs"
                        ]
                    ),

                "out_degree":
                    int(
                        work.loc[
                            module,
                            "out_degree"
                        ]
                    ),
            }
        )


    return pd.DataFrame(
        rows
    )


def _edge_width_levels(
    display_edges,
):

    if display_edges.empty:

        return (
            0.0,
            0.0
        )


    values = (

        display_edges[
            "weight"
        ]
        .abs()
        .to_numpy(
            dtype=float
        )
    )


    return (
        float(
            np.quantile(
                values,
                1 / 3
            )
        ),

        float(
            np.quantile(
                values,
                2 / 3
            )
        ),
    )


def _edge_width(
    abs_weight,
    q1,
    q2,
):

    if abs_weight >= q2:

        return float(
            EDGE_WIDTH_STRONG
        )


    if abs_weight >= q1:

        return float(
            EDGE_WIDTH_MEDIUM
        )


    return float(
        EDGE_WIDTH_WEAK
    )


def _edge_curvature(
    source,
    target,
    reciprocal,
):

    integer = _stable_integer(
        f"{source}->{target}"
    )


    sign = (

        -1

        if integer % 2 == 0

        else 1
    )


    if reciprocal:

        magnitude = float(
            RECIPROCAL_CURVATURE
        )


    else:

        magnitude = (

            CURVATURE_BASE

            +

            CURVATURE_STEP

            *

            (
                integer
                %
                4
            )
        )


    return float(
        sign
        *
        magnitude
    )


def _hex_to_rgba(
    color,
    alpha,
):

    color = (
        str(
            color
        )
        .replace(
            "#",
            ""
        )
    )


    r = int(
        color[
            0:2
        ],
        16
    )


    g = int(
        color[
            2:4
        ],
        16
    )


    b = int(
        color[
            4:6
        ],
        16
    )


    return (
        f"rgba({r},{g},{b},{float(alpha):.4f})"
    )


def generate_effect_download_artifacts(
    group_dir,
):
    """
    Build the effect-download artifacts from what run_group() already saved.

    Reads ``15_effect_decomposition`` (``predicted.csv``, ``intercept.csv`` and one
    CSV per module under ``effect/``) and writes ``plots/<module>_effect.png`` /
    ``.pdf`` plus ``16_effect_decomposition_download.zip``. The HTML does not render
    these figures, it only links to them.
    """

    effect_root = os.path.join(
        group_dir,
        "15_effect_decomposition"
    )


    predicted_file = os.path.join(
        effect_root,
        "predicted.csv"
    )


    intercept_file = os.path.join(
        effect_root,
        "intercept.csv"
    )


    effect_dir = os.path.join(
        effect_root,
        "effect"
    )


    if not (
        os.path.exists(
            predicted_file
        )
        and
        os.path.exists(
            intercept_file
        )
        and
        os.path.isdir(
            effect_dir
        )
    ):

        print(
            "[Effect warning] incomplete effect decomposition files: "
        )

        print(
            effect_root
        )


        return {
            "effect_root":
                effect_root,

            "plot_dir":
                None,

            "zip_path":
                None,

            "targets":
                [],
        }


    plot_dir = os.path.join(
        effect_root,
        "plots"
    )


    os.makedirs(
        plot_dir,
        exist_ok=True
    )


    predicted = pd.read_csv(
        predicted_file,
        index_col=0
    )


    predicted.index = pd.to_numeric(
        predicted.index,
        errors="coerce"
    )


    predicted = predicted.loc[
        np.isfinite(
            predicted.index
        )
    ].copy()


    predicted = predicted.apply(
        pd.to_numeric,
        errors="coerce"
    )


    intercept = pd.read_csv(
        intercept_file,
        index_col=0
    )


    intercept.index = (
        intercept.index
        .astype(str)
    )


    effect_files = [

        x

        for x in os.listdir(
            effect_dir
        )

        if x.lower().endswith(
            ".csv"
        )
    ]


    effect_files = sorted(
        effect_files,
        key=module_number
    )


    targets = []


    for filename in effect_files:

        target = os.path.splitext(
            filename
        )[0]


        effect_file = os.path.join(
            effect_dir,
            filename
        )


        effect_df = pd.read_csv(
            effect_file,
            index_col=0
        )


        effect_df.index = pd.to_numeric(
            effect_df.index,
            errors="coerce"
        )


        effect_df = effect_df.loc[
            np.isfinite(
                effect_df.index
            )
        ].copy()


        effect_df = effect_df.apply(
            pd.to_numeric,
            errors="coerce"
        )


        if target not in predicted.columns:

            print(
                f"[Effect warning] {target} is missing from predicted.csv; skipping its plot."
            )

            continue


        targets.append(
            target
        )


        x_effect = effect_df.index.to_numpy(
            dtype=float
        )


        x_pred = predicted.index.to_numpy(
            dtype=float
        )


        predicted_y = predicted[
            target
        ].to_numpy(
            dtype=float
        )


        intercept_value = 0.0


        if target in intercept.index:

            intercept_value = float(

                pd.to_numeric(

                    intercept.loc[
                        target
                    ].iloc[
                        0
                    ],

                    errors="coerce"
                )
            )


        fig, ax = plt.subplots(
            figsize=(
                9.2,
                5.5
            )
        )



        ax.plot(

            x_pred,

            predicted_y,

            linewidth=2.8,

            label="Predicted",

            zorder=10,
        )



        if target in effect_df.columns:

            self_plus_intercept = (

                effect_df[
                    target
                ].to_numpy(
                    dtype=float
                )

                +

                intercept_value
            )


            ax.plot(

                x_effect,

                self_plus_intercept,

                linewidth=2.1,

                linestyle="--",

                label="Self + intercept",

                zorder=9,
            )



        nonself_sources = [

            str(
                c
            )

            for c in effect_df.columns

            if str(
                c
            )
            !=
            target
        ]


        for source in nonself_sources:

            ax.plot(

                x_effect,

                effect_df[
                    source
                ].to_numpy(
                    dtype=float
                ),

                linewidth=1.35,

                alpha=0.88,

                label=source,
            )


        ax.axhline(

            0.0,

            linewidth=0.7,

            alpha=0.30,
        )


        ax.set_xlabel(
            "Quasi index"
        )


        ax.set_ylabel(
            "Effect / reconstructed state"
        )


        ax.set_title(
            f"{target} effect decomposition"
        )


        ax.grid(
            alpha=0.15
        )


        ax.legend(

            frameon=False,

            fontsize=7.5,

            ncol=2,

            bbox_to_anchor=(
                1.02,
                1.0
            ),

            loc="upper left",
        )


        fig.tight_layout()


        if SAVE_EFFECT_PNG:

            fig.savefig(

                os.path.join(
                    plot_dir,
                    f"{target}_effect.png"
                ),

                dpi=EFFECT_DPI,

                bbox_inches="tight",

                facecolor="white"
            )


        if SAVE_EFFECT_PDF:

            fig.savefig(

                os.path.join(
                    plot_dir,
                    f"{target}_effect.pdf"
                ),

                bbox_inches="tight",

                facecolor="white"
            )


        plt.close(
            fig
        )



    zip_path = os.path.join(
        group_dir,
        "16_effect_decomposition_download.zip"
    )


    with zipfile.ZipFile(

        zip_path,

        mode="w",

        compression=
            zipfile.ZIP_DEFLATED

    ) as zf:


        for root, _, files in os.walk(
            effect_root
        ):

            for filename in files:

                full_path = os.path.join(
                    root,
                    filename
                )


                arcname = os.path.relpath(

                    full_path,

                    start=effect_root
                )


                zf.write(

                    full_path,

                    arcname=arcname
                )


    return {

        "effect_root":
            effect_root,

        "plot_dir":
            plot_dir,

        "zip_path":
            zip_path,

        "targets":
            targets,

        "predicted_file":
            predicted_file,

        "intercept_file":
            intercept_file,

        "effect_dir":
            effect_dir,
    }


def _relative_href(
    target_path,
    html_path,
):

    if target_path is None:

        return None


    rel = os.path.relpath(

        target_path,

        start=
            os.path.dirname(
                html_path
            )
    )


    return rel.replace(
        "\\",
        "/"
    )


def write_interactive_network_html(
    edges,
    degree_df,
    output_path,
    title,
    group_name,
    group_dir,
    modules=None,
):

    """
    Single-page HTML with the network, the degree chart and the effect downloads.

    Left: the hub-centred concentric-ring IDOP network. Right: the in/out-degree
    chart, recomputed from exactly the edges currently visible. Bottom (collapsed):
    the effect-decomposition download links. The weak-edge filter slider and the
    degree chart always share the same visible edge set.
    """

    if edges.empty:

        print(
            "[HTML] edges empty, skip."
        )

        return



    display_edges = (
        _select_display_edges(
            edges
        )
    )


    display_edges.to_csv(

        os.path.join(
            group_dir,
            "13B_display_edges.csv"
        ),

        index=False,

        encoding="utf-8-sig"
    )



    layout_df = (
        _build_ring_layout(
            degree_df,
            group_name,
            modules=modules,
        )
    )


    layout_df.to_csv(

        os.path.join(
            group_dir,
            "13A_ring_layout.csv"
        ),

        index=False,

        encoding="utf-8-sig"
    )


    layout_index = (

        layout_df

        .set_index(
            "module"
        )
    )


    degree_index = (

        degree_df

        .copy()

        .set_index(
            "module"
        )
    )



    effect_info = (
        generate_effect_download_artifacts(
            group_dir
        )
    )



    nodes_json = []

    initial_positions = {}


    for module in _resolve_modules(modules):

        row = layout_index.loc[
            module
        ]


        ring_number = int(
            row[
                "ring_number"
            ]
        )


        is_hub = bool(
            row[
                "is_hub"
            ]
        )


        x = float(
            row[
                "x"
            ]
        ) * POSITION_SCALE


        y = -float(
            row[
                "y"
            ]
        ) * POSITION_SCALE


        diameter = (
            _node_diameter(
                ring_number
            )
        )


        node_size = float(
            diameter
            *
            0.78
        )


        font_size = (

            14.0

            if is_hub

            else max(
                8.0,
                12.0
                -
                0.7
                *
                ring_number
            )
        )


        if module in degree_index.index:

            drow = degree_index.loc[
                module
            ]


            out_degree = int(
                drow[
                    "out_degree"
                ]
            )


            in_degree = int(
                drow[
                    "in_degree"
                ]
            )


            weighted_out = float(
                drow[
                    "weighted_out_abs"
                ]
            )


            weighted_in = float(
                drow[
                    "weighted_in_abs"
                ]
            )


        else:

            out_degree = 0

            in_degree = 0

            weighted_out = 0.0

            weighted_in = 0.0


        node_color = (
            _ring_color(
                ring_number
            )
        )


        nodes_json.append(
            {
                "id":
                    module,

                "label":
                    module,

                "x":
                    x,

                "y":
                    y,

                "baseX":
                    x,

                "baseY":
                    y,

                "ring":
                    ring_number,

                "isHub":
                    is_hub,

                "baseSize":
                    node_size,

                "size":
                    node_size,

                "baseFont":
                    font_size,

                "font":
                    {
                        "size":
                            font_size,

                        "color":
                            TEXT_COLOR,

                        "face":
                            "Arial",
                    },

                "shape":
                    "dot",

                "color":
                    {
                        "background":
                            node_color,

                        "border":
                            NODE_EDGE_COLOR,

                        "highlight":
                            {
                                "background":
                                    node_color,

                                "border":
                                    "#111111",
                            },
                    },

                "borderWidth":
                    1.4
                    if is_hub
                    else
                    0.8,

                "title":
                    (
                        f"<b>{module}</b>"
                        f"<br>Group: {group_name}"
                        f"<br>Ring: {ring_number}"
                        f"<br>Hub: {is_hub}"
                        f"<br>Full out-degree: {out_degree}"
                        f"<br>Full in-degree: {in_degree}"
                        f"<br>Full weighted out: {weighted_out:.6g}"
                        f"<br>Full weighted in: {weighted_in:.6g}"
                    ),
            }
        )


        initial_positions[
            module
        ] = {
            "x":
                x,

            "y":
                y,
        }



    q1, q2 = (
        _edge_width_levels(
            display_edges
        )
    )


    pair_set = {

        (
            str(
                r.source
            ),
            str(
                r.target
            ),
        )

        for r in display_edges.itertuples(
            index=False
        )
    }


    edges_json = []


    for edge_id, r in enumerate(
        display_edges.itertuples(
            index=False
        )
    ):

        source = str(
            r.source
        )


        target = str(
            r.target
        )


        weight = float(
            r.weight
        )


        abs_weight = abs(
            weight
        )


        reciprocal = (

            (
                target,
                source
            )

            in

            pair_set
        )


        curvature = (
            _edge_curvature(
                source,
                target,
                reciprocal,
            )
        )


        smooth_type = (

            "curvedCW"

            if curvature >= 0

            else "curvedCCW"
        )


        roundness = float(

            np.clip(

                abs(
                    curvature
                )
                *
                3.2,

                0.045,

                0.34
            )
        )


        base_width = (
            _edge_width(
                abs_weight,
                q1,
                q2,
            )
        )


        edge_color = (

            POSITIVE_EDGE_COLOR

            if weight > 0

            else NEGATIVE_EDGE_COLOR
        )


        edges_json.append(
            {
                "id":
                    f"e{edge_id}",

                "from":
                    source,

                "to":
                    target,

                "weight":
                    weight,

                "absWeight":
                    abs_weight,

                "baseWidth":
                    base_width,

                "baseColor":
                    edge_color,

                "arrows":
                    {
                        "to":
                            {
                                "enabled":
                                    True,

                                "scaleFactor":
                                    0.62,
                            }
                    },

                "smooth":
                    {
                        "enabled":
                            True,

                        "type":
                            smooth_type,

                        "roundness":
                            roundness,
                    },

                "arrowStrikethrough":
                    False,

                "title":
                    (
                        f"{source} → {target}"
                        f"<br>weight={weight:.6g}"
                    ),
            }
        )



    effect_targets = {}


    if (
        effect_info.get(
            "plot_dir"
        )
        is not None
    ):

        for target in effect_info.get(
            "targets",
            []
        ):

            raw_csv = os.path.join(
                effect_info[
                    "effect_dir"
                ],
                f"{target}.csv"
            )


            png = os.path.join(
                effect_info[
                    "plot_dir"
                ],
                f"{target}_effect.png"
            )


            pdf = os.path.join(
                effect_info[
                    "plot_dir"
                ],
                f"{target}_effect.pdf"
            )


            effect_targets[
                target
            ] = {

                "csv":
                    _relative_href(
                        raw_csv,
                        output_path
                    ),

                "png":
                    (
                        _relative_href(
                            png,
                            output_path
                        )
                        if os.path.exists(
                            png
                        )
                        else None
                    ),

                "pdf":
                    (
                        _relative_href(
                            pdf,
                            output_path
                        )
                        if os.path.exists(
                            pdf
                        )
                        else None
                    ),
            }


    effect_download_data = {

        "zip":
            (
                _relative_href(
                    effect_info.get(
                        "zip_path"
                    ),
                    output_path
                )
                if effect_info.get(
                    "zip_path"
                )
                else None
            ),

        "predicted":
            (
                _relative_href(
                    effect_info.get(
                        "predicted_file"
                    ),
                    output_path
                )
                if effect_info.get(
                    "predicted_file"
                )
                else None
            ),

        "intercept":
            (
                _relative_href(
                    effect_info.get(
                        "intercept_file"
                    ),
                    output_path
                )
                if effect_info.get(
                    "intercept_file"
                )
                else None
            ),

        "targets":
            effect_targets,
    }



    nodes_text = json.dumps(
        nodes_json,
        ensure_ascii=False
    )


    edges_text = json.dumps(
        edges_json,
        ensure_ascii=False
    )


    initial_text = json.dumps(
        initial_positions,
        ensure_ascii=False
    )


    effect_text = json.dumps(
        effect_download_data,
        ensure_ascii=False
    )


    storage_key = (
        f"ProteinIDOP|{group_name}|coarse"
    )



    html = f"""<!DOCTYPE html>

<html lang="zh-CN">

<head>

<meta charset="utf-8"/>

<meta name="viewport"
      content="width=device-width, initial-scale=1.0"/>

<title>{title}</title>


<script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>


<style>

* {{
    box-sizing: border-box;
}}


html,
body {{

    margin: 0;

    padding: 0;

    background: #f5f6f8;

    color: #202020;

    font-family:
        Arial,
        Helvetica,
        sans-serif;
}}


#toolbar {{

    position: sticky;

    top: 0;

    z-index: 999;

    padding: 9px 13px;

    background:
        rgba(
            255,
            255,
            255,
            0.98
        );

    border-bottom:
        1px solid #d7d7d7;

    box-shadow:
        0 1px 8px
        rgba(
            0,
            0,
            0,
            0.05
        );
}}


#title {{

    font-size: 17px;

    font-weight: 700;

    margin-bottom: 8px;
}}


.controls {{

    display: flex;

    flex-wrap: wrap;

    align-items: center;

    gap: 7px 12px;

    font-size: 12px;
}}


.control {{

    display: inline-flex;

    align-items: center;

    gap: 5px;
}}


.control input[type="range"] {{

    width: 100px;
}}


.value {{

    min-width: 36px;

    text-align: right;

    font-variant-numeric:
        tabular-nums;
}}


button,
.download-link {{

    border:
        1px solid #bdbdbd;

    border-radius: 5px;

    background: white;

    padding: 5px 8px;

    color: #202020;

    text-decoration: none;

    cursor: pointer;

    font-size: 12px;
}}


button:hover,
.download-link:hover {{

    background: #eeeeee;
}}


.legend {{

    margin-top: 7px;

    display: flex;

    flex-wrap: wrap;

    gap: 10px 16px;

    color: #666666;

    font-size: 11px;
}}


.legend-line {{

    display: inline-block;

    width: 25px;

    height: 3px;

    vertical-align: middle;

    margin-right: 4px;
}}


.main-grid {{

    display: grid;

    grid-template-columns:
        minmax(
            0,
            1.65fr
        )
        minmax(
            390px,
            0.85fr
        );

    gap: 8px;

    padding: 8px;
}}


.card {{

    min-width: 0;

    background: white;

    border:
        1px solid #d8d8d8;

    border-radius: 7px;

    overflow: hidden;
}}


.card-title {{

    height: 36px;

    display: flex;

    align-items: center;

    justify-content: center;

    border-bottom:
        1px solid #e5e5e5;

    font-size: 13px;

    font-weight: 700;

    background: #fbfbfb;
}}


#network {{

    width: 100%;

    height: {HTML_HEIGHT}px;

    background: white;
}}


.degree-toolbar {{

    padding: 7px 9px;

    border-bottom:
        1px solid #e8e8e8;

    display: flex;

    flex-wrap: wrap;

    align-items: center;

    gap: 7px;

    font-size: 11px;
}}


#degreeScroll {{

    height: {DEGREE_PANEL_HEIGHT - 50}px;

    overflow-y: auto;

    overflow-x: hidden;

    background: white;
}}


#degreeSvg {{

    width: 100%;

    display: block;
}}


.effect-panel {{

    margin:
        0
        8px
        12px
        8px;

    background: white;

    border:
        1px solid #d8d8d8;

    border-radius: 7px;

    padding: 9px 11px;
}}


.effect-panel summary {{

    cursor: pointer;

    font-weight: 700;

    font-size: 13px;
}}


.effect-links {{

    margin-top: 10px;

    display: flex;

    flex-wrap: wrap;

    gap: 7px;

    align-items: center;
}}


.effect-note {{

    margin-top: 7px;

    color: #666666;

    font-size: 11px;
}}


.status {{

    margin-left: auto;

    color: #666666;

    font-size: 11px;
}}


@media (
    max-width: 1150px
) {{

    .main-grid {{

        grid-template-columns:
            1fr;
    }}
}}

</style>

</head>


<body>


<div id="toolbar">

    <div id="title">
        {title}
    </div>


    <div class="controls">


        <label class="control">

            Nodes

            <input
                id="nodeScale"
                type="range"
                min="0.45"
                max="5.00"
                step="0.05"
                value="
{DEFAULT_NODE_SCALE}"
            >

            <span
                id="nodeScaleValue"
                class="value"
            >
                {DEFAULT_NODE_SCALE:.2f}
            </span>

        </label>


        <label class="control">

            Labels

            <input
                id="labelScale"
                type="range"
                min="0.40"
                max="6.00"
                step="0.05"
                value="
{DEFAULT_LABEL_SCALE}"
            >

            <span
                id="labelScaleValue"
                class="value"
            >
                {DEFAULT_LABEL_SCALE:.2f}
            </span>

        </label>


        <label class="control">

            Edge width

            <input
                id="edgeScale"
                type="range"
                min="0.25"
                max="8.00"
                step="0.05"
                value="
{DEFAULT_EDGE_SCALE}"
            >

            <span
                id="edgeScaleValue"
                class="value"
            >
                {DEFAULT_EDGE_SCALE:.2f}
            </span>

        </label>


        <label class="control">

            Edge opacity

            <input
                id="edgeOpacity"
                type="range"
                min="0.05"
                max="1.00"
                step="0.05"
                value="
{DEFAULT_EDGE_OPACITY}"
            >

            <span
                id="edgeOpacityValue"
                class="value"
            >
                {DEFAULT_EDGE_OPACITY:.2f}
            </span>

        </label>


        <label class="control">

            Weak-edge filter

            <input
                id="edgePercentile"
                type="range"
                min="0"
                max="95"
                step="1"
                value="
{DEFAULT_EDGE_PERCENTILE}"
            >

            <span
                id="edgePercentileValue"
                class="value"
            >
                {DEFAULT_EDGE_PERCENTILE}%
            </span>

        </label>


        <label class="control">

            <input
                id="showLabels"
                type="checkbox"
                checked
            >

            Labels

        </label>


        <button id="fitBtn">
            Fit
        </button>


        <button id="resetBtn">
            Reset rings
        </button>


        <button id="saveBtn">
            Save layout
        </button>


        <button id="loadBtn">
            Load layout
        </button>


        <button id="exportPngBtn">
            Export network PNG
        </button>


        <span
            id="status"
            class="status"
        >
            Ready
        </span>

    </div>


    <div class="legend">

        <span>

            <span
                class="legend-line"
                style="background:
{POSITIVE_EDGE_COLOR};"
            ></span>

            Positive edge

        </span>


        <span>

            <span
                class="legend-line"
                style="background:{NEGATIVE_EDGE_COLOR};"
            ></span>

            Negative edge

        </span>


        <span>

            <span
                class="legend-line"
                style="background:{IN_DEGREE_COLOR};"
            ></span>

            In-degree

        </span>


        <span>

            <span
                class="legend-line"
                style="background:{OUT_DEGREE_COLOR};"
            ></span>

            Out-degree

        </span>


        <span>
            Degree is always recomputed from the currently visible edges
        </span>

    </div>

</div>


<div class="main-grid">


    <section class="card">

        <div
            class="card-title"
        >
            IDOP Network
        </div>


        <div id="network"></div>

    </section>


    <section class="card">

        <div
            id="degreeTitle"
            class="card-title"
        >
            Filter-synchronized In / Out Degree
        </div>


        <div class="degree-toolbar">


            <label>

                Degree metric

                <select id="degreeMetric">

                    <option value="count">
                        Count
                    </option>

                    <option value="weighted">
                        Weighted |weight|
                    </option>

                </select>

            </label>


            <button id="degreeCsvBtn">
                Export current Degree CSV
            </button>


            <button id="degreeSvgBtn">
                Export current Degree SVG
            </button>


        </div>


        <div id="degreeScroll">

            <svg
                id="degreeSvg"
                role="img"
                aria-label="Filter synchronized in and out degree chart"
            ></svg>

        </div>

    </section>


</div>


<details class="effect-panel">

    <summary>
        Effect decomposition downloads
        (hidden from the main HTML view by default)
    </summary>


    <div class="effect-links">


        <a
            id="effectZipLink"
            class="download-link"
            href="#"
            download
        >
            Download all effect files (ZIP)
        </a>


        <a
            id="predictedLink"
            class="download-link"
            href="#"
            download
        >
            predicted.csv
        </a>


        <a
            id="interceptLink"
            class="download-link"
            href="#"
            download
        >
            intercept.csv
        </a>


        <label>

            Target

            <select id="effectTargetSelect">
            </select>

        </label>


        <a
            id="effectPngLink"
            class="download-link"
            href="#"
            download
        >
            Download target PNG
        </a>


        <a
            id="effectPdfLink"
            class="download-link"
            href="#"
            download
        >
            Download target PDF
        </a>


        <a
            id="effectCsvLink"
            class="download-link"
            href="#"
            download
        >
            Download target effect CSV
        </a>


    </div>


    <div class="effect-note">

        Effect figures are pre-rendered locally;
        the page does not draw them, which would compete with the network and degree figures for browser resources.

    </div>

</details>


<script>


const BASE_NODES = 
{nodes_text};

const BASE_EDGES = {edges_text};

const INITIAL_POSITIONS = {initial_text};

const EFFECT_DATA = {effect_text};

const STORAGE_KEY = {json.dumps(storage_key)};


const nodeSet = new vis.DataSet(
    BASE_NODES
);


const edgeSet = new vis.DataSet(
    BASE_EDGES
);


const container = document.getElementById(
    "network"
);


const network = new vis.Network(

    container,

    {{
        nodes:
            nodeSet,

        edges:
            edgeSet
    }},

    {{
        autoResize:
            true,

        layout:
            {{
                improvedLayout:
                    false
            }},

        physics:
            {{
                enabled:
                    false
            }},

        interaction:
            {{
                hover:
                    true,

                hoverConnectedEdges:
                    true,

                dragNodes:
                    true,

                dragView:
                    true,

                zoomView:
                    true,

                multiselect:
                    true,

                navigationButtons:
                    true,

                keyboard:
                    true,

                tooltipDelay:
                    80
            }},

        edges:
            {{
                arrowStrikethrough:
                    false,

                selectionWidth:
                    1.8,

                hoverWidth:
                    0.8
            }}
    }}
);


let currentVisibleEdges = [];

let currentDegreeRows = [];


// =============================================================================
// JS helpers
// =============================================================================

function setStatus(
    text
) {{

    document
    .getElementById(
        "status"
    )
    .textContent = text;
}}


function hexToRgba(
    hex,
    alpha
) {{

    const h = String(
        hex
    ).replace(
        "#",
        ""
    );


    const r = parseInt(
        h.substring(
            0,
            2
        ),
        16
    );


    const g = parseInt(
        h.substring(
            2,
            4
        ),
        16
    );


    const b = parseInt(
        h.substring(
            4,
            6
        ),
        16
    );


    return (
        `rgba(${{r}},${{g}},${{b}},${{alpha}})`
    );
}}


function percentile(
    values,
    p
) {{

    if (
        values.length === 0
    ) {{

        return 0;
    }}


    const x = values
        .slice()
        .sort(
            (a, b) => a - b
        );


    const index = (

        (
            x.length
            -
            1
        )

        *

        p

        /

        100.0
    );


    const lo = Math.floor(
        index
    );


    const hi = Math.ceil(
        index
    );


    if (
        lo === hi
    ) {{

        return x[
            lo
        ];
    }}


    const frac = (
        index
        -
        lo
    );


    return (

        x[
            lo
        ]
        *
        (
            1
            -
            frac
        )

        +

        x[
            hi
        ]
        *
        frac
    );
}}


function downloadBlob(
    filename,
    blob
) {{

    const url = URL.createObjectURL(
        blob
    );


    const a = document.createElement(
        "a"
    );


    a.href = url;

    a.download = filename;


    document.body.appendChild(
        a
    );


    a.click();


    document.body.removeChild(
        a
    );


    setTimeout(
        () => URL.revokeObjectURL(
            url
        ),
        1000
    );
}}


// =============================================================================
// Degree recomputation using CURRENT VISIBLE EDGES
// =============================================================================

function calculateCurrentDegree() {{

    const rows = {{}};


    BASE_NODES.forEach(
        node => {{

            rows[
                node.id
            ] = {{

                module:
                    node.id,

                in_degree:
                    0,

                out_degree:
                    0,

                weighted_in:
                    0,

                weighted_out:
                    0,
            }};
        }}
    );


    currentVisibleEdges.forEach(
        edge => {{

            if (
                rows[
                    edge.from
                ]
            ) {{

                rows[
                    edge.from
                ].out_degree += 1;


                rows[
                    edge.from
                ].weighted_out += Number(
                    edge.absWeight
                );
            }}


            if (
                rows[
                    edge.to
                ]
            ) {{

                rows[
                    edge.to
                ].in_degree += 1;


                rows[
                    edge.to
                ].weighted_in += Number(
                    edge.absWeight
                );
            }}
        }}
    );


    currentDegreeRows = Object.values(
        rows
    );


    return currentDegreeRows;
}}


function renderDegreeChart() {{

    const rows = calculateCurrentDegree();


    const metric = (
        document.getElementById(
            "degreeMetric"
        ).value
    );


    const inKey = (

        metric === "weighted"

        ? "weighted_in"

        : "in_degree"
    );


    const outKey = (

        metric === "weighted"

        ? "weighted_out"

        : "out_degree"
    );


    const sorted = rows
        .slice()
        .sort(
            (a, b) => {{

                if (
                    b[
                        outKey
                    ]
                    !==
                    a[
                        outKey
                    ]
                ) {{

                    return (
                        b[
                            outKey
                        ]
                        -
                        a[
                            outKey
                        ]
                    );
                }}


                if (
                    b[
                        inKey
                    ]
                    !==
                    a[
                        inKey
                    ]
                ) {{

                    return (
                        b[
                            inKey
                        ]
                        -
                        a[
                            inKey
                        ]
                    );
                }}


                return (
                    String(
                        a.module
                    )
                    .localeCompare(
                        String(
                            b.module
                        ),
                        undefined,
                        {{
                            numeric:
                                true
                        }}
                    )
                );
            }}
        );


    const svg = document.getElementById(
        "degreeSvg"
    );


    while (
        svg.firstChild
    ) {{

        svg.removeChild(
            svg.firstChild
        );
    }}


    const panelWidth = Math.max(

        430,

        document.getElementById(
            "degreeScroll"
        ).clientWidth
        -
        8
    );


    const rowHeight = 24;

    const topMargin = 34;

    const bottomMargin = 34;


    const height = (

        topMargin

        +

        sorted.length
        *
        rowHeight

        +

        bottomMargin
    );


    svg.setAttribute(
        "viewBox",
        `0 0 ${{panelWidth}} ${{height}}`
    );


    svg.setAttribute(
        "height",
        String(
            height
        )
    );


    const center = (
        panelWidth
        /
        2
    );


    const labelHalfWidth = 43;

    const labelGap = 6;

    const sidePadding = 25;


    const maxBarWidth = Math.max(

        70,

        center
        -
        labelHalfWidth
        -
        labelGap
        -
        sidePadding
    );


    const maxValue = Math.max(

        1e-12,

        ...sorted.map(
            row => Math.max(

                Number(
                    row[
                        inKey
                    ]
                ),

                Number(
                    row[
                        outKey
                    ]
                )
            )
        )
    );


    const ns = (
        "http://www.w3.org/2000/svg"
    );


    function svgElement(
        name,
        attrs
    ) {{

        const el = document.createElementNS(
            ns,
            name
        );


        Object.entries(
            attrs
        ).forEach(
            ([key, value]) => {{

                el.setAttribute(
                    key,
                    String(
                        value
                    )
                );
            }}
        );


        return el;
    }}


    const headerIn = svgElement(

        "text",

        {{
            x:
                center
                -
                labelHalfWidth
                -
                10,

            y:
                20,

            "text-anchor":
                "end",

            "font-size":
                11,

            fill:
                "#555555",
        }}
    );


    headerIn.textContent = (

        metric === "weighted"

        ? "Weighted in"

        : "In-degree"
    );


    svg.appendChild(
        headerIn
    );


    const headerOut = svgElement(

        "text",

        {{
            x:
                center
                +
                labelHalfWidth
                +
                10,

            y:
                20,

            "text-anchor":
                "start",

            "font-size":
                11,

            fill:
                "#555555",
        }}
    );


    headerOut.textContent = (

        metric === "weighted"

        ? "Weighted out"

        : "Out-degree"
    );


    svg.appendChild(
        headerOut
    );


    svg.appendChild(

        svgElement(

            "line",

            {{
                x1:
                    center,

                x2:
                    center,

                y1:
                    topMargin
                    -
                    8,

                y2:
                    height
                    -
                    bottomMargin
                    +
                    4,

                stroke:
                    "#bdbdbd",

                "stroke-width":
                    1,
            }}
        )
    );


    sorted.forEach(
        (row, index) => {{

            const y = (

                topMargin

                +

                index
                *
                rowHeight

                +

                rowHeight
                /
                2
            );


            const inValue = Number(
                row[
                    inKey
                ]
            );


            const outValue = Number(
                row[
                    outKey
                ]
            );


            const inWidth = (

                inValue

                /
                maxValue

                *
                maxBarWidth
            );


            const outWidth = (

                outValue

                /
                maxValue

                *
                maxBarWidth
            );


            svg.appendChild(

                svgElement(

                    "rect",

                    {{
                        x:
                            center
                            -
                            labelHalfWidth
                            -
                            labelGap
                            -
                            inWidth,

                        y:
                            y
                            -
                            6,

                        width:
                            inWidth,

                        height:
                            12,

                        fill:
                            "{IN_DEGREE_COLOR}",

                        opacity:
                            0.84,
                    }}
                )
            );


            svg.appendChild(

                svgElement(

                    "rect",

                    {{
                        x:
                            center
                            +
                            labelHalfWidth
                            +
                            labelGap,

                        y:
                            y
                            -
                            6,

                        width:
                            outWidth,

                        height:
                            12,

                        fill:
                            "{OUT_DEGREE_COLOR}",

                        opacity:
                            0.84,
                    }}
                )
            );


            const label = svgElement(

                "text",

                {{
                    x:
                        center,

                    y:
                        y
                        +
                        3.5,

                    "text-anchor":
                        "middle",

                    "font-size":
                        10,

                    fill:
                        "#202020",
                }}
            );


            label.textContent = (
                row.module
            );


            svg.appendChild(
                label
            );


            const formatter = (

                metric === "weighted"

                ? value => (
                    value.toFixed(
                        3
                    )
                )

                : value => (
                    String(
                        Math.round(
                            value
                        )
                    )
                )
            );


            const inText = svgElement(

                "text",

                {{
                    x:
                        center
                        -
                        labelHalfWidth
                        -
                        labelGap
                        -
                        inWidth
                        -
                        4,

                    y:
                        y
                        +
                        3.5,

                    "text-anchor":
                        "end",

                    "font-size":
                        8.5,

                    fill:
                        "#555555",
                }}
            );


            inText.textContent = formatter(
                inValue
            );


            svg.appendChild(
                inText
            );


            const outText = svgElement(

                "text",

                {{
                    x:
                        center
                        +
                        labelHalfWidth
                        +
                        labelGap
                        +
                        outWidth
                        +
                        4,

                    y:
                        y
                        +
                        3.5,

                    "text-anchor":
                        "start",

                    "font-size":
                        8.5,

                    fill:
                        "#555555",
                }}
            );


            outText.textContent = formatter(
                outValue
            );


            svg.appendChild(
                outText
            );
        }}
    );


    document
    .getElementById(
        "degreeTitle"
    )
    .textContent = (

        "Filter-synchronized In / Out Degree"
        +
        ` | visible edges: ${{currentVisibleEdges.length}} / ${{BASE_EDGES.length}}`
    );
}}


// =============================================================================
// Network controls
// =============================================================================

function applyControls() {{

    const nodeScale = parseFloat(
        document.getElementById(
            "nodeScale"
        ).value
    );


    const labelScale = parseFloat(
        document.getElementById(
            "labelScale"
        ).value
    );


    const edgeScale = parseFloat(
        document.getElementById(
            "edgeScale"
        ).value
    );


    const edgeOpacity = parseFloat(
        document.getElementById(
            "edgeOpacity"
        ).value
    );


    const edgePercentile = parseFloat(
        document.getElementById(
            "edgePercentile"
        ).value
    );


    const showLabels = (
        document.getElementById(
            "showLabels"
        ).checked
    );


    document
    .getElementById(
        "nodeScaleValue"
    )
    .textContent = (
        nodeScale.toFixed(
            2
        )
    );


    document
    .getElementById(
        "labelScaleValue"
    )
    .textContent = (
        labelScale.toFixed(
            2
        )
    );


    document
    .getElementById(
        "edgeScaleValue"
    )
    .textContent = (
        edgeScale.toFixed(
            2
        )
    );


    document
    .getElementById(
        "edgeOpacityValue"
    )
    .textContent = (
        edgeOpacity.toFixed(
            2
        )
    );


    document
    .getElementById(
        "edgePercentileValue"
    )
    .textContent = (
        Math.round(
            edgePercentile
        )
        +
        "%"
    );


    nodeSet.update(

        BASE_NODES.map(
            n => (
                {{

                    id:
                        n.id,

                    size:
                        n.baseSize
                        *
                        nodeScale,

                    label:
                        showLabels
                        ?
                        n.id
                        :
                        "",

                    font:
                        {{

                            size:
                                n.baseFont
                                *
                                labelScale,

                            color:
                                "{TEXT_COLOR}",

                            face:
                                "Arial"
                        }}
                }}
            )
        )
    );


    // -------------------------------------------------------------------------
    // ONE threshold controls both network and degree.
    // -------------------------------------------------------------------------

    const absValues = BASE_EDGES
        .map(
            e => Number(
                e.absWeight
            )
        )
        .filter(
            x => Number.isFinite(
                x
            )
        );


    const threshold = percentile(
        absValues,
        edgePercentile
    );


    currentVisibleEdges = BASE_EDGES.filter(
        e => (
            Number(
                e.absWeight
            )
            >=
            threshold
        )
    );


    const visibleIds = new Set(

        currentVisibleEdges.map(
            e => e.id
        )
    );


    edgeSet.update(

        BASE_EDGES.map(
            e => (
                {{

                    id:
                        e.id,

                    width:
                        e.baseWidth
                        *
                        edgeScale,

                    hidden:
                        !visibleIds.has(
                            e.id
                        ),

                    color:
                        {{

                            color:
                                hexToRgba(
                                    e.baseColor,
                                    edgeOpacity
                                ),

                            highlight:
                                hexToRgba(

                                    e.baseColor,

                                    Math.min(
                                        1.0,
                                        edgeOpacity
                                        +
                                        0.30
                                    )
                                )
                        }}
                }}
            )
        )
    );


    renderDegreeChart();


    setStatus(

        `visible edges ${{currentVisibleEdges.length}} / ${{BASE_EDGES.length}}`
        +
        ` | |w| threshold = ${{threshold.toPrecision(4)}}`
    );
}}


// =============================================================================
// Effect downloads
// =============================================================================

function setDownloadLink(
    id,
    href
) {{

    const link = document.getElementById(
        id
    );


    if (
        href
    ) {{

        link.href = href;

        link.style.display = "inline-block";

    }} else {{

        link.href = "#";

        link.style.display = "none";
    }}
}}


function updateTargetEffectLinks() {{

    const select = document.getElementById(
        "effectTargetSelect"
    );


    const target = select.value;


    const info = (

        EFFECT_DATA.targets[
            target
        ]

        ||
        {{}}
    );


    setDownloadLink(
        "effectPngLink",
        info.png
    );


    setDownloadLink(
        "effectPdfLink",
        info.pdf
    );


    setDownloadLink(
        "effectCsvLink",
        info.csv
    );
}}


function initializeEffectDownloads() {{

    setDownloadLink(
        "effectZipLink",
        EFFECT_DATA.zip
    );


    setDownloadLink(
        "predictedLink",
        EFFECT_DATA.predicted
    );


    setDownloadLink(
        "interceptLink",
        EFFECT_DATA.intercept
    );


    const select = document.getElementById(
        "effectTargetSelect"
    );


    Object.keys(
        EFFECT_DATA.targets
        ||
        {{}}
    )
    .sort(
        (a, b) => (
            a.localeCompare(
                b,
                undefined,
                {{
                    numeric:
                        true
                }}
            )
        )
    )
    .forEach(
        target => {{

            const option = document.createElement(
                "option"
            );


            option.value = target;

            option.textContent = target;


            select.appendChild(
                option
            );
        }}
    );


    updateTargetEffectLinks();
}}


// =============================================================================
// Exports
// =============================================================================

function downloadBlob(
    filename,
    blob
) {{

    const url = URL.createObjectURL(
        blob
    );


    const a = document.createElement(
        "a"
    );


    a.href = url;

    a.download = filename;


    document.body.appendChild(
        a
    );


    a.click();


    document.body.removeChild(
        a
    );


    setTimeout(
        () => URL.revokeObjectURL(
            url
        ),
        1000
    );
}}


function exportDegreeCsv() {{

    calculateCurrentDegree();


    const lines = [

        [
            "module",
            "in_degree",
            "out_degree",
            "weighted_in",
            "weighted_out",
        ].join(
            ","
        )
    ];


    currentDegreeRows.forEach(
        row => {{

            lines.push(

                [
                    row.module,
                    row.in_degree,
                    row.out_degree,
                    row.weighted_in,
                    row.weighted_out,
                ].join(
                    ","
                )
            );
        }}
    );


    const blob = new Blob(

        [
            "\\ufeff"
            +
            lines.join(
                "\\n"
            )
        ],

        {{
            type:
                "text/csv;charset=utf-8"
        }}
    );


    downloadBlob(

        "{group_name}_CURRENT_FILTER_degree.csv",

        blob
    );
}}


function exportDegreeSvg() {{

    const svg = document.getElementById(
        "degreeSvg"
    );


    const source = (
        new XMLSerializer()
        .serializeToString(
            svg
        )
    );


    const blob = new Blob(

        [
            source
        ],

        {{
            type:
                "image/svg+xml;charset=utf-8"
        }}
    );


    downloadBlob(

        "{group_name}_CURRENT_FILTER_degree.svg",

        blob
    );
}}


// =============================================================================
// Events
// =============================================================================

[
    "nodeScale",
    "labelScale",
    "edgeScale",
    "edgeOpacity",
    "edgePercentile",
].forEach(

    id => {{

        document
        .getElementById(
            id
        )
        .addEventListener(
            "input",
            applyControls
        );
    }}
);


document
.getElementById(
    "showLabels"
)
.addEventListener(
    "change",
    applyControls
);


document
.getElementById(
    "degreeMetric"
)
.addEventListener(
    "change",
    renderDegreeChart
);


document
.getElementById(
    "effectTargetSelect"
)
.addEventListener(
    "change",
    updateTargetEffectLinks
);


document
.getElementById(
    "fitBtn"
)
.addEventListener(

    "click",

    () => {{

        network.fit(
            {{
                animation:
                    {{
                        duration:
                            300
                    }}
            }}
        );
    }}
);


document
.getElementById(
    "resetBtn"
)
.addEventListener(

    "click",

    () => {{

        nodeSet.update(

            Object.entries(
                INITIAL_POSITIONS
            )
            .map(

                ([id, p]) => (
                    {{

                        id:
                            id,

                        x:
                            p.x,

                        y:
                            p.y
                    }}
                )
            )
        );


        network.fit(
            {{
                animation:
                    {{
                        duration:
                            250
                    }}
            }}
        );


        setStatus(
            "Initial ring layout restored"
        );
    }}
);


document
.getElementById(
    "saveBtn"
)
.addEventListener(

    "click",

    () => {{

        localStorage.setItem(

            STORAGE_KEY,

            JSON.stringify(
                {{

                    positions:
                        network.getPositions(),

                    savedAt:
                        new Date()
                        .toISOString()
                }}
            )
        );


        setStatus(
            "Current layout saved"
        );
    }}
);


document
.getElementById(
    "loadBtn"
)
.addEventListener(

    "click",

    () => {{

        const raw = localStorage.getItem(
            STORAGE_KEY
        );


        if (
            !raw
        ) {{

            setStatus(
                "No saved layout"
            );

            return;
        }}


        const payload = JSON.parse(
            raw
        );


        nodeSet.update(

            Object.entries(
                payload.positions
                ||
                {{}}
            )
            .map(

                ([id, p]) => (
                    {{

                        id:
                            id,

                        x:
                            p.x,

                        y:
                            p.y
                    }}
                )
            )
        );


        network.fit(
            {{
                animation:
                    {{
                        duration:
                            250
                    }}
            }}
        );


        setStatus(
            "Saved layout loaded"
        );
    }}
);


document
.getElementById(
    "exportPngBtn"
)
.addEventListener(

    "click",

    () => {{

        network.redraw();


        const canvas = (
            network
            .canvas
            .frame
            .canvas
        );


        const a = document.createElement(
            "a"
        );


        a.download = (
            "{group_name}_CURRENT_FILTER_network.png"
        );


        a.href = canvas.toDataURL(
            "image/png"
        );


        document.body.appendChild(
            a
        );


        a.click();


        document.body.removeChild(
            a
        );
    }}
);


document
.getElementById(
    "degreeCsvBtn"
)
.addEventListener(
    "click",
    exportDegreeCsv
);


document
.getElementById(
    "degreeSvgBtn"
)
.addEventListener(
    "click",
    exportDegreeSvg
);


window.addEventListener(

    "resize",

    () => {{

        renderDegreeChart();
    }}
);


// =============================================================================
// Restore saved layout if present
// =============================================================================

const saved = localStorage.getItem(
    STORAGE_KEY
);


if (
    saved
) {{

    try {{

        const payload = JSON.parse(
            saved
        );


        nodeSet.update(

            Object.entries(
                payload.positions
                ||
                {{}}
            )
            .map(

                ([id, p]) => (
                    {{

                        id:
                            id,

                        x:
                            p.x,

                        y:
                            p.y
                    }}
                )
            )
        );


    }} catch(
        error
    ) {{

        console.error(
            error
        );
    }}
}}


// =============================================================================
// Initial render
// =============================================================================

network.once(

    "afterDrawing",

    () => {{

        network.fit(
            {{
                animation:
                    false
            }}
        );
    }}
);


initializeEffectDownloads();

applyControls();


</script>


</body>

</html>
"""


    with open(

        output_path,

        "w",

        encoding="utf-8"

    ) as f:

        f.write(
            html
        )


    print(
        "[HTML overwritten]",
        output_path
    )


# Standalone entry point: network_data / CSV -> HTML


def plot_interactive_network_html(
    data,
    output_path,
    title=None,
    group_name=None,
    group_dir=None,
    modules=None,
):
    """
    Render an edge table (DataFrame or CSV path) as an interactive HTML network.

    The input matches ``network_data.generate_network_data()``: at least ``source`` /
    ``target`` / ``weight``, with any extra ``sign`` column ignored. ``title``
    defaults to ``f"{group_name} | Functional Modules Original IDOP"``;
    ``group_name`` labels the hubs and defaults to the CSV stem (``"network"`` for a
    DataFrame); ``group_dir`` holds the ring-layout, display-edge and effect-download
    files and defaults to the directory of ``output_path``; ``modules`` is the node
    set and display order, derived from ``source | target`` when omitted. Returns
    ``output_path``.
    """
    global MODULES

    if isinstance(data, (str, os.PathLike)):

        # CSV path: read as utf-8-sig, which also accepts the utf-8 (no BOM)
        # output written by network_data.py.
        edges = pd.read_csv(
            data,
            encoding="utf-8-sig",
        )

        source_name = (
            os.path.splitext(
                os.path.basename(
                    data
                )
            )[0]
        )

    elif isinstance(data, pd.DataFrame):

        edges = data.copy()

        source_name = None

    else:

        raise TypeError(
            f"got {type(data).__name__}."

        )

    missing = [
        col
        for col in [
            "source",
            "target",
            "weight",
        ]
        if col not in edges.columns
    ]

    if missing:

        raise ValueError(
            "missing input columns: "
            + ", ".join(
                repr(col)
                for col in missing
            )
            + "; source / target / weight are required, "
            "and the sign column written by network_data is ignored."
        )

    edges = (
        edges[
            [
                "source",
                "target",
                "weight",
            ]
        ]
        .copy()
    )

    edges["source"] = (
        edges["source"]
        .astype(str)
        .str.strip()
    )

    edges["target"] = (
        edges["target"]
        .astype(str)
        .str.strip()
    )

    edges["weight"] = pd.to_numeric(
        edges["weight"],
        errors="coerce",
    )

    edges = (
        edges.dropna(
            subset=["weight"]
        )
    )

    if edges.empty:

        raise ValueError(
            "no valid edge record in the CSV."
        )

    # Node set = source | target with the original sort rule; an explicit
    # modules argument wins so concurrent sessions cannot overwrite each other.
    resolved_modules = (
        list(modules)
        if modules is not None
        else
        sort_modules(
            set(
                edges["source"]
            )
            |
            set(
                edges["target"]
            )
        )
    )

    # Keep standalone-script usage working by refreshing the module global.
    MODULES = resolved_modules

    degree = (
        make_degree(
            edges,
            modules=resolved_modules,
        )
    )

    if group_name is None:

        group_name = (
            source_name
            or
            "network"
        )

    if group_dir is None:

        group_dir = os.path.dirname(
            os.path.abspath(
                output_path
            )
        )

    os.makedirs(
        group_dir,
        exist_ok=True
    )

    if title is None:

        title = (
            f"{group_name} | Functional Modules Original IDOP"
        )

    write_interactive_network_html(
        edges,

        degree_df=
            degree,

        output_path=
            output_path,

        title=
            title,

        group_name=
            group_name,

        group_dir=
            group_dir,

        modules=
            resolved_modules,
    )

    return output_path


def network_edges_from_adjacency(
    adj_df,
    target_node=None,
    top_edges=None,
):
    """
    Convert an adjacency matrix into a ``source`` / ``target`` / ``weight`` edge table.

    ``adj_df.loc[source, target]`` is the ``source -> target`` weight. This bridges
    the static plotting pipeline and the interactive HTML: an ``adj_df`` must become
    an edge table before it can be given to :func:`plot_network`. ``target_node``
    keeps only edges pointing at that node and ``top_edges`` keeps the N largest
    ``|weight|`` edges. Returns a DataFrame with the three columns, empty when no
    edge qualifies.
    """
    frame = pd.DataFrame(adj_df).copy()
    if frame.empty:
        return pd.DataFrame(columns=["source", "target", "weight"])

    frame.index = pd.Index([str(v).strip() for v in frame.index])
    frame.columns = pd.Index([str(c).strip() for c in frame.columns])
    frame = frame.apply(pd.to_numeric, errors="coerce")

    # Merge duplicate source / target labels; duplicate axis labels break stack.
    if frame.index.has_duplicates:
        frame = frame.groupby(level=0, sort=False).sum()
    if frame.columns.has_duplicates:
        frame = frame.T.groupby(level=0, sort=False).sum().T

    frame.index.name = "source"
    frame.columns.name = "target"

    if target_node:
        node = str(target_node).strip()
        if node not in frame.columns:
            return pd.DataFrame(columns=["source", "target", "weight"])
        frame = frame[[node]]

    # Newer pandas removed stack(dropna=...), so stack() is used and the weight
    # column is dropped uniformly afterwards.
    edges = (
        frame
        .stack()
        .rename("weight")
        .reset_index()
    )
    edges.columns = ["source", "target", "weight"]
    edges["weight"] = pd.to_numeric(edges["weight"], errors="coerce")
    edges = edges.dropna(subset=["weight"])
    edges = edges[edges["source"] != edges["target"]]
    edges = edges[edges["weight"].abs() > 0.0]

    if top_edges is not None and int(top_edges) > 0:
        order = edges["weight"].abs().sort_values(ascending=False).index
        edges = edges.loc[order].head(int(top_edges))

    return edges.reset_index(drop=True)


# Two functions were originally named plot_network (static topology plot and
# interactive HTML). The interactive one is exposed here as
# plot_interactive_network_html; only the name changed.


# Two functions were originally named plot_network (a static topology plot
# and an interactive HTML one). The interactive version is exposed here as
# plot_interactive_network_html; only the name changed.
