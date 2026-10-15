# -*- coding: utf-8 -*-
"""idopNetwork main entry point.

Pipeline
--------
    data import -> quasi-dynamic (pseudo-time) analysis -> [functional clustering]
    -> network reconstruction -> GLMY analysis -> save results and figures

Single-layer vs two-layer is controlled by ``config.NETWORK_LAYER``:

    "single"  build one network over all variables (clustering skipped)
    "two"     cluster first, then build an inter-module macro network and one
              intra-module micro network per module

Naming follows the original convention:

    inter_cluster = macro layer (modules as nodes)
    intra_cluster = micro layer (variables inside a module as nodes)

Every step writes its result tables and figures to ``results/<condition>/``
together with the underlying plotting data, so visualisations can be changed
later without re-running the analysis.
"""

from __future__ import annotations

import os
import sys
import warnings

# Headless figure saving; remove to view figures interactively.
os.environ.setdefault("MPLBACKEND", "Agg")

# Some consoles default to a legacy encoding that cannot print characters such
# as R^2 or beta. Switch stdout/stderr to UTF-8 to avoid aborting the pipeline
# on a print statement.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import numpy as np
import pandas as pd

import config as C
from DataPreprocessor import DataPreprocessor
from QuasiDynamicAnalyzer import (
    DATA_MODE_DYNAMIC,
    DATA_MODE_QUASI,
    fit_uploaded_csv,
    get_power_function_params,
)
from FunctionalClusterer import FunClu, compute_bic_scores
from NetworkReconstructor import StaticIDOPRegressor, select_edges_lasso
from GlmyAnalyzer import (
    bar_counts,
    load_from_to_from_zip,
    list_from_to_members,
    member_display_label,
    run_glmy,
    run_glmy_split,
    suggest_max_x,
)
import NetworkVisualizer as NV
from ResultManager import (
    ResultManager,
    STEP_CLUSTER,
    STEP_CURVE,
    STEP_GLMY,
    STEP_INTERACTIVE,
    STEP_NETWORK,
)


def _banner(step: int, total: int, title: str) -> None:
    print(f"\n{'=' * 74}\n[{step}/{total}] {title}\n{'=' * 74}")


# Clustering is fitted jointly across conditions, so its outputs belong to no
# single condition and live in this shared directory.
JOINT_TAG = "00_joint_clustering"


# ---------------------------------------------------------------------------
# Step 1: data import and quasi-dynamic analysis (curve fitting)
# ---------------------------------------------------------------------------

def step_curve_fitting(rm: ResultManager):
    """Import data, build quasi-dynamic series and fit power laws.

    The power-law figures are drawn later by :func:`plot_power_law_figures`,
    after variable filtering, so that only retained variables are plotted.

    Returns
    -------
    dict
        ``{condition: {"quasi_dynamic", "curve_params", "curve_sample"}}``
    """
    fitted = {}
    for file_path, condition in C.DATA_FILES:
        print(f"\n   condition [ {condition} ]  {file_path}")

        # 1.1 Import raw data (read only, no transform).
        raw = DataPreprocessor.load_data(
            file_path=file_path, n_samples=C.N_SAMPLES, n_features=C.N_FEATURES,
            scaler_type="None", drop_features=C.DROP_FEATURES,
        )
        print(f"      raw data: {raw.shape[0]} rows x {raw.shape[1]} columns")
        rm.save_dataframe(raw, "raw_data.csv", condition, STEP_CURVE)

        content = DataPreprocessor.read_raw_bytes(file_path)

        # 1.2 Quasi-dynamic conversion (one of two routes).
        if C.DATA_MODE == DATA_MODE_DYNAMIC:
            results = fit_uploaded_csv(
                content, data_mode=DATA_MODE_DYNAMIC,
                swt_wavelet=C.SWT_WAVELET, swt_level=C.SWT_LEVEL, swt_alpha=C.SWT_ALPHA,
            )
        else:
            results = fit_uploaded_csv(
                content, data_mode=DATA_MODE_QUASI,
                first_transform=C.FIRST_TRANSFORM, second_transform=C.SECOND_TRANSFORM,
                n_samples=C.N_CURVE_SAMPLES, trim_percent=C.TRIM_PERCENT,
                ignore_zero=C.IGNORE_ZERO,
            )
        # Keep an unfiltered copy: network reconstruction uses the observed
        # quasi-dynamic data.
        results["quasi_dynamic_full"] = results["quasi_dynamic"].copy()
        fitted[condition] = results
        quasi = results["quasi_dynamic"]
        params = results["curve_params"]
        samples = results["curve_sample"]
        n_pos = params["n_positive"]
        r2 = params["r2"] if "r2" in params.columns else pd.Series(np.nan, index=params.index)
        print(f"      quasi-dynamic: {quasi.shape}   params: {params.shape}   "
              f"samples: {samples.shape}")
        print(f"      fit diagnostics: median positive obs = {int(n_pos.median())}  "
              f"median R^2 = {np.nanmedian(r2.to_numpy(float)):.3f}  "
              f"unfittable features = {int(r2.isna().sum())}")

        # 1.3 Persist the plotting inputs.
        rm.save_dataframe(quasi, "quasi_dynamic.csv", condition, STEP_CURVE)
        rm.save_dataframe(params, "curve_params.csv", condition, STEP_CURVE)
        rm.save_dataframe(samples, "curve_sample.csv", condition, STEP_CURVE)
        print("      saved 3 base tables")

    return fitted


def plot_power_law_figures(rm: ResultManager, condition_names, fitted, kept) -> None:
    """Draw publication-style power-law fit figures and the parameter summary.

    ``kept`` are the variables retained by filtering, which are the objects the
    downstream clustering and network steps actually use.
    """
    module_map = {}
    labels_csv = os.path.join(C.RESULT_BASE_DIR, JOINT_TAG, STEP_CLUSTER,
                              "cluster_labels.csv")
    if os.path.isfile(labels_csv):
        lab = pd.read_csv(labels_csv)
        if {"feature", "cluster"}.issubset(lab.columns):
            module_map = dict(zip(lab["feature"].astype(str), lab["cluster"].astype(str)))

    scope = str(getattr(C, "PLOT_FIT_SCOPE", "retained")).lower()
    ncols = max(1, int(C.PLOT_FIT_NCOLS))
    max_rows = max(1, int(C.PLOT_FIT_MAX_ROWS))

    for condition in condition_names:
        res = fitted[condition]
        subsets = []
        if scope in ("retained", "both"):
            subsets.append(("retained", [c for c in res["curve_sample"].columns
                                         if c in set(map(str, kept))]))
        if scope in ("all", "both"):
            subsets.append(("all", list(res["curve_sample"].columns)))

        for tag, subset in subsets:
            if not subset:
                continue

            figs = NV.plot_power_law_fit(
                res["quasi_dynamic"][subset], res["curve_params"].loc[subset],
                ncols=ncols, max_rows=max_rows,
                log_y=bool(getattr(C, "PLOT_FIT_LOGY", False)),
                module_map=module_map or None,
                title="Power-law fit" + (" (log y)" if getattr(C, "PLOT_FIT_LOGY", False) else ""),
                condition_label=condition,
            )
            stem = "power_law_fit" if tag == "retained" else "power_law_fit_all"
            for pi, fig in enumerate(figs, 1):
                nm = (f"{stem}.png" if len(figs) == 1
                      else f"{stem}_p{pi:02d}of{len(figs):02d}.png")
                rm.save_figure(fig, nm, condition, STEP_CURVE)
            print(f"      power-law figures [{tag}]: {len(figs)} page(s), "
                  f"{len(subset)} variables")

        if bool(getattr(C, "PLOT_FIT_SUMMARY", True)):
            sub = [c for c in res["curve_sample"].columns if c in set(map(str, kept))]
            if sub:
                rm.save_figure(
                    NV.plot_power_law_summary(
                        res["curve_params"].loc[sub], module_map=module_map or None,
                        condition_label=condition),
                    "power_law_summary.png", condition, STEP_CURVE)


def apply_feature_filter(condition_names, fitted, rm) -> list:
    """Drop unreliable variables by positive-observation count and fit R^2.

    In zero-inflated data, features that are mostly zero have power curves
    determined by a few points and a near-zero R^2. In clustering they appear as
    an anomalous (very flat or far-off) curve that contaminates the mean curve
    and the module partition, so they are removed before clustering.

    Returns
    -------
    list
        Features retained in all conditions, in ``curve_sample`` column order.
    """
    all_cols = [str(c) for c in fitted[condition_names[0]]["curve_sample"].columns]

    per_cond_pass = {}
    r2_frames = {}
    for condition in condition_names:
        p = fitted[condition]["curve_params"]
        r2 = p["r2"].astype(float) if "r2" in p.columns else pd.Series(np.nan, index=p.index)
        r2_frames[condition] = r2

        n_obs = len(fitted[condition]["quasi_dynamic_full"])
        pct = float(getattr(C, "MIN_POSITIVE_PCT", 0.0) or 0.0)
        abs_min = float(getattr(C, "MIN_POSITIVE", 0) or 0)
        thr_n = max(abs_min, pct * n_obs)
        ok = p["n_positive"].astype(float) > thr_n
        per_cond_pass[condition] = set(p.index[ok])
        print(f"  {condition:10s} samples={n_obs:4d} positive-obs threshold>{thr_n:6.1f}  "
              f"pass {len(per_cond_pass[condition]):4d} / {len(p)}")

    usable = set.intersection(*per_cond_pass.values()) if C.FILTER_CONSENSUS \
        else set().union(*per_cond_pass.values())
    usable = [c for c in all_cols if c in usable]

    # Rank by the global mean R^2 so that all conditions share exactly the same
    # feature set (intersecting per-condition quantiles would remove almost
    # everything).
    if C.MIN_FIT_R2_PCT is not None and usable:
        r2_mat = pd.DataFrame({c: r2_frames[c] for c in condition_names}).loc[usable]
        mean_r2 = r2_mat.mean(axis=1, skipna=True)
        thr = np.nanpercentile(mean_r2.to_numpy(float), float(C.MIN_FIT_R2_PCT))
        keep_mask = np.isfinite(mean_r2) & (mean_r2 >= thr)
        abs_min = getattr(C, "MIN_FIT_R2_ABS", None)
        if abs_min is not None:
            keep_mask &= (mean_r2 >= float(abs_min))
        kept_set = set(mean_r2.index[keep_mask])
        extra = "" if abs_min is None else f", and R^2 >= {float(abs_min):.4g}"
        print(f"  global mean R^2 threshold = {thr:+.4f} "
              f"(drops the worst {C.MIN_FIT_R2_PCT}%{extra})")
    else:
        kept_set = set(usable)

    kept_list = [c for c in all_cols if c in kept_set]
    print(f"  keeping {len(kept_list)} variables (identical set across conditions), "
          f"dropping {len(all_cols) - len(kept_list)}")

    # Set dropped variables to NaN so clustering excludes them automatically.
    # curve_params is indexed by feature while curve_sample / quasi_dynamic are
    # column-indexed, so the two are handled separately.
    for condition in condition_names:
        res = fitted[condition]

        for key in ("quasi_dynamic", "curve_sample"):
            table = res[key]
            miss = [c for c in table.columns if c not in kept_set]
            if miss:
                table.loc[:, miss] = np.nan

        params = res["curve_params"]
        miss_rows = [c for c in params.index if c not in kept_set]
        if miss_rows:
            for col in ("c", "beta", "a", "b", "r2"):
                if col in params.columns:
                    params.loc[miss_rows, col] = np.nan

        rm.save_dataframe(
            pd.DataFrame({"feature": all_cols,
                          "kept": [c in kept_set for c in all_cols],
                          "r2": [float(r2_frames[condition].get(c, np.nan))
                                 for c in all_cols]}),
            "feature_filter.csv", condition, STEP_CURVE)
    return kept_list


# ---------------------------------------------------------------------------
# Step 2: functional clustering (two-layer mode only)
# ---------------------------------------------------------------------------

def rebalance_modules(model, labels: np.ndarray, n_features: int):
    """Reassign members so that module sizes are comparable.

    The EM partition on this dataset produces one module holding roughly 70% of
    the members plus several tiny ones. When comparing Betti numbers across
    modules, beta_0 and beta_1 are dominated by node and edge counts, so very
    unequal sizes are not comparable.

    The module centres ``params_mu`` (log scale, dimensionless) are used as
    module positions: repeatedly take the least typical member of the largest
    module and move it to the smallest module, until all sizes fall within
    ``[mean - 1, mean + 1]``. If ``MODULE_MAX_SHARE`` is a number, a cap-only
    strategy is used instead.

    Returns
    -------
    (labels, n_moved)
    """
    K = int(model.n_components)
    labels = np.asarray(labels, dtype=int).copy()
    mu = model.params_mu.detach().cpu().numpy()          # (K, L, 2)
    L = mu.shape[1]
    # Module position: mean log c and mean beta.
    centers = np.column_stack([
        np.log(np.clip(mu[:, :, 0], 1e-300, None)).mean(axis=1),
        mu[:, :, 1].mean(axis=1),
    ])
    sd = centers.std(axis=0)
    sd[sd < 1e-12] = 1.0
    mean0 = centers.mean(axis=0)
    centers_std = (centers - mean0) / sd

    # Member positions in the same space.
    member_pos = np.zeros((n_features, 2), dtype=float)
    for ell in range(L):
        c_ell = np.clip(mu[labels, ell, 0], 1e-300, None)
        member_pos[:, 0] += np.log(c_ell)
        member_pos[:, 1] += mu[labels, ell, 1]
    member_pos /= max(L, 1)
    member_pos_std = (member_pos - mean0) / sd

    share = C.MODULE_MAX_SHARE
    if share is None:
        # Equal-size mode: target sizes differ by at most one member.
        target_hi = int(np.ceil(n_features / K))
    else:
        target_hi = max(1, int(np.floor(float(share) * n_features)))
        if target_hi * K < n_features:
            target_hi = int(np.ceil(n_features / K))

    moved = 0
    for _ in range(n_features * 3):            # bounded loop
        sizes = np.bincount(labels, minlength=K)
        over = np.where(sizes > target_hi)[0]
        if over.size == 0:
            break
        src = int(over[np.argmax(sizes[over])])
        members = np.where(labels == src)[0]
        dist = np.linalg.norm(member_pos_std[members] - centers_std[src], axis=1)
        idx = int(members[int(np.argmax(dist))])        # least typical member
        if share is None:
            target = int(np.argmin(sizes))
            if target == src:
                break
        else:
            order = np.argsort(np.linalg.norm(centers_std - member_pos_std[idx], axis=1))
            target = next((int(t) for t in order
                           if t != src and sizes[int(t)] < target_hi), None)
            if target is None:
                break
        labels[idx] = int(target)
        moved += 1

    return labels, moved

def step_functional_clustering(rm: ResultManager, condition_names, fitted):
    """Joint clustering across conditions; saves labels, BIC scan and figures.

    Notes
    -----
    Clustering is performed jointly on all conditions. Labels correspond
    positionally to ``model.common_cols`` (``labels[i]`` <-> ``common_cols[i]``).
    """
    curve_samples = [fitted[name]["curve_sample"] for name in condition_names]
    parameter_sets = [fitted[name]["curve_params"] for name in condition_names]

    # 2.1 Optional BIC scan to select K.
    bic = None
    if C.CLUSTER_USE_BIC:
        k_max = min(C.CLUSTER_K_MAX, min(s.shape[1] for s in curve_samples))
        if k_max >= C.CLUSTER_K_MIN:
            print(f"   BIC scan K = {C.CLUSTER_K_MIN} .. {k_max} ...")
            bic = compute_bic_scores(
                curve_samples, k_min=C.CLUSTER_K_MIN, k_max=k_max,
                parameter_data=parameter_sets, random_state=C.RANDOM_STATE,
            )
            print(bic[["K", "BIC", "SI", "converged", "fit_successes"]].to_string(index=False))
            valid = bic.dropna(subset=["BIC"])
            best_k = int(valid.loc[valid["BIC"].idxmin(), "K"]) if not valid.empty else C.CLUSTER_K
        else:
            best_k = C.CLUSTER_K
    else:
        best_k = C.CLUSTER_K
    print(f"   using K = {best_k}")

    # 2.2 EM fit
    model = FunClu(
        n_components=int(best_k), max_iter=C.CLUSTER_MAX_ITER,
        tol=C.CLUSTER_TOL, random_state=C.RANDOM_STATE,
    )
    model.fit(curve_samples, parameter_data=parameter_sets)
    labels = model.labels.detach().cpu().numpy().astype(int)
    common_cols = list(model.common_cols)

    # 2.2b Optional module balancing: reassign members only, no refitting.
    if C.MODULE_BALANCE:
        labels, moved = rebalance_modules(model, labels, len(common_cols))
        if moved:
            print(f"   module balancing: moved {moved} members")
        print(f"   module sizes after balancing: "
              f"{list(np.bincount(labels, minlength=int(best_k)))}")

    labels_df = pd.DataFrame({
        "feature": common_cols,
        "cluster_id": labels + 1,
        "cluster": [f"{C.TWO_LAYER_MODULE_PREFIX}{k + 1}" for k in labels],
    })
    cluster_sizes = (
        labels_df.groupby(["cluster_id", "cluster"], as_index=False)
        .size().rename(columns={"size": "n_features"}).sort_values("cluster_id")
    )
    print(f"   BIC={model.bic:.6f}  loglik={model.log_likelihood:.6f}  "
          f"iterations={model.n_iter_run}  converged={model.converged}")
    print("   " + cluster_sizes.to_string(index=False).replace("\n", "\n   "))

    # 2.3 Save clustering results (joint table plus per-condition copies).
    for condition in condition_names:
        rm.save_dataframe(labels_df, "cluster_labels.csv", condition, STEP_CLUSTER)
        rm.save_dataframe(cluster_sizes, "cluster_sizes.csv", condition, STEP_CLUSTER)
        if bic is not None:
            rm.save_dataframe(bic, "bic_scores.csv", condition, STEP_CLUSTER)
    rm.save_dataframe(labels_df, "cluster_labels.csv", JOINT_TAG, STEP_CLUSTER)
    rm.save_dataframe(cluster_sizes, "cluster_sizes.csv", JOINT_TAG, STEP_CLUSTER)
    if bic is not None:
        rm.save_dataframe(bic, "bic_scores.csv", JOINT_TAG, STEP_CLUSTER)

    # 2.4 Clustering figures
    scatter_all = [fitted[name]["quasi_dynamic"].loc[:, common_cols]
                   for name in condition_names]
    curve_all = [fitted[name]["curve_sample"].loc[:, common_cols]
                 for name in condition_names]

    fig = NV.plot_cluster_profiles(
        data_scatter=scatter_all, data_curve=curve_all,
        labels=labels, common_cols=common_cols, n_components=int(best_k),
        condition_labels=list(condition_names), layout="combined", n_cols=1,
        title="Cluster profiles (joint clustering, all conditions)",
    )
    rm.save_figure(fig, "cluster_profiles_joint.png", JOINT_TAG, STEP_CLUSTER)

    figures = NV.plot_cluster_profiles_per_cluster(
        data_scatter=scatter_all, data_curve=curve_all,
        labels=labels, common_cols=common_cols, n_components=int(best_k),
        condition_labels=list(condition_names),
    )
    stitched = NV.stitch_cluster_figures(figures, n_cols=min(3, int(best_k)))
    rm.save_figure(stitched, "cluster_profiles_per_module_joint.png", JOINT_TAG, STEP_CLUSTER)

    for condition in condition_names:
        fig = NV.plot_cluster_profiles(
            data_scatter=[fitted[condition]["quasi_dynamic"].loc[:, common_cols]],
            data_curve=[fitted[condition]["curve_sample"].loc[:, common_cols]],
            labels=labels, common_cols=common_cols, n_components=int(best_k),
            condition_labels=[condition], layout="combined", n_cols=1,
        )
        rm.save_figure(fig, "cluster_profiles.png", condition, STEP_CLUSTER)

        figures = NV.plot_cluster_profiles_per_cluster(
            data_scatter=[fitted[condition]["quasi_dynamic"].loc[:, common_cols]],
            data_curve=[fitted[condition]["curve_sample"].loc[:, common_cols]],
            labels=labels, common_cols=common_cols, n_components=int(best_k),
            condition_labels=[condition],
        )
        stitched = NV.stitch_cluster_figures(figures, n_cols=min(3, int(best_k)))
        rm.save_figure(stitched, "cluster_profiles_per_module.png",
                       condition, STEP_CLUSTER)

    if bic is not None:
        rm.save_figure(NV.plot_bic_elbow(bic_results=bic), "bic_elbow.png",
                       JOINT_TAG, STEP_CLUSTER)

    # 2.6 Clustering diagnostic grid: compare the KMeans initial partition with
    # the final EM partition. Only the member curves and the cluster mean curves
    # are drawn; a reconstructed power-law line is deliberately omitted because
    # params_mu stores (c, beta) in standardised log-time, and drawing it in the
    # raw parameter space biases it high by 5-17% and makes it look as if the
    # mean deviates from the members.
    try:
        import torch

        kernel = model.kernel_

        X_list = [torch.as_tensor(
            fitted[name]["curve_sample"].loc[:, common_cols].to_numpy(float).T)
            for name in condition_names]
        times_list = [np.asarray(fitted[name]["curve_sample"].index, float)
                      for name in condition_names]

        # kernel.condition_names_ holds string indices "0"/"1"/...; the real
        # condition names are used for the panel titles.
        cond_keys = list(kernel.condition_names_)
        cond_keys = cond_keys if cond_keys else [str(i) for i in range(len(condition_names))]

        if hasattr(kernel, "initial_mean_curves_"):
            fig = NV.plot_cluster_fit_grid(
                X_list=X_list, times_list=times_list,
                labels=kernel.initial_labels_,
                mean_curves=kernel.initial_mean_curves_,
                params_mu=kernel.initial_params_mu_,
                condition_names=cond_keys,
                condition_labels=list(condition_names),
                sharey=C.CLUSTER_FIG_SHAREY,
                normalize=C.CLUSTER_FIG_NORMALIZE,
            )
            rm.save_figure(fig, "cluster_fit_kmeans.png", JOINT_TAG, STEP_CLUSTER)

        if hasattr(kernel, "mean_curves_"):
            fig = NV.plot_cluster_fit_grid(
                X_list=X_list, times_list=times_list,
                labels=labels,
                mean_curves=kernel.mean_curves_,
                params_mu=kernel.params_mu_,
                condition_names=cond_keys,
                condition_labels=list(condition_names),
                sharey=C.CLUSTER_FIG_SHAREY,
                normalize=C.CLUSTER_FIG_NORMALIZE,
            )
            rm.save_figure(fig, "cluster_fit_em.png", JOINT_TAG, STEP_CLUSTER)
        print("   saved: clustering diagnostic grid (member curves + cluster means)")
    except Exception as exc:
        print(f"   (clustering diagnostic grid skipped: {type(exc).__name__}: {exc})")

    # 2.5 Save the clustering inputs for redrawing.
    for name in condition_names:
        rm.save_dataframe(
            fitted[name]["curve_sample"].loc[:, common_cols],
            "curve_sample_common.csv", name, STEP_CLUSTER)
        rm.save_dataframe(
            fitted[name]["quasi_dynamic"].loc[:, common_cols],
            "quasi_dynamic_common.csv", name, STEP_CLUSTER)

    return model, labels_df, common_cols


# ---------------------------------------------------------------------------
# Step 3: network reconstruction (single or two layers)
# ---------------------------------------------------------------------------

def _fit_one_network(condition, tag, design_df, response_df, rm, step, subdir=""):
    """Fit one network from a (design curves, response curves) pair and save it.

    Uses the same convention as the original implementation:
        model     = StaticIDOPRegressor (multi-window LASSO edge selector)
        fit       = model.fit(design_df, response_df,
                              power_function_params=get_power_function_params(response_df))
        aggregate = model.adjacency_matrix(design_df, aggregation="mean")

    Returns
    -------
    dict
        ``{"adj_df", "model", "effects", "predicted", "design_X"}``, or None if
        there are too few curves.
    """
    if design_df.shape[1] < 2:
        print(f"     [{tag}] fewer than 2 curves; skipping network fit")
        return None

    # Response columns are aligned to the design columns.
    response_df = response_df.loc[:, list(design_df.columns)]
    params = get_power_function_params(response_df)

    supports = select_edges_lasso(
        response_df, alpha=C.LASSO_ALPHA, k=C.LASSO_WINDOWS, threshold=C.LASSO_THRESHOLD)

    model = StaticIDOPRegressor(
        max_order=C.BASIS_MAX_ORDER, mix=0.5, fix_mix=False,
        nonneg_self=C.NONNEG_SELF, max_interactions=C.MAX_INTERACTIONS,
        adaptive_weights=False,
        lasso_alpha=C.LASSO_ALPHA, lasso_windows=C.LASSO_WINDOWS,
        lasso_threshold=C.LASSO_THRESHOLD,
    )
    model.fit(design_df, response_df, power_function_params=params)

    design_X = model._design(design_df)
    predicted = model.predict(design_df)
    effects = model.effect(design_df)
    adj = model.adjacency_matrix(design_df, aggregation="mean")
    node_names = [str(c) for c in design_df.columns]

    # Self-check: intercept + sum(effects) == predicted.
    intercept = model.coef_.loc["intercept"].to_numpy(dtype=float)
    total = np.column_stack([
        intercept[j] + sum(effects[j][c].to_numpy(float) for c in effects[j].columns)
        for j in range(len(effects))
    ])
    ok = np.allclose(total, predicted.to_numpy(float), atol=1e-8)
    print(f"     [{tag}] nodes={len(node_names)}  design={design_X.shape}  "
          f"mse={model.mse_:.6g}  nonzero coefs={int((model.coef_.to_numpy() != 0).sum())}  "
          f"identity={ok}")

    step_dir = os.path.join(step, subdir) if subdir else step

    rm.save_matrix(adj.to_numpy(float), "adjacency_matrix.csv",
                   list(adj.index), list(adj.columns), condition, step_dir)
    rm.save_edge_list(adj.to_numpy(float), node_names, "edge_list.csv", condition, step_dir)
    rm.save_dataframe(design_X, "design_matrix.csv", condition, step_dir)
    rm.save_dataframe(predicted, "predicted.csv", condition, step_dir)
    rm.save_dataframe(response_df, "response.csv", condition, step_dir)
    rm.save_dataframe(params, "curve_params_used.csv", condition, step_dir)
    rm.save_dataframe(supports, "edge_supports.csv", condition, step_dir)
    for j, effect in enumerate(effects):
        rm.save_dataframe(effect, f"effect_{node_names[j]}.csv", condition, step_dir)

    rm.save_figure(NV.plot_network(adj), "network_topology.png", condition, step_dir)
    rm.save_figure(NV.plot_adjusted_matrix_heatmap(adj), "adjacency_heatmap.png",
                   condition, step_dir)
    # Same convention as the original plot_effect:
    #   quasi_dynamic_df = response table, curve_df = predicted (not curve_sample)
    # _effect_label_mode writes total/self/variable names at the end of each
    # curve, with matching colours and unabbreviated names.
    rm.save_figure(
        NV._effect_label_mode(
            response_df, predicted, effects,
            intercept=model.coef_.loc["intercept"],
            plot_ncols=int(getattr(C, "EFFECT_NCOLS", 3)),
            panel_width_in=float(getattr(C, "EFFECT_PANEL_WIDTH", 5.5)),
            label_fontsize=float(getattr(C, "EFFECT_LABEL_FONTSIZE", 7.0)),
            var_label_align=str(getattr(C, "EFFECT_VAR_LABEL_ALIGN", "left")),
            dpi=int(getattr(C, "EFFECT_DPI", 150)),
        ),
        "effect_decomposition.png", condition, step_dir)

    if C.ENABLE_INTERACTIVE_HTML:
        edges = NV.network_edges_from_adjacency(adj)
        if edges is not None and not edges.empty:
            html_dir = rm.get_dir(condition, os.path.join(STEP_INTERACTIVE, subdir)
                                  if subdir else STEP_INTERACTIVE)
            NV.plot_interactive_network_html(
                edges, os.path.join(html_dir, "network_interactive.html"),
                group_name=tag, group_dir=html_dir)

    return {"adj_df": adj, "model": model, "effects": effects,
            "predicted": predicted, "design_X": design_X}


def step_network_reconstruction(rm: ResultManager, condition_names, fitted,
                                clustering=None):
    """Reconstruct networks (single or two layers).

    Returns ``{condition: {"single"|"inter"|"intra": ...}}``.

    Two-layer conventions:
    * macro (inter_cluster) design matrix = the cluster centre curves from
      ``FunClu.get_cluster_curves()`` (columns ``M1..Mk``), not member means;
    * macro response = arithmetic mean of the members' ``quasi_dynamic`` columns;
    * micro (intra_cluster) = one network per module using its member columns;
    * modules with fewer than two members are skipped.
    """
    results = {}
    two_layer = C.NETWORK_LAYER == "two"

    for condition in condition_names:
        print(f"\n   condition [ {condition} ]")
        results[condition] = {}
        curve_sample = fitted[condition]["curve_sample"]
        # Network reconstruction uses the unfiltered observed quasi-dynamic
        # table, otherwise filtering to NaN would contaminate the module means.
        quasi = fitted[condition].get(
            "quasi_dynamic_full", fitted[condition]["quasi_dynamic"])
        params = fitted[condition]["curve_params"]

        # -- single layer ---------------------------------------------------
        if not two_layer:
            net = _fit_one_network(condition, "single_layer", curve_sample, quasi,
                                   rm, STEP_NETWORK)
            if net is not None:
                results[condition]["single"] = net
            continue

        # -- two layers: macro (inter_cluster) ------------------------------
        if clustering is None:
            raise RuntimeError("two-layer mode requires functional clustering first")
        model, labels_df, common_cols = clustering
        common_cols = list(common_cols)

        cond_key = str(condition_names.index(condition))
        kernel = model.kernel_
        key = cond_key if cond_key in kernel.mean_curves_ else kernel.condition_names_[0]
        center_frame = kernel.mean_curves_[key]
        center_df = pd.DataFrame(
            center_frame.to_numpy(float),
            index=center_frame.index,
            columns=[f"{C.TWO_LAYER_MODULE_PREFIX}{k + 1}"
                     for k in range(model.n_components)],
        )

        center_response = pd.DataFrame(index=quasi.index)
        for k in range(model.n_components):
            module = f"{C.TWO_LAYER_MODULE_PREFIX}{k + 1}"
            member_cols = labels_df.loc[
                labels_df["cluster_id"] == k + 1, "feature"].tolist()
            member_cols = [c for c in member_cols if c in quasi.columns]
            if member_cols:
                center_response[module] = quasi.loc[:, member_cols].mean(axis=1)
        center_response.index.name = quasi.index.name or "time"

        rm.save_dataframe(center_df, "macro_center_curve_sample.csv",
                          condition, STEP_NETWORK)
        rm.save_dataframe(center_response, "macro_center_quasi_dynamic.csv",
                          condition, STEP_NETWORK)

        net = _fit_one_network(condition, "inter_cluster", center_df, center_response,
                               rm, STEP_NETWORK, subdir="inter_cluster")
        if net is not None:
            results[condition]["inter"] = net

        # -- two layers: micro (intra_cluster, per module) ------------------
        for k in range(model.n_components):
            module = f"{C.TWO_LAYER_MODULE_PREFIX}{k + 1}"
            member_cols = labels_df.loc[
                labels_df["cluster_id"] == k + 1, "feature"].tolist()
            member_cols = [c for c in member_cols if c in curve_sample.columns]
            if len(member_cols) < 2:
                print(f"     [{module}] {len(member_cols)} member variable(s) (<2); "
                      f"skipping micro network")
                continue
            member_design = curve_sample.loc[:, member_cols]
            member_response = quasi.loc[:, member_cols]
            micro = _fit_one_network(condition, module, member_design, member_response,
                                     rm, STEP_NETWORK, subdir=f"intra_cluster/{module}")
            if micro is not None:
                results[condition].setdefault("intra", {})[module] = micro

    return results


# ---------------------------------------------------------------------------
# Step 4: GLMY analysis
# ---------------------------------------------------------------------------

def _glmy_one(rm, tag, network, condition, step_dir):
    """Run GLMY analysis on one network and save the results and figures."""
    adj = network["adj_df"]
    from_to = pd.DataFrame([
        {"from": str(s), "to": str(t), "weight": float(adj.loc[s, t]),
         "type": "+" if float(adj.loc[s, t]) > 0 else "-"}
        for s in adj.index for t in adj.columns
        if s != t and float(adj.loc[s, t]) != 0.0
    ], columns=["from", "to", "weight", "type"])
    if from_to.empty:
        print(f"     [{tag}] no nonzero cross edges; skipping GLMY")
        return None

    result = run_glmy(from_to)
    split = run_glmy_split(from_to)
    homology = result["homology"]
    counts = bar_counts(homology)
    print(f"     [{tag}] edges={len(from_to)}  weight offset={result['weight_offset']:.6g}  "
          f"Betti={counts}  sign split={split['edge_counts']}")

    rm.save_dataframe(from_to, f"from_to_{tag}.csv", condition, step_dir, index=False)
    rm.save_json({k: v for k, v in homology.items()}, f"homology_{tag}.json",
                 condition, step_dir)
    rm.save_json(split["homologies"], f"homology_split_{tag}.json", condition, step_dir)
    rm.save_json(counts, f"betti_counts_{tag}.json", condition, step_dir)

    max_x = suggest_max_x(from_to)
    rm.save_figure(NV.plot_glmy_barcode(homology, max_x=max_x),
                   f"barcode_{tag}.png", condition, step_dir)
    rm.save_figure(NV.plot_glmy_barcode_combined(homology, split["homologies"], max_x=max_x),
                   f"barcode_combined_{tag}.png", condition, step_dir)
    rm.save_figure(NV.plot_betti_counts(homology, title=f"Betti numbers - {tag}"),
                   f"betti_numbers_{tag}.png", condition, step_dir)
    return {"from_to": from_to, "homology": homology, "split": split, "counts": counts}


def step_glmy_analysis(rm: ResultManager, condition_names, networks):
    """Run GLMY analysis on every network of every condition."""
    out = {}
    for condition in condition_names:
        print(f"\n   condition [ {condition} ]")
        out[condition] = {}
        nets = networks[condition]

        if "single" in nets:
            res = _glmy_one(rm, "single_layer", nets["single"], condition, STEP_GLMY)
            if res:
                out[condition]["single_layer"] = res

        if "inter" in nets:
            res = _glmy_one(rm, "inter_cluster", nets["inter"], condition, STEP_GLMY)
            if res:
                out[condition]["inter_cluster"] = res

        for module, micro in (nets.get("intra") or {}).items():
            res = _glmy_one(rm, f"intra_cluster_{module}", micro, condition,
                            os.path.join(STEP_GLMY, "modules"))
            if res:
                out[condition].setdefault("intra_cluster", {})[module] = res
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)

    font = NV.setup_matplotlib(C.FONT_CANDIDATES)
    print(f"CJK font: {font or 'not found (CJK glyphs will show as boxes; '
                               'numerical results are unaffected)'}")

    two_layer = C.NETWORK_LAYER == "two"
    rm = ResultManager(C.RESULT_BASE_DIR, dpi=C.FIGURE_DPI)
    condition_names = [name for _path, name in C.DATA_FILES]

    total = 4
    print(f"network layer: {'two (inter_cluster + intra_cluster)' if two_layer else 'single'}"
          f"   conditions: {condition_names}")
    print(f"zero handling: ignore_zero={C.IGNORE_ZERO}   "
          f"filtering: n_positive>{C.MIN_POSITIVE}, "
          f"R^2 keeps the best {100 - int(C.MIN_FIT_R2_PCT or 0)}% "
          f"(drops the worst {int(C.MIN_FIT_R2_PCT or 0)}%), "
          f"{'intersection of conditions' if C.FILTER_CONSENSUS else 'union'}")
    if C.MODULE_BALANCE:
        mode = ("equal-size mode" if C.MODULE_MAX_SHARE is None
                else f"cap mode, module limit {float(C.MODULE_MAX_SHARE):.0%}")
        print(f"module balancing: True ({mode})")
    else:
        print("module balancing: False (raw EM partition)")

    _banner(1, total, "Data import + quasi-dynamic analysis (power-law fitting)")
    fitted = step_curve_fitting(rm)

    print("\n   variable filtering (drop zero-inflated / poorly fitted variables):")
    kept = apply_feature_filter(condition_names, fitted, rm)

    clustering = None
    if two_layer:
        _banner(2, total, "Functional clustering (FunClu EM)")
        clustering = step_functional_clustering(rm, condition_names, fitted)
    else:
        print("\n(single-layer mode: functional clustering skipped)")

    # The power-law figures must come after clustering because they are coloured
    # by module, which is written by the clustering step.
    print("\n   power-law figures (coloured by module):")
    plot_power_law_figures(rm, condition_names, fitted, kept)

    _banner(3, total, "Network reconstruction (IDOP effect decomposition)")
    networks = step_network_reconstruction(rm, condition_names, fitted, clustering)

    _banner(4, total, "GLMY network analysis (persistent path homology)")
    if C.ENABLE_GLMY:
        glmy = step_glmy_analysis(rm, condition_names, networks)
    else:
        glmy = {}
        print("(skipped: config.ENABLE_GLMY=False)")

    # Run summary for reproducibility.
    rm.save_json({
        "network_layer": C.NETWORK_LAYER,
        "data_mode": C.DATA_MODE,
        "conditions": condition_names,
        "first_transform": C.FIRST_TRANSFORM,
        "second_transform": C.SECOND_TRANSFORM,
        "trim_percent": C.TRIM_PERCENT,
        "n_curve_samples": C.N_CURVE_SAMPLES,
        "basis_max_order": C.BASIS_MAX_ORDER,
        "lasso_alpha": C.LASSO_ALPHA,
        "lasso_windows": C.LASSO_WINDOWS,
        "lasso_threshold": C.LASSO_THRESHOLD,
        "nonneg_self": C.NONNEG_SELF,
        "max_interactions": C.MAX_INTERACTIONS,
        "glmy_dim": C.GLMY_DIM,
        "cluster_k": C.CLUSTER_K,
    }, "run_config.json")

    print(f"\n{'=' * 74}\nDone. Results written under '{C.RESULT_BASE_DIR}/'.")
    for condition in condition_names:
        nets = networks.get(condition, {})
        n_single = 1 if "single" in nets else 0
        n_inter = 1 if "inter" in nets else 0
        n_intra = len(nets.get("intra") or {})
        print(f"  {condition}: single={n_single}  macro={n_inter}  micro={n_intra}")


if __name__ == "__main__":
    main()
