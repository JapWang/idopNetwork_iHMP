# -*- coding: utf-8 -*-
"""Functional clustering (FunClu).

Power-mean EM Gaussian-mixture clustering kernel, the public export interface,
BIC scan / silhouette diagnostics and the clustering figures (initialisation
grid, cluster profiles, BIC elbow). Class names, function names, parameter
defaults and all computations are unchanged from the original implementation.
"""
from __future__ import annotations

import math
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans, MiniBatchKMeans

from QuasiDynamicAnalyzer import power_fitting

class _V4FunCluKernel:
    """Finite-mixture clustering of fitted quasi-dynamic curves."""

    def __init__(
        self,
        n_components: int,
        max_iter: int = 1000,
        inner_max_iter: int = 1,
        tol: float = 1e-4, # outer-iteration convergence tolerance
        inner_tol: float = 1e-3, # inner-iteration convergence tolerance
        beta_bounds: tuple[float, float] = (-6.0, 6.0),
        beta_xtol: float = 1e-4, # beta search tolerance
        phi_bound: float = 0.90, # bound on phi
        phi_ridge: float = 0.30, # ridge term on phi
        tightness_weight: float = 0.0, # looser clustering
        n_init: int = 10,
        use_mini_batch: bool = True, # True False
        balance_conditions: bool = True,
        compute_device: str = "auto",
        random_state: int = 0,
        progress: bool = True,
    ):
        self.n_components = n_components
        self.max_iter = max_iter
        self.inner_max_iter = inner_max_iter
        self.tol = tol
        self.inner_tol = inner_tol
        self.beta_bounds = beta_bounds
        self.beta_xtol = beta_xtol
        self.phi_bound = phi_bound
        self.phi_ridge = phi_ridge
        self.tightness_weight = tightness_weight
        self.n_init = n_init
        self.use_mini_batch = use_mini_batch
        self.balance_conditions = balance_conditions
        self.compute_device = compute_device
        self.random_state = random_state
        self.progress = progress

    def _prepare_inputs(self, sample_sets, parameter_sets):
        self.condition_names_ = list(sample_sets)
        first = self.condition_names_[0]
        original_features = pd.Index(sample_sets[first].columns)
        common_features = original_features

        for condition in self.condition_names_:
            common_features = common_features.intersection(
                sample_sets[condition].columns,
                sort=False,
            )
            common_features = common_features.intersection(
                parameter_sets[condition].index,
                sort=False,
            )

        valid = np.ones(len(common_features), dtype=bool)
        for condition in self.condition_names_:
            curves = sample_sets[condition].loc[:, common_features].to_numpy(float)
            params = parameter_sets[condition].loc[common_features, ["c", "beta"]].to_numpy(float)
            valid &= np.isfinite(curves).all(axis=0)
            valid &= np.isfinite(params).all(axis=1) & (params[:, 0] > 0)

        self.feature_names_ = common_features[valid]
        self.excluded_features_ = original_features.difference(self.feature_names_, sort=False)
        self.sample_sets_ = {
            condition: sample_sets[condition].loc[:, self.feature_names_].copy()
            for condition in self.condition_names_
        }
        self.parameter_sets_ = {
            condition: parameter_sets[condition].loc[self.feature_names_].copy()
            for condition in self.condition_names_
        }
        self.n_features_ = len(self.feature_names_)
        self.n_conditions_ = len(self.condition_names_)

    def _combine_sample_sets(self):
        self.curves_ = np.concatenate(
            [
                self.sample_sets_[condition].to_numpy(float).T
                for condition in self.condition_names_
            ],
            axis=1,
        )

    def _build_initialization_data(self):
        blocks = []
        for condition in self.condition_names_:
            params = self.parameter_sets_[condition]
            block = np.column_stack((
                np.log(params["c"].to_numpy(float)),
                params["beta"].to_numpy(float),
            ))
            center = np.median(block, axis=0)
            q25, q75 = np.quantile(block, [0.25, 0.75], axis=0)
            scale = (q75 - q25) / 1.349
            scale = np.where(scale > 1e-12, scale, block.std(axis=0))
            scale = np.where(scale > 1e-12, scale, 1.0)
            blocks.append((block - center) / scale)
        self.initialization_data_ = np.concatenate(blocks, axis=1)

    def _initialize_clusters(self):
        method = MiniBatchKMeans if self.use_mini_batch else KMeans
        self.clustering_model_ = method(
            n_clusters=self.n_components,
            init="k-means++",
            n_init=self.n_init,
            random_state=self.random_state,
        )
        self.initial_labels_ = self.clustering_model_.fit_predict(
            self.initialization_data_
        )
        self.initial_cluster_sizes_ = np.bincount(
            self.initial_labels_,
            minlength=self.n_components,
        )

    def _initialize_weights(self):
        self.initial_weights_ = (
            self.initial_cluster_sizes_
            / self.initial_cluster_sizes_.sum()
        )

    def _initialize_mean_curves(self):
        membership = np.eye(self.n_components)[self.initial_labels_]
        columns = [f"cluster_{k}" for k in range(self.n_components)]
        self.initial_mean_curves_ = {}
        for condition, samples in self.sample_sets_.items():
            means = (
                samples.to_numpy(float)
                @ membership
                / np.maximum(self.initial_cluster_sizes_, 1)
            )
            # Identical curves can leave KMeans clusters empty. Keep finite
            # initial means so zero-weight components do not poison the E-step.
            empty = self.initial_cluster_sizes_ == 0
            if empty.any():
                means[:, empty] = samples.to_numpy(float).mean(axis=1, keepdims=True)
            self.initial_mean_curves_[condition] = pd.DataFrame(
                means,
                index=samples.index,
                columns=columns,
            )

    def _initialize_mean_parameters(self):
        self.initial_params_mu_ = np.empty(
            (self.n_components, self.n_conditions_, 2)
        )
        for ell, curves in enumerate(self.initial_mean_curves_.values()):
            params, _ = power_fitting(curves, n_samples=len(curves))
            self.initial_params_mu_[:, ell] = params[["c", "beta"]].to_numpy(float)

    def _initialize_covariance_parameters(self):
        self.initial_params_cov_ = np.tile(
            [0.01, 0.95],
            (self.n_components, self.n_conditions_, 1),
        )

    def _initialize_responsibilities(self):
        self.initial_responsibilities_ = np.eye(self.n_components)[
            self.initial_labels_
        ]

    def _initialize(self):
        self._combine_sample_sets()
        self._build_initialization_data()
        self._initialize_clusters()
        self._initialize_weights()
        self._initialize_mean_curves()
        self._initialize_mean_parameters()
        self._initialize_covariance_parameters()
        self._initialize_responsibilities()

    def _set_device(self):
        if self.compute_device == "auto":
            name = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            name = self.compute_device
        self.device_ = torch.device(name)
        self.dtype_ = torch.float32 if self.device_.type == "cuda" else torch.float64

    def _prepare_solver_data(self):
        self._set_device()
        dtype, device = self.dtype_, self.device_
        self._prepared = []
        curve_blocks, statistic_blocks, self._curve_slices = [], [], []
        start = 0

        for condition in self.condition_names_:
            samples = self.sample_sets_[condition]
            time = torch.as_tensor(
                np.array(samples.index, dtype=float, copy=True),
                dtype=dtype,
                device=device,
            )
            curves = torch.as_tensor(
                np.array(samples.to_numpy(float).T, dtype=float, order="C", copy=True),
                dtype=dtype,
                device=device,
            )
            log_time = torch.log(time)
            center = log_time.mean()
            scale = torch.clamp(log_time.std(unbiased=False), min=1e-8)
            z = (log_time - center) / scale
            sum_y2 = torch.sum(curves.square(), dim=1)
            sum_y2_previous = torch.sum(curves[:, :-1].square(), dim=1)
            sum_cross = torch.sum(curves[:, 1:] * curves[:, :-1], dim=1)
            n_time = curves.shape[1]

            self._prepared.append({
                "curves": curves,
                "z": z,
                "sum_y2": sum_y2,
                "sum_y2_previous": sum_y2_previous,
                "sum_cross": sum_cross,
                "n_time": n_time,
            })
            curve_blocks.append(curves)
            statistic_blocks.extend([
                sum_y2[:, None],
                sum_y2_previous[:, None],
                sum_cross[:, None],
            ])
            self._curve_slices.append(slice(start, start + n_time))
            start += n_time

        self._combined_curves = torch.cat(curve_blocks, dim=1).contiguous()
        self._combined_statistics = torch.cat(statistic_blocks, dim=1).contiguous()
        counts = np.asarray([x["n_time"] for x in self._prepared], dtype=float)
        condition_weights = counts.mean() / counts if self.balance_conditions else np.ones_like(counts)
        self.condition_weights_ = condition_weights
        self._condition_weights = torch.as_tensor(condition_weights, dtype=dtype, device=device)
        self._log_two_pi = math.log(2.0 * math.pi)

    @staticmethod
    def _evaluate_power(z, parameters):
        exponent = torch.clamp(parameters[:, 1, None] * z[None, :], -60.0, 60.0)
        return parameters[:, 0, None] * torch.exp(exponent)

    def _fit_power_sad(self, curves, z, phi, initial_parameters):
        transformed_curves = curves.clone()
        transformed_curves[:, 1:] -= phi * curves[:, :-1]
        curve_norm = (
            torch.sum(transformed_curves.square(), dim=1)
            + self.tightness_weight * torch.sum(curves.square(), dim=1)
        )

        def profile(beta):
            basis = torch.exp(torch.clamp(beta[:, None] * z[None, :], -60.0, 60.0))
            transformed_basis = basis.clone()
            transformed_basis[:, 1:] -= phi * basis[:, :-1]
            numerator = torch.sum(transformed_basis * transformed_curves, dim=1)
            denominator = torch.sum(transformed_basis.square(), dim=1).clamp_min(1e-12)
            numerator += self.tightness_weight * torch.sum(basis * curves, dim=1)
            denominator += self.tightness_weight * torch.sum(basis.square(), dim=1)
            c = torch.clamp(numerator / denominator, min=0.0)
            loss = (curve_norm - 2.0 * c * numerator + c.square() * denominator).clamp_min(0.0)
            return loss, c

        lower, upper = self.beta_bounds
        golden = 0.5 * (math.sqrt(5.0) - 1.0)
        lo = torch.full((len(curves),), lower, dtype=self.dtype_, device=self.device_)
        hi = torch.full_like(lo, upper)
        beta_1, beta_2 = hi - golden * (hi - lo), lo + golden * (hi - lo)
        loss_1, loss_2 = profile(beta_1)[0], profile(beta_2)[0]
        n_search = max(1, math.ceil(math.log(self.beta_xtol / (upper - lower)) / math.log(golden)))

        for _ in range(n_search):
            move_right = loss_1 > loss_2
            next_lo = torch.where(move_right, beta_1, lo)
            next_hi = torch.where(move_right, hi, beta_2)
            candidate = torch.where(
                move_right,
                next_lo + golden * (next_hi - next_lo),
                next_hi - golden * (next_hi - next_lo),
            )
            candidate_loss = profile(candidate)[0]
            beta_1, loss_1, beta_2, loss_2 = (
                torch.where(move_right, beta_2, candidate),
                torch.where(move_right, loss_2, candidate_loss),
                torch.where(move_right, candidate, beta_1),
                torch.where(move_right, candidate_loss, loss_1),
            )
            lo, hi = next_lo, next_hi

        best_beta = torch.where(loss_1 <= loss_2, beta_1, beta_2)
        best_loss, best_c = profile(best_beta)
        candidates = (
            initial_parameters[:, 1].clamp(lower, upper),
            torch.zeros_like(best_beta),
            torch.full_like(best_beta, lower),
            torch.full_like(best_beta, upper),
        )
        for candidate_beta in candidates:
            candidate_loss, candidate_c = profile(candidate_beta)
            better = candidate_loss < best_loss
            best_loss = torch.where(better, candidate_loss, best_loss)
            best_beta = torch.where(better, candidate_beta, best_beta)
            best_c = torch.where(better, candidate_c, best_c)

        zero_curve = torch.max(torch.abs(curves), dim=1).values <= 1e-12
        best_c = torch.where(zero_curve, torch.zeros_like(best_c), best_c)
        best_beta = torch.where(zero_curve, torch.zeros_like(best_beta), best_beta)
        return torch.stack((best_c, best_beta), dim=1)

    def _e_step(self, weights, params_mu, params_cov):
        feature_constant = torch.zeros(self.n_features_, dtype=self.dtype_, device=self.device_)
        cluster_constant = torch.zeros(self.n_components, dtype=self.dtype_, device=self.device_)
        coefficient_blocks = []

        for ell, data in enumerate(self._prepared):
            phi = params_cov[ell, 0]
            gamma = params_cov[ell, 1].clamp_min(1e-8)
            condition_weight = self._condition_weights[ell]
            mean_curves = self._evaluate_power(data["z"], params_mu[:, ell])
            transformed_mean = mean_curves.clone()
            transformed_mean[:, 1:] -= phi * mean_curves[:, :-1]
            coefficient = transformed_mean.clone()
            coefficient[:, :-1] -= phi * transformed_mean[:, 1:]
            coefficient += self.tightness_weight * mean_curves
            inverse_variance = condition_weight / gamma.square()
            coefficient_blocks.append(coefficient * inverse_variance)

            feature_quadratic = (
                data["sum_y2"] - 2.0 * phi * data["sum_cross"]
                + phi.square() * data["sum_y2_previous"]
                + self.tightness_weight * data["sum_y2"]
            )
            feature_constant += (
                -0.5 * inverse_variance * feature_quadratic
                - condition_weight * (
                    (1.0 + self.tightness_weight)
                    * data["n_time"]
                    * (0.5 * self._log_two_pi + torch.log(gamma))
                )
            )
            cluster_constant += -0.5 * inverse_variance * (
                torch.sum(transformed_mean.square(), dim=1)
                + self.tightness_weight * torch.sum(mean_curves.square(), dim=1)
            )

        coefficients = torch.cat(coefficient_blocks, dim=1).contiguous()
        log_joint = (
            torch.log(weights.clamp_min(1e-30))[None]
            + feature_constant[:, None]
            + cluster_constant[None]
            + self._combined_curves @ coefficients.T
        )
        normalizer = torch.logsumexp(log_joint, dim=1, keepdim=True)
        responsibilities = torch.exp(log_joint - normalizer)
        responsibilities /= responsibilities.sum(dim=1, keepdim=True).clamp_min(1e-30)
        log_likelihood = float(normalizer.to(torch.float64).sum().item())
        return responsibilities, log_likelihood

    def _m_step(self, responsibilities, params_mu, params_cov):
        cluster_sizes = responsibilities.sum(dim=0)
        active = cluster_sizes > 1e-8
        weights = (cluster_sizes / self.n_features_).clamp_min(1e-12)
        weights /= weights.sum()
        new_mu, new_cov = params_mu.clone(), params_cov.clone()
        responsibility_transpose = responsibilities.T.contiguous()
        weighted_sums = responsibility_transpose @ self._combined_curves
        weighted_statistics = responsibility_transpose @ self._combined_statistics

        for ell, data in enumerate(self._prepared):
            weighted_sum = weighted_sums[:, self._curve_slices[ell]]
            offset = 3 * ell
            weighted_y2 = weighted_statistics[:, offset]
            weighted_y2_previous = weighted_statistics[:, offset + 1]
            weighted_cross = weighted_statistics[:, offset + 2]
            mean_curves = weighted_sum / cluster_sizes[:, None].clamp_min(1e-12)
            mean_curves = torch.where(active[:, None], mean_curves, torch.zeros_like(mean_curves))
            mean_parameters = new_mu[:, ell].clone()
            phi = new_cov[ell, 0].clone()
            gamma = new_cov[ell, 1].clamp_min(1e-8).clone()

            for _ in range(self.inner_max_iter):
                previous_parameters = mean_parameters.clone()
                previous_phi, previous_gamma = phi.clone(), gamma.clone()
                if bool(active.any().item()):
                    mean_parameters[active] = self._fit_power_sad(
                        mean_curves[active],
                        data["z"],
                        phi,
                        mean_parameters[active],
                    )

                fitted = self._evaluate_power(data["z"], mean_parameters)
                residual_ss = (
                    weighted_y2 - 2.0 * torch.sum(weighted_sum * fitted, dim=1)
                    + cluster_sizes * torch.sum(fitted.square(), dim=1)
                )
                residual_ss_previous = (
                    weighted_y2_previous
                    - 2.0 * torch.sum(weighted_sum[:, :-1] * fitted[:, :-1], dim=1)
                    + cluster_sizes * torch.sum(fitted[:, :-1].square(), dim=1)
                )
                residual_cross = (
                    weighted_cross
                    - torch.sum(weighted_sum[:, 1:] * fitted[:, :-1], dim=1)
                    - torch.sum(weighted_sum[:, :-1] * fitted[:, 1:], dim=1)
                    + cluster_sizes * torch.sum(fitted[:, 1:] * fitted[:, :-1], dim=1)
                )
                denominator = torch.sum(residual_ss_previous)
                phi = torch.where(
                    denominator > 1e-12,
                    torch.clamp(
                        torch.sum(residual_cross)
                        / ((1.0 + self.phi_ridge) * denominator).clamp_min(1e-12),
                        -self.phi_bound,
                        self.phi_bound,
                    ),
                    torch.zeros_like(phi),
                )
                innovation_ss = (
                    residual_ss
                    - 2.0 * phi * residual_cross
                    + phi.square() * residual_ss_previous
                    + self.tightness_weight * residual_ss
                )
                gamma = torch.sqrt(
                    torch.clamp(
                        torch.sum(innovation_ss)
                        / (
                            (1.0 + self.tightness_weight)
                            * self.n_features_
                            * data["n_time"]
                        ),
                        min=1e-12,
                    )
                )

                parameter_change = torch.max(torch.abs(mean_parameters - previous_parameters))
                covariance_change = torch.maximum(
                    torch.abs(phi - previous_phi),
                    torch.abs(gamma - previous_gamma),
                )
                scale = 1.0 + torch.maximum(
                    torch.max(torch.abs(previous_parameters)),
                    torch.maximum(torch.abs(previous_phi), torch.abs(previous_gamma)),
                )
                if float(torch.maximum(parameter_change, covariance_change).item()) <= self.inner_tol * float(scale.item()):
                    break

            new_mu[:, ell] = mean_parameters
            new_cov[ell] = torch.stack((phi, gamma))

        return weights, new_mu, new_cov, cluster_sizes

    def _solve(self):
        weights = torch.as_tensor(self.initial_weights_, dtype=self.dtype_, device=self.device_).clone()
        params_mu = torch.as_tensor(self.initial_params_mu_, dtype=self.dtype_, device=self.device_).clone()
        params_cov = torch.as_tensor(self.initial_params_cov_[0], dtype=self.dtype_, device=self.device_).clone()
        responsibilities, log_likelihood = self._e_step(weights, params_mu, params_cov)
        history, converged, n_iter = [log_likelihood], False, 0

        if self.progress:
            print(f"[funclu] EM start | log_likelihood={log_likelihood:,.6f} | device={self.device_}")

        for iteration in range(1, self.max_iter + 1):
            new_weights, new_mu, new_cov, _ = self._m_step(responsibilities, params_mu, params_cov)
            new_responsibilities, new_log_likelihood = self._e_step(new_weights, new_mu, new_cov)
            change = new_log_likelihood - log_likelihood
            threshold = max(self.tol, 1e-5 if self.device_.type == "cuda" else self.tol) * (1.0 + abs(log_likelihood))

            if self.progress:
                print(
                    f"[funclu] iter={iteration:03d} | "
                    f"log_likelihood={new_log_likelihood:,.6f} | change={change:+.6e}"
                )
            if change < -threshold:
                break
            if change >= 0.0:
                weights, params_mu, params_cov = new_weights, new_mu, new_cov
                responsibilities, log_likelihood = new_responsibilities, new_log_likelihood
                history.append(log_likelihood)
                n_iter = iteration
            if abs(change) <= threshold:
                converged = True
                break

        self._weights = weights
        self._params_mu = params_mu
        self._params_cov = params_cov
        self._responsibilities = responsibilities
        self.log_likelihood_ = log_likelihood
        self.log_likelihood_history_ = np.asarray(history)
        self.n_iter_ = n_iter
        self.converged_ = converged

    def _finalize(self):
        self.weights_ = self._weights.detach().cpu().numpy()
        self.params_mu_ = self._params_mu.detach().cpu().numpy()
        self.shared_params_cov_ = self._params_cov.detach().cpu().numpy()
        self.params_cov_ = np.broadcast_to(
            self.shared_params_cov_[None],
            (self.n_components, self.n_conditions_, 2),
        ).copy()
        self.responsibilities_ = self._responsibilities.detach().cpu().numpy()
        self.labels_ = np.argmax(self.responsibilities_, axis=1)
        self.cluster_sizes_ = np.bincount(self.labels_, minlength=self.n_components)
        self.effective_cluster_sizes_ = self.responsibilities_.sum(axis=0)
        self.mean_curves_ = {}

        for ell, condition in enumerate(self.condition_names_):
            z = self._prepared[ell]["z"].detach().cpu().numpy()
            c = self.params_mu_[:, ell, 0]
            beta = self.params_mu_[:, ell, 1]
            values = (c[:, None] * np.exp(np.clip(beta[:, None] * z[None], -60.0, 60.0))).T
            self.mean_curves_[condition] = pd.DataFrame(
                values,
                index=self.sample_sets_[condition].index,
                columns=[f"cluster_{k}" for k in range(self.n_components)],
            )

        self.assignments_ = pd.DataFrame({
            "feature": self.feature_names_,
            "cluster": self.labels_,
            "probability": self.responsibilities_.max(axis=1),
        }).set_index("feature")
        self.compute_device_ = str(self.device_)
        if self.device_.type == "cuda":
            torch.cuda.synchronize()
        for name in (
            "_weights",
            "_params_mu",
            "_params_cov",
            "_responsibilities",
            "_prepared",
            "_combined_curves",
            "_combined_statistics",
            "_condition_weights",
        ):
            delattr(self, name)

    def fit(self, sample_sets: dict[str, pd.DataFrame], parameter_sets: dict[str, pd.DataFrame]):
        self._prepare_inputs(sample_sets, parameter_sets)
        if not 1 <= self.n_components <= self.n_features_:
            raise ValueError("n_components must not exceed the valid common feature count")
        for samples in self.sample_sets_.values():
            time = samples.index.to_numpy(float)
            if len(time) < 2 or not np.isfinite(time).all() or (time <= 0).any() or np.unique(time).size < 2:
                raise ValueError("Samples require at least two distinct finite positive times")
        self._initialize()
        self._prepare_solver_data()
        self._solve()
        self._finalize()
        if not np.isfinite(self.log_likelihood_) or not np.isfinite(self.responsibilities_).all():
            raise ValueError("FunClu produced non-finite results; check data magnitudes and parameters")
        return self

    def fit_predict(self, sample_sets, parameter_sets):
        return self.fit(sample_sets, parameter_sets).predict()

    def predict(self):
        return self.labels_.copy()

    def predict_proba(self):
        return self.responsibilities_.copy()

    def score(self):
        return self.log_likelihood_

    def bic(self):
        n_parameters = (
            self.n_components - 1
            + 2 * self.n_components * self.n_conditions_
            + 2 * self.n_conditions_
        )
        return -2.0 * self.log_likelihood_ + n_parameters * np.log(self.n_features_)

    def result(self):
        return {
            "labels": self.labels_,
            "responsibilities": self.responsibilities_,
            "weights": self.weights_,
            "params_mu": self.params_mu_,
            "params_cov": self.params_cov_,
            "shared_params_cov": self.shared_params_cov_,
            "condition_names": self.condition_names_,
            "feature_names": self.feature_names_,
            "excluded_features": self.excluded_features_,
            "cluster_sizes": self.cluster_sizes_,
            "effective_cluster_sizes": self.effective_cluster_sizes_,
            "mean_curves": self.mean_curves_,
            "log_likelihood": self.log_likelihood_,
            "log_likelihood_history": self.log_likelihood_history_,
            "n_iter": self.n_iter_,
            "converged": self.converged_,
            "compute_device": self.compute_device_,
            "condition_weights": self.condition_weights_,
        }

    def save(self, output_dir: str | Path):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        assignments = self.assignments_.copy()
        assignments["cluster"] += 1
        assignments.to_csv(output_dir / "assignments.csv")

        pd.DataFrame(
            self.responsibilities_,
            index=self.feature_names_,
            columns=[f"M{k + 1}" for k in range(self.n_components)],
        ).rename_axis("feature").to_csv(output_dir / "responsibilities.csv")

        pd.DataFrame({
            "cluster": np.arange(1, self.n_components + 1),
            "weight": self.weights_,
            "cluster_size": self.cluster_sizes_,
            "effective_cluster_size": self.effective_cluster_sizes_,
        }).to_csv(output_dir / "clusters.csv", index=False)

        mean_parameters = []
        for k in range(self.n_components):
            for ell, condition in enumerate(self.condition_names_):
                mean_parameters.append({
                    "cluster": k + 1,
                    "condition": condition,
                    "c": self.params_mu_[k, ell, 0],
                    "beta": self.params_mu_[k, ell, 1],
                })
        pd.DataFrame(mean_parameters).to_csv(
            output_dir / "mean_parameters.csv",
            index=False,
        )

        pd.DataFrame({
            "condition": self.condition_names_,
            "phi": self.shared_params_cov_[:, 0],
            "gamma": self.shared_params_cov_[:, 1],
        }).to_csv(output_dir / "sad_parameters.csv", index=False)

        pd.DataFrame({
            "iteration": np.arange(len(self.log_likelihood_history_)),
            "log_likelihood": self.log_likelihood_history_,
        }).to_csv(output_dir / "log_likelihood.csv", index=False)

        pd.DataFrame({
            "condition_code": [f"C{ell + 1}" for ell in range(self.n_conditions_)],
            "condition": self.condition_names_,
        }).to_csv(output_dir / "conditions.csv", index=False)

        pd.DataFrame([{
            "n_components": self.n_components,
            "n_features": self.n_features_,
            "n_conditions": self.n_conditions_,
            "log_likelihood": self.log_likelihood_,
            "bic": self.bic(),
            "n_iter": self.n_iter_,
            "converged": self.converged_,
            "compute_device": self.compute_device_,
            "phi_bound": self.phi_bound,
            "phi_ridge": self.phi_ridge,
            "tightness_weight": self.tightness_weight,
        }]).to_csv(output_dir / "model_summary.csv", index=False)

        result_dir = output_dir / "res"
        result_dir.mkdir(parents=True, exist_ok=True)
        for k in range(self.n_components):
            folder = result_dir / f"M{k + 1}"
            folder.mkdir(parents=True, exist_ok=True)
            members = self.feature_names_[self.labels_ == k]
            for ell, condition in enumerate(self.condition_names_):
                curves = self.sample_sets_[condition].loc[:, members].copy()
                curves["mean"] = self.mean_curves_[condition].iloc[:, k]
                curves.to_csv(folder / f"C{ell + 1}_M{k + 1}.csv")

        return output_dir


# Alias of the original ``_v4.FunClu`` numerical kernel.
_funclu_v4_kernel = _V4FunCluKernel


def select_best_k(
    sample_sets: dict[str, pd.DataFrame],
    parameter_sets: dict[str, pd.DataFrame],
    k_values: list[int] | tuple[int, ...] | None = None,
    k_min: int = 2,
    k_max: int = 500,
    k_step: int = 10,
    max_iter: int = 1000,
    min_cluster_size: int = 10,
    n_init: int = 10,
    compute_device: str = "auto",
    random_state: int = 42,
    progress: bool = True,
    **funclu_options,
):
    """Fit every supplied K with identical precision and select the lowest BIC."""

    n_features = min(samples.shape[1] for samples in sample_sets.values())
    k_max = min(k_max, n_features // max(min_cluster_size, 1))
    if k_values is None:
        candidates = list(range(k_min, k_max + 1, max(k_step, 1)))
        if candidates and candidates[-1] != k_max:
            candidates.append(k_max)
    else:
        candidates = sorted({
            int(k) for k in k_values
            if k_min <= int(k) <= k_max
        })

    records = []
    options = dict(funclu_options)
    for key in ("n_components", "max_iter", "n_init", "compute_device", "random_state", "progress"):
        options.pop(key, None)

    def fit_candidate(k, index):
        if progress:
            print(f"[funclu] BIC {index}/{len(candidates)}: fitting K={k} ...", flush=True)
        started = perf_counter()
        try:
            model = _V4FunCluKernel(
                n_components=k,
                max_iter=max_iter,
                n_init=n_init,
                compute_device=compute_device,
                random_state=random_state,
                progress=False,
                **options,
            ).fit(sample_sets, parameter_sets)
            weights = model.cluster_sizes_ / model.cluster_sizes_.sum()
            positive_weights = weights[weights > 0]
            entropy = -np.sum(positive_weights * np.log(positive_weights)) / np.log(k)
            record = {
                "k": k,
                "bic": model.bic(),
                "log_likelihood": model.log_likelihood_,
                "n_iter": model.n_iter_,
                "converged": model.converged_,
                "min_cluster_size": int(model.cluster_sizes_.min()),
                "normalized_entropy": float(entropy),
                "eligible": bool(model.cluster_sizes_.min() >= min_cluster_size),
                "seconds": perf_counter() - started,
                "error": "",
            }
        except Exception as error:
            model = None
            record = {
                "k": k,
                "bic": np.inf,
                "log_likelihood": np.nan,
                "n_iter": 0,
                "converged": False,
                "min_cluster_size": 0,
                "normalized_entropy": np.nan,
                "eligible": False,
                "seconds": perf_counter() - started,
                "error": str(error),
            }
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        records.append(record)
        if progress:
            if model is None:
                print(f"[funclu] BIC K={k} failed: {record['error']}", flush=True)
            else:
                print(
                    f"[funclu] BIC K={k} | BIC={record['bic']:,.4f} | "
                    f"iter={record['n_iter']} | converged={record['converged']} | "
                    f"min_size={record['min_cluster_size']} | {record['seconds']:.1f}s",
                    flush=True,
                )
        return model, record

    best_model, best_record, best_key = None, None, None
    for index, k in enumerate(candidates, start=1):
        model, record = fit_candidate(k, index)
        if model is None:
            continue
        quality = (
            0 if record["eligible"] and record["converged"]
            else 1 if record["eligible"]
            else 2
        )
        key = (quality, record["bic"])
        if best_key is None or key < best_key:
            del best_model
            best_model, best_record, best_key = model, record, key
        else:
            del model

    bic_table = pd.DataFrame(records)
    bic_table["selected"] = False
    if best_record is not None:
        bic_table.loc[bic_table["k"] == best_record["k"], "selected"] = True
        if progress:
            print(
                f"[funclu] BIC selected K={best_record['k']} | "
                f"BIC={best_record['bic']:,.4f} | converged={best_record['converged']}",
                flush=True,
            )

    return {
        "best_k": None if best_record is None else best_record["k"],
        "best_model": best_model,
        "bic_table": bic_table,
        "candidates": candidates,
    }


"""Application-compatible interface backed by the FunClu v4 solver."""
from typing import Any, Callable, Dict, List, Optional, Tuple

from sklearn.metrics import silhouette_score



class _PublicFunClu:
    """Application-facing tensor/export interface, fitted by the v4 kernel.

    ``params_mu`` stays ``(a, b)``; the native standardised ``(c, beta)``
    parameters are available on ``kernel_.params_mu_``.
    """
    def __init__(self, n_components=3, max_iter=1000, tol=1e-4,
                 device=None, dtype=torch.float64, *, random_state=42,
                 use_minibatch_kmeans=None, kmeans_minibatch_threshold=8000,
                 minibatch_batch_size=4096, minibatch_max_iter=100, **options):
        self.n_components = int(n_components)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.device = torch.device(device or "cpu")
        self.dtype = dtype
        self.random_state = int(random_state)
        self.use_minibatch_kmeans = use_minibatch_kmeans
        self.options = options
        self.common_cols = None
        self.n_features = self.n_conditions = 0
        self.times_list = self.n_times_conditions = None
        self.params_mu = self.params_cov = self.weights = self.labels = None

    def _prepare_data(
        self,
        data: List[pd.DataFrame],
    ) -> Tuple[List[torch.Tensor], List[np.ndarray]]:
        """Extract tensors and time vectors from one DataFrame per condition.

        Time lengths may differ between conditions, so nothing is aligned; only
        the intersection of the column names is kept, and every condition is
        clustered on those same features.

        Raises:
            ValueError: ``data`` is empty or the column intersection is empty.
        """
        if not data:
            raise ValueError("data must not be empty: at least 1 condition DataFrame is required")

        common_cols: List[str] = list(data[0].columns)
        for d in data[1:]:
            cols_d = set(d.columns)
            common_cols = [c for c in common_cols if c in cols_d]
        if len(common_cols) == 0:
            raise ValueError("the condition DataFrames share no common columns; cannot cluster")

        self.common_cols = common_cols
        self.n_features = len(common_cols)
        self.n_conditions = len(data)

        X_list: List[torch.Tensor] = []
        times_list: List[np.ndarray] = []
        for df in data:
            sub = df[common_cols]
            vals = sub.to_numpy(dtype=np.float64, copy=True).T
            idx = np.asarray(sub.index, dtype=np.float64)
            X_list.append(torch.from_numpy(vals).to(self.device, self.dtype))
            times_list.append(idx)

        self.times_list = times_list
        self.n_times_conditions = [len(t) for t in times_list]

        return X_list, times_list


    def fit(self, data, *, parameter_data=None, verbose=False):
        self._prepare_data(data)
        if parameter_data is not None and len(parameter_data) != len(data):
            raise ValueError("the number of parameter tables must match the number of conditions")
        samples, parameters = {}, {}
        for i, df in enumerate(data):
            t = np.asarray(df.index, dtype=float)
            if len(t) < 2 or not np.isfinite(t).all() or (t <= 0).any() or np.unique(t).size < 2:
                raise ValueError("clustering requires at least two distinct finite positive time points; rerun the fitter")
            if df.columns.has_duplicates:
                raise ValueError("feature names must be unique")
            samples[str(i)] = df
            parameters[str(i)] = (power_fitting(df)[0] if parameter_data is None
                                  else parameter_data[i])
        options = dict(self.options)
        options.setdefault("compute_device", str(self.device))
        options.setdefault("use_mini_batch", True if self.use_minibatch_kmeans is None
                           else self.use_minibatch_kmeans)
        options.setdefault("progress", verbose)
        kernel = _funclu_v4_kernel(n_components=self.n_components, max_iter=self.max_iter,
                          tol=self.tol, random_state=self.random_state, **options)
        kernel._prepare_inputs(samples, parameters)
        if not 1 <= self.n_components <= kernel.n_features_:
            raise ValueError(f"K must be between 1 and the valid common feature count ({kernel.n_features_})")
        kernel.fit(samples, parameters)
        self.kernel_ = kernel
        self.common_cols = list(kernel.feature_names_)
        self.n_features = kernel.n_features_
        self.excluded_features_ = kernel.excluded_features_
        mu = kernel.params_mu_.copy()
        for i, t in enumerate(self.times_list):
            log_t = np.log(t)
            b = mu[:, i, 1] / max(log_t.std(), 1e-8)
            mu[:, i, 0] *= np.exp(-b * log_t.mean())
            mu[:, i, 1] = b
        self.params_mu = torch.as_tensor(mu, dtype=self.dtype, device=self.device)
        self.params_cov = torch.as_tensor(kernel.params_cov_, dtype=self.dtype, device=self.device)
        self.weights = torch.as_tensor(kernel.weights_, dtype=self.dtype, device=self.device)
        self.labels = torch.as_tensor(kernel.labels_, dtype=torch.long, device=self.device)
        self.log_likelihood = kernel.log_likelihood_
        self.neg_log_likelihood = -self.log_likelihood
        self.n_params = self.n_components - 1 + 2 * self.n_components * self.n_conditions + 2 * self.n_conditions
        self.bic = float(kernel.bic())
        self.converged = kernel.converged_
        self.n_iter_run = kernel.n_iter_
        self.loglik_history = kernel.log_likelihood_history_.tolist()
        return self

    def predict(self, data=None):
        if not hasattr(self, "kernel_"):
            raise RuntimeError("call fit first")
        if data is None:
            return self.labels.clone()
        # Use the training time basis and covariance without changing fitted state.
        if len(data) != self.n_conditions:
            raise ValueError("the number of prediction conditions must match training")
        import copy
        kernel = copy.copy(self.kernel_)
        features = list(data[0].columns)
        for i, df in enumerate(data):
            if not np.array_equal(np.asarray(df.index, dtype=float), self.times_list[i]):
                raise ValueError("prediction time points must match the training ones")
            features = [c for c in features if c in df.columns]
        if not features:
            raise ValueError("prediction data has no common features")
        kernel.sample_sets_ = {str(i): df.loc[:, features] for i, df in enumerate(data)}
        if any(not np.isfinite(df.to_numpy(float)).all() for df in kernel.sample_sets_.values()):
            raise ValueError("prediction data must be finite")
        kernel.n_features_ = len(features)
        kernel._prepare_solver_data()
        convert = lambda a: torch.as_tensor(a, dtype=kernel.dtype_, device=kernel.device_)
        resp, _ = kernel._e_step(convert(kernel.weights_), convert(kernel.params_mu_),
                                 convert(kernel.shared_params_cov_))
        return resp.argmax(dim=1).to(self.device)

    def get_cluster_curves(self, condition_idx=0):
        if not hasattr(self, "kernel_"):
            raise RuntimeError("call fit first")
        frame = self.kernel_.mean_curves_[self.kernel_.condition_names_[condition_idx]]
        return frame.index.to_numpy(float), frame.to_numpy(float).T

    def get_params(self) -> Dict[str, Optional[np.ndarray]]:
        """Export the key model parameters as a numpy dict for persistence.

        Keys: ``mu_params (K,L,2)``, ``cov_params (K,L,2)``, ``weights (K,)``
        and ``labels (N,)``; fields not filled in yet are ``None``.
        """
        return {
            "mu_params": (
                self.params_mu.detach().cpu().numpy()
                if self.params_mu is not None
                else None
            ),
            "cov_params": (
                self.params_cov.detach().cpu().numpy()
                if self.params_cov is not None
                else None
            ),
            "weights": (
                self.weights.detach().cpu().numpy()
                if self.weights is not None
                else None
            ),
            "labels": (
                self.labels.detach().cpu().numpy()
                if self.labels is not None
                else None
            ),
        }



def _standardized_curve_matrix(X_list: List[torch.Tensor]) -> np.ndarray:
    """Return a standardized feature-by-time matrix for distance-based scores."""
    if not X_list:
        return np.empty((0, 0), dtype=np.float64)

    X_concat = torch.cat(X_list, dim=1).detach().cpu().numpy().astype(np.float64)
    mean = np.nanmean(X_concat, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    X_concat = np.where(np.isfinite(X_concat), X_concat, mean)

    scale = np.nanstd(X_concat, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    return (X_concat - mean) / scale


def _compute_silhouette_index(
    X_list: List[torch.Tensor],
    labels: Optional[torch.Tensor],
) -> float:
    """Compute Silhouette Index (SI) from final hard cluster labels."""
    if labels is None:
        return float("nan")

    y = labels.detach().cpu().numpy().astype(int)
    n_samples = int(y.size)
    n_labels = int(np.unique(y).size)
    if n_samples < 3 or n_labels < 2 or n_labels >= n_samples:
        return float("nan")

    try:
        return float(silhouette_score(_standardized_curve_matrix(X_list), y))
    except Exception:
        return float("nan")


def _cluster_size_stats(
    labels: Optional[torch.Tensor],
    n_components: int,
    min_cluster_size: int,
) -> Dict[str, Any]:
    """Summarize hard-label cluster sizes for K-selection diagnostics."""
    if labels is None:
        return {
            "min_cluster_size": float("nan"),
            "max_cluster_size": float("nan"),
            "small_cluster_count": 0,
            "empty_cluster_count": 0,
            "passes_min_cluster_size": False,
        }

    y = labels.detach().cpu().numpy().astype(int)
    counts = np.bincount(y, minlength=int(n_components))
    return {
        "min_cluster_size": int(counts.min()) if counts.size else 0,
        "max_cluster_size": int(counts.max()) if counts.size else 0,
        "small_cluster_count": int((counts < int(min_cluster_size)).sum()),
        "empty_cluster_count": int((counts == 0).sum()),
        "passes_min_cluster_size": bool(np.all(counts >= int(min_cluster_size))),
    }


def _finite_values(values: List[Any]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def _aggregate_values(values: List[Any], method: str) -> float:
    arr = _finite_values(values)
    if arr.size == 0:
        return float("nan")
    if method == "mean":
        return float(np.mean(arr))
    return float(np.median(arr))


def _mean_values(values: List[Any]) -> float:
    arr = _finite_values(values)
    return float(np.mean(arr)) if arr.size else float("nan")


def _median_values(values: List[Any]) -> float:
    arr = _finite_values(values)
    return float(np.median(arr)) if arr.size else float("nan")


def _median_int_values(values: List[Any]) -> int:
    arr = _finite_values(values)
    return int(np.median(arr)) if arr.size else 0


def _std_values(values: List[Any]) -> float:
    arr = _finite_values(values)
    return float(np.std(arr, ddof=0)) if arr.size else float("nan")


def compute_bic_scores(
    data: List[pd.DataFrame],
    k_min: int = 2,
    k_max: int = 10,
    step: int = 1,
    *,
    max_iter: int = 1000,
    tol: float = 1e-4,
    random_state: int = 42,
    n_random_starts: int = 1,
    aggregation: str = "median",
    min_cluster_size: int = 1,
    parameter_data: Optional[List[pd.DataFrame]] = None,
    verbose: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:
    """Scan a range of K values and return BIC scores.

    Each K instantiates a :class:`FunClu`, fits it and records BIC,
    log-likelihood and convergence; a failing K becomes a NaN row instead of
    aborting the scan.

    Returns:
        DataFrame with ``K``, ``BIC``, ``SI``, ``log_likelihood``, ``NLL``,
        aggregate diagnostics, seed counts and cluster-size flags.
    """
    if k_min < 2:
        raise ValueError(f"k_min must be >= 2, got {k_min}")
    if k_max < k_min:
        raise ValueError(f"k_max ({k_max}) must be >= k_min ({k_min})")
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")
    if n_random_starts < 1:
        raise ValueError(f"n_random_starts must be >= 1, got {n_random_starts}")
    if aggregation not in {"median", "mean"}:
        raise ValueError("aggregation must be 'median' or 'mean'")
    if min_cluster_size < 1:
        raise ValueError(f"min_cluster_size must be >= 1, got {min_cluster_size}")

    n_features = min((d.shape[1] for d in data), default=0)
    if n_features < 2:
        raise ValueError(f"Data has fewer than 2 features (got {n_features})")
    capped_k_max = min(k_max, n_features)

    ks = list(range(k_min, capped_k_max + 1, step))
    if not ks:
        raise ValueError(
            f"No K values to scan: k_min={k_min}, "
            f"k_max={k_max}, n_features={n_features}"
        )

    rows: List[Dict[str, Any]] = []
    total = len(ks)
    seeds = [int(random_state) + offset for offset in range(int(n_random_starts))]

    for idx, k in enumerate(ks):
        seed_rows: List[Dict[str, Any]] = []
        for seed in seeds:
            seed_row: Dict[str, Any] = {
                "seed": seed,
                "BIC": float("nan"),
                "SI": float("nan"),
                "log_likelihood": float("nan"),
                "NLL": float("nan"),
                "n_params": 0,
                "converged": False,
                "n_iter_run": 0,
                "n_features": 0,
                "n_conditions": 0,
                "min_cluster_size": float("nan"),
                "max_cluster_size": float("nan"),
                "small_cluster_count": 0,
                "empty_cluster_count": 0,
                "passes_min_cluster_size": False,
                "fit_success": False,
                "error": "",
            }
            try:
                model = FunClu(
                    n_components=k, max_iter=max_iter, tol=tol,
                    random_state=seed,
                )
                model.fit(data, parameter_data=parameter_data, verbose=verbose)
                X_list = [torch.as_tensor(d.loc[:, model.common_cols].to_numpy(float).T) for d in data]
                size_stats = _cluster_size_stats(
                    model.labels,
                    n_components=k,
                    min_cluster_size=min_cluster_size,
                )
                seed_row.update({
                    "BIC": model.bic,
                    "SI": _compute_silhouette_index(X_list, model.labels),
                    "log_likelihood": model.log_likelihood,
                    "NLL": model.neg_log_likelihood,
                    "n_params": model.n_params,
                    "converged": model.converged,
                    "n_iter_run": model.n_iter_run,
                    "n_features": model.n_features,
                    "n_conditions": model.n_conditions,
                    "fit_success": bool(np.isfinite(model.bic)),
                    **size_stats,
                })
            except Exception as error:
                seed_row["error"] = str(error)
            seed_rows.append(seed_row)

        fit_successes = int(sum(bool(r["fit_success"]) for r in seed_rows))
        row: Dict[str, Any] = {
            "K": k,
            "error": "; ".join(r["error"] for r in seed_rows if r["error"]),
            "BIC": _aggregate_values([r["BIC"] for r in seed_rows], aggregation),
            "BIC_mean": _mean_values([r["BIC"] for r in seed_rows]),
            "BIC_median": _median_values([r["BIC"] for r in seed_rows]),
            "BIC_sd": _std_values([r["BIC"] for r in seed_rows]),
            "SI": _aggregate_values([r["SI"] for r in seed_rows], aggregation),
            "SI_mean": _mean_values([r["SI"] for r in seed_rows]),
            "SI_median": _median_values([r["SI"] for r in seed_rows]),
            "log_likelihood": _aggregate_values(
                [r["log_likelihood"] for r in seed_rows], aggregation
            ),
            "NLL": _aggregate_values([r["NLL"] for r in seed_rows], aggregation),
            "n_params": _median_int_values([r["n_params"] for r in seed_rows]),
            "converged": bool(
                _mean_values([float(r["converged"]) for r in seed_rows]) >= 0.5
            ),
            "converged_rate": _mean_values(
                [float(r["converged"]) for r in seed_rows]
            ),
            "n_iter_run": _aggregate_values(
                [r["n_iter_run"] for r in seed_rows], aggregation
            ),
            "n_features": _median_int_values([r["n_features"] for r in seed_rows]),
            "n_conditions": _median_int_values([r["n_conditions"] for r in seed_rows]),
            "min_cluster_size": _aggregate_values(
                [r["min_cluster_size"] for r in seed_rows], aggregation
            ),
            "min_cluster_size_min": (
                float(np.min(_finite_values([r["min_cluster_size"] for r in seed_rows])))
                if _finite_values([r["min_cluster_size"] for r in seed_rows]).size
                else float("nan")
            ),
            "max_cluster_size": _aggregate_values(
                [r["max_cluster_size"] for r in seed_rows], aggregation
            ),
            "small_cluster_count": _aggregate_values(
                [r["small_cluster_count"] for r in seed_rows], aggregation
            ),
            "small_cluster_count_max": (
                float(np.max(_finite_values([r["small_cluster_count"] for r in seed_rows])))
                if _finite_values([r["small_cluster_count"] for r in seed_rows]).size
                else float("nan")
            ),
            "empty_cluster_count": _aggregate_values(
                [r["empty_cluster_count"] for r in seed_rows], aggregation
            ),
            "size_ok_rate": _mean_values(
                [float(r["passes_min_cluster_size"]) for r in seed_rows]
            ),
            "passes_min_cluster_size": bool(
                _mean_values([float(r["passes_min_cluster_size"]) for r in seed_rows])
                >= 0.5
            ),
            "fit_successes": fit_successes,
            "random_starts": int(n_random_starts),
            "aggregation": aggregation,
            "seed_start": int(random_state),
            "seed_end": int(random_state) + int(n_random_starts) - 1,
            "min_cluster_size_threshold": int(min_cluster_size),
        }
        rows.append(row)

        if progress_callback is not None:
            progress_callback(idx + 1, total)

    return pd.DataFrame(rows)


# Two modules were flattened into one, so the original ``funclu.FunClu`` export
# interface was renamed to ``_PublicFunClu``: module level must keep the v4
# numerical kernel, which ``select_best_k`` builds under the name ``FunClu``.
class FunClu(_PublicFunClu):  # noqa: F811 - restore the original public name
    """Application-facing export interface (original ``funclu.FunClu``)."""


from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties

from QuasiDynamicAnalyzer import fit_power_loglinear

try:
    import torch
except ImportError:
    raise ImportError(
        "idopnetwork[ml] is required for clustering plots. "
        "Install with: pip install idopnetwork[ml]"
    )

font_prop = None  # Set by application layer for CJK font support


# Palette: (member line, KMeans centre marker, power-law fit curve).
_DEFAULT_INIT_PALETTE: List[Tuple[str, str, str]] = [
    ("#FAD7A0", "#D35400", "#A04000"),
    ("#AED6F1", "#2E86C1", "#1B4F72"),
    ("#A9DFBF", "#239B56", "#196F3D"),
    ("#F5B7B1", "#C0392B", "#922B21"),
    ("#D7BDE2", "#8E44AD", "#5B2C6F"),
    ("#A3E4D7", "#16A085", "#0E6655"),
]

# Cluster-profile palette: (member line colour, mean curve colour), aligned
# with the funclu_v4 reference implementation.
_CLUSTER_PROFILE_DEFAULT_PALETTE: List[Tuple[str, str]] = [
    ("#FFB3A6", "#D94E3D"),
    ("#A8D8EA", "#2B7A9E"),
    ("#C1E1C1", "#2E8B57"),
]


def _font_at(size: float, weight: Optional[str] = None) -> FontProperties:
    """Return the project font at the given size (and optional weight).

    ``font_prop`` is built by the application layer from ``FontProperties(fname=...)``
    and carries no size of its own, so handing it to ``fontproperties=`` also
    overrides ``fontsize=``/``fontweight=``. Size and weight therefore have to
    be set on a copy and the explicit keywords cannot be relied on.
    """
    if font_prop is None:
        prop = FontProperties(size=size)
    else:
        prop = font_prop.copy()
        prop.set_size(size)
    if weight is not None:
        prop.set_weight(weight)
    return prop


def _legend_font(size: float) -> object:
    """Project font at the given size, for use as a legend prop."""
    return _font_at(size)


def _set_chinese_axes(ax: plt.Axes, labelsize: Optional[float] = None) -> None:
    """Apply the project font to the tick labels.

    ``labelsize=None`` keeps the size baked into ``font_prop``; an explicit size
    is set on a copy, since ``font_prop`` would otherwise override
    ``tick_params(labelsize=...)``.
    """
    prop = _font_at(labelsize) if labelsize is not None else font_prop
    if prop is None:
        return
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontproperties(prop)


def plot_initialization_grid(
    *,
    X_list: List[torch.Tensor],
    times_list: List[np.ndarray],
    labels: np.ndarray,
    centers_kl: List[List[np.ndarray]],
    params_mu: np.ndarray,
    condition_labels: Optional[List[str]] = None,
    cluster_label_prefix: str = "M",
    layout: str = "k_by_l",
    figsize_per_panel: Tuple[float, float] = (3.0, 2.2),
    member_alpha: float = 0.75,
    member_lw: float = 0.8,
    center_marker_size: float = 1.0,
    fit_lw: float = 2.0,
    use_semilogx: bool = False,
    use_semilogy: bool = True,
    palette: Optional[Sequence[Tuple[str, str, str]]] = None,
    show_legend: bool = True,
    dpi: int = 200,
    share_x: bool = False,
    share_y: bool = False,
) -> Figure:
    """Draw the KMeans-initialisation diagnostic grid.

    Each panel is one ``(cluster k, condition i)`` pair and overlays the member
    curves, the KMeans sub-centre and the ``y = a * t^b`` fit taken from
    ``params_mu[k, i]``.

    Raises:
        ValueError: invalid ``layout`` or inconsistent shapes.
    """
    K = int(params_mu.shape[0])
    L = int(params_mu.shape[1])
    if len(X_list) != L or len(times_list) != L or len(centers_kl) != K:
        raise ValueError(
            f"inconsistent shapes: K={K}, L={L}; len(X_list)={len(X_list)}, "
            f"len(times_list)={len(times_list)}, len(centers_kl)={len(centers_kl)}"
        )
    if layout not in ("k_by_l", "l_by_k"):
        raise ValueError(f"layout must be 'k_by_l' or 'l_by_k', got {layout!r}")

    if condition_labels is None:
        condition_labels = [f"Cond {i + 1}" for i in range(L)]

    pal: List[Tuple[str, str, str]] = (
        list(palette) if palette is not None else list(_DEFAULT_INIT_PALETTE)
    )
    while len(pal) < max(K, L):
        pal.extend(_DEFAULT_INIT_PALETTE)

    n_rows, n_cols = (K, L) if layout == "k_by_l" else (L, K)
    sw, sh = figsize_per_panel
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(sw * n_cols, sh * n_rows),
        sharex=share_x,
        sharey=share_y,
        dpi=dpi,
    )
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.asarray(axes).reshape(1, n_cols)
    elif n_cols == 1:
        axes = np.asarray(axes).reshape(n_rows, 1)

    labels_np = np.asarray(labels).astype(np.int64, copy=False)
    legend_drawn = False

    for k in range(K):
        mask_k = labels_np == k
        for i in range(L):
            t = np.asarray(times_list[i], dtype=np.float64)
            X_i_np = X_list[i].detach().cpu().numpy() if isinstance(
                X_list[i], torch.Tensor
            ) else np.asarray(X_list[i], dtype=np.float64)
            members = X_i_np[mask_k] if mask_k.any() else np.empty((0, t.size))
            center = np.asarray(centers_kl[k][i], dtype=np.float64)
            a = float(params_mu[k, i, 0])
            b = float(params_mu[k, i, 1])

            row, col = (k, i) if layout == "k_by_l" else (i, k)
            ax = axes[row, col]

            # Colour by condition (index i) so a condition keeps its colour across clusters.
            mem_c, ctr_c, fit_c = pal[i % len(pal)]

            # 1) member curves
            for mb in members:
                ax.plot(
                    t,
                    mb,
                    "-",
                    color=mem_c,
                    linewidth=member_lw,
                    alpha=member_alpha,
                    zorder=1,
                )

            # 2) sub-centre markers (zorder=4 keeps them above the fitted curve)
            ax.scatter(
                t,
                center,
                s=center_marker_size,
                color=ctr_c,
                edgecolors="black",
                linewidths=0.3,
                alpha=0.25,
                zorder=4,
                label="KMeans center" if not legend_drawn else None,
            )

            # 3) fitted power-law curve
            t_min = float(t.min()) if t.size else 0.0
            t_max = float(t.max()) if t.size else 1.0
            t_lo = max(t_min, 1e-12) if use_semilogx else t_min
            t_dense = np.linspace(t_lo, t_max, 200)
            with np.errstate(over="ignore", invalid="ignore"):
                y_fit = a * np.power(t_dense, b)
            ax.plot(
                t_dense,
                y_fit,
                "-",
                color=fit_c,
                linewidth=fit_lw,
                zorder=3,
                label=r"$y=a\cdot t^{b}$" if not legend_drawn else None,
            )
            legend_drawn = True

            if use_semilogx:
                ax.set_xscale("log")
            if use_semilogy:
                ax.set_yscale("log")

            n_k = int(mask_k.sum())
            title = (
                f"{cluster_label_prefix}{k + 1} | {condition_labels[i]}\n"
                f"a={a:.2g}, b={b:.2g}, n={n_k}"
            )
            ax.set_title(title, fontsize=8, fontproperties=font_prop)
            ax.tick_params(labelsize=7)
            ax.grid(False)
            _set_chinese_axes(ax)

    if show_legend:
        handles, lbls = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                lbls,
                loc="upper center",
                ncol=2,
                prop=_legend_font(9.0),
                bbox_to_anchor=(0.5, 0.99),
            )

    fig.tight_layout(rect=(0, 0, 1, 0.96) if show_legend else (0, 0, 1, 1))

    return fig


def plot_cluster_fit_grid(
    *,
    X_list: List["torch.Tensor"],
    times_list: List[np.ndarray],
    labels: np.ndarray,
    mean_curves: dict,
    params_mu: np.ndarray,
    condition_names: List[str],
    condition_labels: Optional[List[str]] = None,
    cluster_label_prefix: str = "M",
    sharey: object = "row",
    sharex: object = False,
    global_ylim: Optional[Tuple[float, float]] = None,
    normalize: str = "none",
    ylabel_only_first_col: bool = True,
    ylabel_text: Optional[str] = None,
    figsize_per_panel: Tuple[float, float] = (3.6, 2.6),
    member_alpha: float = 0.75,
    member_lw: float = 0.8,
    center_lw: float = 3.2,
    fit_lw: float = 2.2,
    dpi: int = 200,
    show_legend: bool = True,
    decimal_axis: bool = True,
) -> Figure:
    """Draw the clustering diagnostic grid: member curves + cluster mean curves.

    Only these two line types are drawn. Unlike :func:`plot_initialization_grid`
    no ``y = a * t^b`` fit is added: ``params_mu[k, i]`` holds the power-law
    parameters ``(c, beta)`` of standardised log time (``y = c * exp(beta * z)``),
    so treating ``c`` as ``a`` would overstate the curve systematically.
    """
    K = int(params_mu.shape[0])
    L = int(params_mu.shape[1])
    if len(X_list) != L or len(times_list) != L:
        raise ValueError(
            f"inconsistent shapes: K={K}, L={L}; len(X_list)={len(X_list)}, "
            f"len(times_list)={len(times_list)}"
        )

    pal = list(_DEFAULT_INIT_PALETTE)
    while len(pal) < max(K, L):
        pal.extend(_DEFAULT_INIT_PALETTE)

    sw, sh = figsize_per_panel
    fig, axes = plt.subplots(
        K, L, figsize=(sw * L, sh * K), sharex=sharex, sharey=sharey, dpi=dpi,
        layout="constrained",
    )
    axes = np.asarray(axes).reshape(K, L)

    labels_np = np.asarray(labels).astype(np.int64, copy=False)
    legend_drawn = False
    formatter = ticker.ScalarFormatter(useOffset=False) if decimal_axis else None
    if formatter is not None:
        formatter.set_scientific(False)

    def _apply_scale(values: np.ndarray, ref: Optional[float] = None) -> np.ndarray:
        """Apply the ``normalize`` scaling to one curve (plot only)."""
        v = np.asarray(values, dtype=float)
        if normalize in (None, "none"):
            return v
        if normalize in ("baseline", "log_baseline"):
            base = ref if ref is not None else float(np.nanmean(v[:3]))
            base = base if np.isfinite(base) and abs(base) > 1e-12 else 1.0
            v = v / base
            return np.log1p(np.maximum(v, 0.0)) if normalize == "log_baseline" else v
        if normalize == "mean":
            m = float(np.nanmean(v))
            return v / (m if np.isfinite(m) and abs(m) > 1e-12 else 1.0)
        if normalize == "log1p":
            return np.log1p(np.maximum(v, 0.0))
        raise ValueError(f"unsupported normalize: {normalize!r}")

    for k in range(K):
        mask_k = labels_np == k
        for i in range(L):
            t = np.asarray(times_list[i], dtype=np.float64)
            X_np = (X_list[i].detach().cpu().numpy()
                    if isinstance(X_list[i], torch.Tensor)
                    else np.asarray(X_list[i], dtype=np.float64))
            members = X_np[mask_k] if mask_k.any() else np.empty((0, t.size))
            cond = condition_names[i]
            shown = (condition_labels[i] if condition_labels is not None
                     and i < len(condition_labels) else cond)
            center_raw = mean_curves[cond].to_numpy(float)[:, k]
            # Scale reference for this condition, taken from the start of the mean curve.
            ref = float(np.nanmean(center_raw[:3])) if center_raw.size else 1.0
            center = _apply_scale(center_raw, ref)
            c = float(params_mu[k, i, 0])
            beta = float(params_mu[k, i, 1])

            ax = axes[k, i]
            mem_c, ctr_c, fit_c = pal[i % len(pal)]

            # Pale thin line = member curve, thick line = cluster mean curve.
            # 1) member curves
            for mb in members:
                ax.plot(t, _apply_scale(mb, ref), "-", color=mem_c,
                        linewidth=member_lw, alpha=member_alpha, zorder=1)

            # 2) cluster mean curve - the thickest line
            ax.plot(t, center, "-", color=ctr_c, linewidth=center_lw,
                    zorder=5, solid_capstyle="round",
                    label="Cluster mean" if not legend_drawn else None)

            if formatter is not None:
                ax.yaxis.set_major_formatter(formatter)
                ax.xaxis.set_major_formatter(formatter)
            # sharey="row" gives one y scale per row, so only the leftmost column
            # needs tick numbers and a y label.
            if ylabel_only_first_col and i > 0:
                ax.tick_params(labelsize=8, labelleft=False)
                ax.set_ylabel("")
            else:
                ax.tick_params(labelsize=8)
                if ylabel_text is not None:
                    ax.set_ylabel(ylabel_text, fontsize=8)
                elif normalize == "baseline":
                    # y / y0, with y0 the module abundance at the start of quasi-time
                    ax.set_ylabel(r"Fold change (y / $y_0$)", fontsize=8)
                elif normalize == "log_baseline":
                    ax.set_ylabel(r"log1p(fold change)", fontsize=8)
                elif normalize == "mean":
                    ax.set_ylabel("Abundance / mean", fontsize=8)
                elif normalize == "log1p":
                    ax.set_ylabel("log1p(abundance)", fontsize=8)
            ax.set_title(
                f"{cluster_label_prefix}{k + 1} | {shown}   n={int(mask_k.sum())}",
                fontsize=9,
            )
            # Only the bottom row carries the x-axis label.
            if k == K - 1:
                ax.set_xlabel("Quasi-time (\u03c4)", fontsize=8.5)
            if not legend_drawn:
                legend_drawn = True

    if global_ylim is not None:
        for ax in axes.ravel():
            ax.set_ylim(*global_ylim)

    if show_legend:
        handles, lbls = axes[0, 0].get_legend_handles_labels()
        if handles:
            # Legend outside the axes, so it cannot cover the first row titles.
            fig.legend(handles, lbls, loc="outside upper center", ncol=2,
                       prop=_legend_font(9.0), frameon=False)
    return fig


def plot_cluster_profiles(
    *,
    data_scatter: List[pd.DataFrame],
    data_curve: Optional[List[pd.DataFrame]] = None,
    labels: np.ndarray,
    common_cols: Optional[Sequence[str]] = None,
    n_components: Optional[int] = None,
    condition_labels: Optional[Sequence[str]] = None,
    member_source: str = "qd_df",
    show_members: bool = True,
    show_mean: bool = True,
    show_mean_ci: bool = False,
    ci_alpha: float = 0.25,
    ci_z: float = 1.96,
    use_semilogy: bool = True,
    use_log_x: bool = False,
    layout: str = "combined",
    n_cols: int = 3,
    subplot_hw: Tuple[float, float] = (4.0, 3.6),
    panel_box_aspect: Optional[float] = 0.75,
    palette: Optional[Sequence[Tuple[str, str]]] = None,
    title: str = "Cluster profiles",
    panel_title_prefix: str = "M",
    panel_title_fontsize: float = 18.0,
    tick_labelsize: float = 12.0,
    show_grid: bool = False,
    xlabel: str = "Index",
    ylabel: Optional[str] = None,
    axis_label_fontsize: float = 12.0,
    linewidth_mean: float = 3.5,
    linewidth_member: float = 1.5,
    markersize_qd: float = 7.0,
    alpha_member_lines: float = 0.85,
    alpha_qd_marker: float = 0.9,
    x_margin: float = 0.1,
    y_margin: float = 0.2,
    show_legend: bool = True,
    legend_loc: str = "upper center",
    legend_ncol: Optional[int] = None,
    legend_fontsize: float = 11.0,
    legend_bbox: Tuple[float, float] = (0.5, 0.96),
    dpi: int = 300,
) -> Figure:
    """Draw the cluster profile figure of the EM fit.

    ``layout="combined"`` puts one cluster per panel with the conditions
    overlaid; ``"k_by_l"`` is K rows x L columns and ``"l_by_k"`` is L rows x K
    columns. The drawing area of every panel is locked to 4:3 by default
    (``panel_box_aspect=0.75``), so the panel ratio is independent of fonts,
    legends and shared axes.
    """
    if not data_scatter:
        raise ValueError("data_scatter must not be empty")
    if member_source not in ("qd_df", "curve"):
        raise ValueError(
            f"member_source must be 'qd_df' or 'curve', got: {member_source!r}"
        )
    if layout not in ("combined", "k_by_l", "l_by_k"):
        raise ValueError(
            "layout must be 'combined', 'k_by_l' or 'l_by_k', "
            f"got: {layout!r}"
        )
    if data_curve is None:
        data_curve = data_scatter
    if len(data_curve) != len(data_scatter):
        raise ValueError(
            f"data_curve / data_scatter length mismatch: "
            f"{len(data_curve)} vs {len(data_scatter)}"
        )

    n_conditions = len(data_scatter)
    labels_np = (
        labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor)
        else np.asarray(labels)
    ).astype(np.int64)

    if n_components is None:
        n_components = int(labels_np.max()) + 1 if labels_np.size > 0 else 1
    n_clusters = int(n_components)

    if condition_labels is None:
        condition_labels = [f"Cond {i + 1}" for i in range(n_conditions)]
    else:
        condition_labels = list(condition_labels)
        if len(condition_labels) < n_conditions:
            condition_labels = condition_labels + [
                f"Cond {i + 1}"
                for i in range(len(condition_labels), n_conditions)
            ]

    pal: List[Tuple[str, str]] = (
        list(palette) if palette is not None else list(_CLUSTER_PROFILE_DEFAULT_PALETTE)
    )
    while len(pal) < n_conditions:
        pal.extend(_CLUSTER_PROFILE_DEFAULT_PALETTE)

    n_cols = max(1, int(n_cols))
    if layout == "combined":
        n_cols_eff = min(n_cols, max(n_clusters, 1))
        n_rows = (n_clusters + n_cols_eff - 1) // n_cols_eff
    elif layout == "k_by_l":
        n_rows = n_clusters
        n_cols_eff = n_conditions
    else:
        n_rows = n_conditions
        n_cols_eff = n_clusters

    panel_w, panel_h = subplot_hw
    fig, axes = plt.subplots(
        n_rows,
        n_cols_eff,
        figsize=(panel_w * n_cols_eff, panel_h * n_rows),
        sharex=True,
        sharey=True,
        dpi=dpi,
    )
    if n_rows == 1 and n_cols_eff == 1:
        axes_arr = np.array([[axes]])
    elif n_rows == 1:
        axes_arr = np.asarray(axes).reshape(1, n_cols_eff)
    elif n_cols_eff == 1:
        axes_arr = np.asarray(axes).reshape(n_rows, 1)
    else:
        axes_arr = np.asarray(axes)

    y_label_eff = ylabel if ylabel is not None else (
        "Value (log scale)" if use_semilogy else "Value"
    )

    def _select_member_columns(df: pd.DataFrame) -> pd.DataFrame:
        if common_cols is not None:
            cols = [c for c in common_cols if c in df.columns]
            return df[cols]
        return df.iloc[:, : labels_np.size]

    def _plot_y(ax_: plt.Axes, x_, y_, *, fmt: str = "-", **kwargs) -> None:
        x_arr = np.asarray(x_, dtype=np.float64)
        y_arr = np.asarray(y_, dtype=np.float64)
        valid = np.isfinite(x_arr) & np.isfinite(y_arr)
        if use_log_x:
            valid &= x_arr > 0
        if use_semilogy:
            valid &= y_arr > 0
        if not valid.any():
            return

        x_plot = x_arr[valid]
        y_plot = y_arr[valid]
        if use_semilogy:
            ax_.semilogy(x_plot, np.maximum(y_plot, 1e-10), fmt, **kwargs)
        else:
            ax_.plot(x_plot, y_plot, fmt, **kwargs)

    def _style_axis(ax_: plt.Axes) -> None:
        if use_log_x:
            ax_.set_xscale("log")
        # Locking the box aspect keeps the panel ratio free of font metrics,
        # legend/title padding and shared y axes.
        if panel_box_aspect is not None:
            ax_.set_box_aspect(panel_box_aspect)
        # Do not write grid(show_grid, linestyle=...): with a False first argument
        # matplotlib turns the grid on anyway and warns.
        if show_grid:
            ax_.grid(True, linestyle=":", linewidth=0.8, alpha=0.35)
        else:
            ax_.grid(False)
        ax_.tick_params(labelsize=tick_labelsize)
        _set_chinese_axes(ax_, tick_labelsize)

    def _draw_cluster_condition(
        ax_: plt.Axes,
        *,
        cluster_idx: int,
        condition_idx: int,
        label_condition: bool,
    ) -> None:
        df_scatter = data_scatter[condition_idx]
        df_curve = data_curve[condition_idx]
        t_scatter = df_scatter.index.values.astype(np.float64)
        t_curve = df_curve.index.values.astype(np.float64)
        scatter_color, line_color = pal[condition_idx % len(pal)]
        mask = labels_np == cluster_idx
        n_in_cluster = int(mask.sum())

        if show_members:
            df_for_members = df_scatter if member_source == "qd_df" else df_curve
            t_for_members = t_scatter if member_source == "qd_df" else t_curve
            df_for_members = _select_member_columns(df_for_members)
            cluster_data = df_for_members.iloc[:, mask]

            if member_source == "qd_df":
                x_vals = np.repeat(t_for_members, n_in_cluster)
                y_vals = cluster_data.values.ravel()
                valid = np.isfinite(x_vals) & np.isfinite(y_vals)
                if use_log_x:
                    valid &= x_vals > 0
                if use_semilogy:
                    valid &= y_vals > 0
                if valid.any():
                    x_vals = x_vals[valid]
                    y_vals = y_vals[valid]
                    if use_semilogy:
                        ax_.semilogy(
                            x_vals,
                            np.maximum(y_vals, 1e-10),
                            "o",
                            markerfacecolor="none",
                            markeredgecolor=scatter_color,
                            markersize=markersize_qd,
                            alpha=alpha_qd_marker,
                            zorder=1,
                        )
                    else:
                        ax_.plot(
                            x_vals,
                            y_vals,
                            "o",
                            markerfacecolor="none",
                            markeredgecolor=scatter_color,
                            markersize=markersize_qd,
                            alpha=alpha_qd_marker,
                            zorder=1,
                        )
            else:
                for col in cluster_data.columns:
                    _plot_y(
                        ax_,
                        t_for_members,
                        cluster_data[col].values,
                        fmt="-",
                        color=scatter_color,
                        linewidth=linewidth_member,
                        alpha=alpha_member_lines,
                        zorder=1,
                    )

        df_curve_sub = _select_member_columns(df_curve)
        cluster_data_curve = df_curve_sub.iloc[:, mask]
        mean_points = cluster_data_curve.mean(axis=1).values.astype(np.float64)
        mean_x = t_curve
        mean_curve = mean_points

        if show_mean_ci and n_in_cluster > 0:
            std_curve = cluster_data_curve.std(axis=1).values.astype(np.float64)
            sem = std_curve / np.sqrt(max(n_in_cluster, 1))
            lo = mean_curve - ci_z * sem
            hi = mean_curve + ci_z * sem
            valid = np.isfinite(t_curve) & np.isfinite(lo) & np.isfinite(hi)
            if use_log_x:
                valid &= t_curve > 0
            if use_semilogy:
                valid &= hi > 0
                lo = np.maximum(lo, 1e-10)
                hi = np.maximum(hi, 1e-10)
            if valid.any():
                ax_.fill_between(
                    t_curve[valid],
                    lo[valid],
                    hi[valid],
                    color=line_color,
                    alpha=ci_alpha,
                    zorder=2,
                    linewidth=0,
                )

        if show_mean:
            _plot_y(
                ax_,
                mean_x,
                mean_curve,
                fmt="-",
                color=line_color,
                linewidth=linewidth_mean,
                label=condition_labels[condition_idx] if label_condition else None,
                zorder=10,
            )

    def _draw_panel(
        ax_: plt.Axes,
        *,
        cluster_idx: int,
        condition_indexes: Sequence[int],
        title_suffix: str = "",
        show_panel_title: bool = True,
    ) -> None:
        n_in_cluster = int((labels_np == cluster_idx).sum())
        if n_in_cluster == 0:
            empty_title = f"{panel_title_prefix}{cluster_idx + 1}\n(empty)"
            if title_suffix:
                empty_title = f"{empty_title}\n{title_suffix}"
            ax_.text(
                0.5,
                0.5,
                empty_title,
                transform=ax_.transAxes,
                ha="center",
                va="center",
                fontproperties=_font_at(panel_title_fontsize),
            )
            _style_axis(ax_)
            return

        for condition_idx in condition_indexes:
            _draw_cluster_condition(
                ax_,
                cluster_idx=cluster_idx,
                condition_idx=condition_idx,
                label_condition=(layout == "combined" and cluster_idx == 0),
            )

        if show_panel_title:
            panel_title = f"{panel_title_prefix}{cluster_idx + 1}({n_in_cluster})"
            if title_suffix:
                panel_title = f"{panel_title} | {title_suffix}"
            ax_.set_title(
                panel_title,
                fontproperties=_font_at(panel_title_fontsize),
            )
        ax_.margins(x=x_margin, y=y_margin)
        _style_axis(ax_)

    if layout == "combined":
        axes_flat = list(axes_arr.flat)
        for cluster_idx in range(n_clusters):
            _draw_panel(
                axes_flat[cluster_idx],
                cluster_idx=cluster_idx,
                condition_indexes=range(n_conditions),
            )
        for idx in range(n_clusters, len(axes_flat)):
            fig.delaxes(axes_flat[idx])
    elif layout == "k_by_l":
        for cluster_idx in range(n_clusters):
            for condition_idx in range(n_conditions):
                _draw_panel(
                    axes_arr[cluster_idx, condition_idx],
                    cluster_idx=cluster_idx,
                    condition_indexes=[condition_idx],
                    show_panel_title=False,
                )
        for condition_idx, condition_label in enumerate(condition_labels):
            axes_arr[0, condition_idx].set_title(
                str(condition_label),
                fontproperties=_font_at(panel_title_fontsize),
            )
        for cluster_idx in range(n_clusters):
            n_in_cluster = int((labels_np == cluster_idx).sum())
            axes_arr[cluster_idx, 0].set_ylabel(
                f"{panel_title_prefix}{cluster_idx + 1}\n({n_in_cluster})",
                fontproperties=_font_at(axis_label_fontsize),
            )
        for condition_idx in range(n_conditions):
            axes_arr[-1, condition_idx].set_xlabel(
                xlabel, fontproperties=font_prop
            )
    else:
        for condition_idx in range(n_conditions):
            for cluster_idx in range(n_clusters):
                _draw_panel(
                    axes_arr[condition_idx, cluster_idx],
                    cluster_idx=cluster_idx,
                    condition_indexes=[condition_idx],
                    show_panel_title=False,
                )
        for cluster_idx in range(n_clusters):
            n_in_cluster = int((labels_np == cluster_idx).sum())
            axes_arr[0, cluster_idx].set_title(
                f"{panel_title_prefix}{cluster_idx + 1}({n_in_cluster})",
                fontproperties=_font_at(panel_title_fontsize),
            )
            axes_arr[-1, cluster_idx].set_xlabel(
                xlabel, fontproperties=font_prop
            )
        for condition_idx, condition_label in enumerate(condition_labels):
            axes_arr[condition_idx, 0].set_ylabel(
                f"{condition_label}\n{y_label_eff}",
                fontproperties=_font_at(axis_label_fontsize),
            )

    if title:
        fig.suptitle(
            title,
            fontproperties=_font_at(axis_label_fontsize + 2, "bold"),
            y=0.995,
        )

    if show_legend:
        ncol_leg = legend_ncol if legend_ncol is not None else min(n_conditions, 6)
        handles = [
            plt.Line2D(
                [],
                [],
                color=pal[i % len(pal)][1],
                linewidth=linewidth_mean,
                label=condition_labels[i],
            )
            for i in range(n_conditions)
        ]
        fig.legend(
            handles=handles,
            loc=legend_loc,
            ncol=ncol_leg,
            prop=_legend_font(legend_fontsize),
            bbox_to_anchor=legend_bbox,
        )

    if layout == "combined":
        fig.text(
            0.5,
            0.01,
            xlabel,
            ha="center",
            fontproperties=_font_at(axis_label_fontsize),
        )
        fig.text(
            0.005,
            0.5,
            y_label_eff,
            va="center",
            rotation="vertical",
            fontproperties=_font_at(axis_label_fontsize),
        )

    # Reserve margins only for content that is actually rendered: the previous
    # fixed 0.89/0.93 reserve squeezed the panels.
    if layout == "combined":
        # combined draws shared x/y labels with fig.text, so it needs margins
        left, bottom = 0.035, 0.045
    else:
        # k_by_l / l_by_k keep their labels on the axes themselves
        left, bottom = 0.015, 0.02
    if show_legend:
        top = 0.89
    elif title:
        top = 0.93
    else:
        top = 0.995
    fig.tight_layout(rect=(left, bottom, 0.995, top))

    return fig


def plot_cluster_profiles_per_cluster(
    *,
    data_scatter: List[pd.DataFrame],
    data_curve: Optional[List[pd.DataFrame]] = None,
    labels: np.ndarray,
    common_cols: Optional[Sequence[str]] = None,
    n_components: Optional[int] = None,
    condition_labels: Optional[Sequence[str]] = None,
    member_source: str = "qd_df",
    show_members: bool = True,
    show_mean: bool = True,
    show_mean_ci: bool = False,
    ci_alpha: float = 0.25,
    ci_z: float = 1.96,
    use_semilogy: bool = False,
    use_log_x: bool = False,
    cluster_ids: Optional[Sequence[int]] = None,
    max_clusters: Optional[int] = None,
    palette: Optional[Sequence[Tuple[str, str]]] = None,
    panel_prefix: str = "M",
    panel_figsize: Tuple[float, float] = (4.0, 3.4),
    panel_box_aspect: Optional[float] = 0.75,
    title_fontsize: float = 18.0,
    tick_labelsize: float = 12.0,
    show_grid: bool = False,
    linewidth_mean: float = 3.5,
    linewidth_member: float = 1.5,
    markersize_qd: float = 7.0,
    alpha_member_lines: float = 0.85,
    alpha_qd_marker: float = 0.9,
    x_margin: float = 0.1,
    share_y: bool = True,
    share_y_pad: float = 0.05,
    dpi: int = 300,
    save_dir: Optional[str | Path] = None,
    show_legend: bool = False,
    legend_fontsize: float = 11.0,
) -> List[Figure]:
    """Draw one ``panel_figsize`` figure per cluster, matching funclu_v4.

    Unlike the grid version in :func:`plot_cluster_profiles` this is one figure
    per cluster. With ``share_y=True`` the y range is scanned over *all* clusters
    so the figures stay comparable, and the members are drawn before the mean
    curves (``zorder=10``) so the mean always sits on top.

    Returns:
        One :class:`~matplotlib.figure.Figure` per cluster; the caller closes them.
    """
    if not data_scatter:
        raise ValueError("data_scatter must not be empty")
    if member_source not in ("qd_df", "curve"):
        raise ValueError(
            f"member_source must be 'qd_df' or 'curve', got: {member_source!r}"
        )
    if data_curve is None:
        data_curve = data_scatter
    if len(data_curve) != len(data_scatter):
        raise ValueError(
            f"data_curve / data_scatter length mismatch: "
            f"{len(data_curve)} vs {len(data_scatter)}"
        )

    n_conditions = len(data_scatter)
    labels_np = (
        labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor)
        else np.asarray(labels)
    ).astype(np.int64)
    if n_components is None:
        n_components = int(labels_np.max()) + 1 if labels_np.size > 0 else 1
    n_clusters = int(n_components)

    if condition_labels is None:
        condition_labels = [f"Cond {i + 1}" for i in range(n_conditions)]
    else:
        condition_labels = list(condition_labels)
        while len(condition_labels) < n_conditions:
            condition_labels.append(f"Cond {len(condition_labels) + 1}")

    pal: List[Tuple[str, str]] = (
        list(palette) if palette is not None else list(_CLUSTER_PROFILE_DEFAULT_PALETTE)
    )
    if not pal:
        pal = list(_CLUSTER_PROFILE_DEFAULT_PALETTE)
    while len(pal) < n_conditions:
        pal.extend(_CLUSTER_PROFILE_DEFAULT_PALETTE)

    def _member_frame(condition_idx: int) -> pd.DataFrame:
        frame = (
            data_scatter[condition_idx]
            if member_source == "qd_df"
            else data_curve[condition_idx]
        )
        if common_cols is not None:
            cols = [c for c in common_cols if c in frame.columns]
            return frame[cols]
        return frame.iloc[:, : labels_np.size]

    def _finite(values: np.ndarray) -> np.ndarray:
        vals = np.asarray(values, dtype=np.float64).ravel()
        vals = vals[np.isfinite(vals)]
        if use_semilogy:
            vals = vals[vals > 0.0]
        return vals

    # Scan *all* clusters (not only the selected ones) so the y range stays
    # identical even when only some figures are exported.
    global_ylim: Optional[Tuple[float, float]] = None
    if share_y:
        vmin, vmax = np.inf, -np.inf
        for k in range(n_clusters):
            mask_k = labels_np == k
            if not mask_k.any():
                continue
            for condition_idx in range(n_conditions):
                sub = _member_frame(condition_idx).iloc[:, mask_k]
                if sub.shape[1] == 0:
                    continue
                members = _finite(sub.to_numpy(float))
                if members.size:
                    vmin = min(vmin, float(members.min()))
                    vmax = max(vmax, float(members.max()))
                means = _finite(sub.mean(axis=1).to_numpy(float))
                if means.size:
                    vmin = min(vmin, float(means.min()))
                    vmax = max(vmax, float(means.max()))
        if vmin == np.inf:
            vmin, vmax = 0.0, 1.0
        span = vmax - vmin
        pad = span * share_y_pad if span > 0 else 1.0
        lo, hi = vmin - pad, vmax + pad
        if use_semilogy and lo <= 0:
            lo = vmin / (1.0 + share_y_pad) if vmin > 0 else 1e-12
        global_ylim = (lo, hi)

    if cluster_ids is not None:
        wanted = {int(c) for c in cluster_ids}
        selected = [k for k in range(n_clusters) if k in wanted]
    else:
        selected = list(range(n_clusters))
    if max_clusters is not None:
        selected = selected[: max(0, int(max_clusters))]

    out_dir = Path(save_dir) if save_dir is not None else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    figures: List[Figure] = []
    for k in selected:
        mask = labels_np == k
        n_members = int(mask.sum())
        fig, ax = plt.subplots(figsize=panel_figsize, dpi=dpi)
        if use_log_x:
            ax.set_xscale("log")
        if use_semilogy:
            ax.set_yscale("log")

        if n_members == 0:
            ax.text(
                0.5, 0.5,
                f"{panel_prefix}{k + 1}\n(empty)",
                transform=ax.transAxes, ha="center", va="center",
                fontproperties=_font_at(title_fontsize),
            )
        else:
            # 1) all member curves first (v4 draws members below)
            if show_members:
                for condition_idx in range(n_conditions):
                    member_color = pal[condition_idx % len(pal)][0]
                    frame = _member_frame(condition_idx)
                    sub = frame.iloc[:, mask]
                    if sub.shape[1] == 0:
                        continue
                    if member_source == "qd_df":
                        times = frame.index.to_numpy(float)
                        x_vals = np.repeat(times, n_members)
                        y_vals = sub.to_numpy(float).ravel()
                        valid = np.isfinite(x_vals) & np.isfinite(y_vals)
                        if use_semilogy:
                            valid &= y_vals > 0.0
                        if use_log_x:
                            valid &= x_vals > 0.0
                        if valid.any():
                            ax.plot(
                                x_vals[valid], y_vals[valid], "o",
                                markerfacecolor="none",
                                markeredgecolor=member_color,
                                markersize=markersize_qd,
                                alpha=alpha_qd_marker,
                                linestyle="none",
                                zorder=1,
                            )
                    else:
                        times = frame.index.to_numpy(float)
                        for col in sub.columns:
                            y_vals = sub[col].to_numpy(float)
                            valid = np.isfinite(times) & np.isfinite(y_vals)
                            if use_semilogy:
                                valid &= y_vals > 0.0
                            if use_log_x:
                                valid &= times > 0.0
                            if not valid.any():
                                continue
                            ax.plot(
                                times[valid], y_vals[valid],
                                color=member_color,
                                linewidth=linewidth_member,
                                alpha=alpha_member_lines,
                                zorder=1,
                            )

            # 2) mean curves last, so they sit on top of the members
            for condition_idx in range(n_conditions):
                mean_color = pal[condition_idx % len(pal)][1]
                frame = _member_frame(condition_idx)
                sub = frame.iloc[:, mask]
                if sub.shape[1] == 0:
                    continue
                times = frame.index.to_numpy(float)
                mean_vals = sub.mean(axis=1).to_numpy(float)

                if show_mean_ci:
                    sem = sub.std(axis=1).to_numpy(float) / np.sqrt(max(n_members, 1))
                    lo_band, hi_band = mean_vals - ci_z * sem, mean_vals + ci_z * sem
                    valid = np.isfinite(times) & np.isfinite(lo_band) & np.isfinite(hi_band)
                    if use_semilogy:
                        valid &= hi_band > 0.0
                        lo_band = np.maximum(lo_band, 1e-10)
                        hi_band = np.maximum(hi_band, 1e-10)
                    if use_log_x:
                        valid &= times > 0.0
                    if valid.any():
                        ax.fill_between(
                            times[valid], lo_band[valid], hi_band[valid],
                            color=mean_color, alpha=ci_alpha, zorder=2, linewidth=0,
                        )

                if show_mean:
                    valid = np.isfinite(times) & np.isfinite(mean_vals)
                    if use_semilogy:
                        valid &= mean_vals > 0.0
                    if use_log_x:
                        valid &= times > 0.0
                    if valid.any():
                        ax.plot(
                            times[valid], mean_vals[valid],
                            color=mean_color,
                            linewidth=linewidth_mean,
                            label=condition_labels[condition_idx],
                            zorder=10,
                        )

        if global_ylim is not None:
            ax.set_ylim(*global_ylim)
            ax.margins(x=x_margin)
        else:
            ax.margins(x=x_margin, y=0.2)
        if panel_box_aspect is not None:
            ax.set_box_aspect(panel_box_aspect)

        ax.set_title(
            f"{panel_prefix}{k + 1}({n_members})",
            fontproperties=_font_at(title_fontsize),
        )
        if show_grid:
            ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.35)
        else:
            ax.grid(False)
        ax.tick_params(labelsize=tick_labelsize)
        _set_chinese_axes(ax, tick_labelsize)
        if show_legend:
            ax.legend(prop=_legend_font(legend_fontsize), frameon=False)

        fig.tight_layout()
        if out_dir is not None:
            fig.savefig(
                out_dir / f"{panel_prefix}{k + 1}.png",
                dpi=dpi, bbox_inches="tight",
            )
        figures.append(fig)

    return figures


def stitch_cluster_figures(
    figures: Sequence[Figure],
    *,
    n_cols: int = 3,
    background: str = "white",
) -> Figure:
    """Stitch equally sized figures into one grid with ``n_cols`` columns.

    Every input figure is rasterised at its own dpi first, so a cell keeps its
    original aspect ratio instead of being squeezed by shared axes or
    legend/title padding; the aspect ratio of the result follows from the row and
    column count. The last row is padded when it is not full.

    Returns:
        A single :class:`~matplotlib.figure.Figure`; the caller closes it.
    """
    figs = [fig for fig in figures if fig is not None]
    if not figs:
        raise ValueError("figures must not be empty")

    n_cols = max(1, int(n_cols))
    n_rows = (len(figs) + n_cols - 1) // n_cols

    tiles = []
    for fig in figs:
        # Interactive backends have no buffer_rgba; swap in an Agg canvas in place
        if not hasattr(fig.canvas, "buffer_rgba"):
            from matplotlib.backends.backend_agg import FigureCanvasAgg

            FigureCanvasAgg(fig)
        fig.canvas.draw()
        tiles.append(np.asarray(fig.canvas.buffer_rgba()))

    cell_h = max(tile.shape[0] for tile in tiles)
    cell_w = max(tile.shape[1] for tile in tiles)

    fill = (np.asarray(to_rgb(background)) * 255).astype(np.uint8)
    canvas = np.empty((cell_h * n_rows, cell_w * n_cols, 3), dtype=np.uint8)
    canvas[...] = fill
    for index, tile in enumerate(tiles):
        row, col = divmod(index, n_cols)
        height, width = tile.shape[:2]
        canvas[
            row * cell_h:row * cell_h + height,
            col * cell_w:col * cell_w + width,
        ] = tile[:, :, :3]

    dpi = float(figs[0].dpi)
    out = plt.figure(
        figsize=(canvas.shape[1] / dpi, canvas.shape[0] / dpi), dpi=dpi,
    )
    axis = out.add_axes((0.0, 0.0, 1.0, 1.0))
    axis.imshow(canvas, interpolation="none")
    axis.set_axis_off()
    return out


def _fit_mean_curve_power_sample(
    t_curve: np.ndarray,
    y_mean: np.ndarray,
    t_sample: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit ``y = a * t^b`` to the mean curve and sample it on ``t_sample``.

    Same log-log linear regression as the fit used inside ``FunClu``. Values are
    finite only inside the valid domain ``[t_lo, t_hi]`` of ``t_curve`` and
    ``NaN`` outside it, so the caller can fall back.
    """
    x_arr = np.asarray(t_curve, dtype=np.float64)
    y_arr = np.asarray(y_mean, dtype=np.float64)
    xg = np.asarray(t_sample, dtype=np.float64)
    out = np.full(xg.shape, np.nan, dtype=np.float64)

    mask = np.isfinite(x_arr) & np.isfinite(y_arr) & (x_arr > 0) & (y_arr > 0)
    if int(mask.sum()) < 2:
        return xg, out
    a, b = fit_power_loglinear(
        x_arr[mask],
        y_arr[mask],
        clip_a=(1e-12, np.inf),
        clip_b=(-10.0, 10.0),
    )
    x_lo = float(np.min(x_arr[mask]))
    x_hi = float(np.max(x_arr[mask]))
    in_range = np.isfinite(xg) & (xg >= x_lo) & (xg <= x_hi)
    with np.errstate(over="ignore", invalid="ignore"):
        out[in_range] = a * np.power(xg[in_range], b)
    return xg, out


def plot_bic_elbow(
    *,
    bic_results,  # pd.DataFrame
    best_K: Optional[int] = None,
    candidate_Ks: Optional[Sequence[int]] = None,
    auto_select_best: bool = True,
    figsize: Tuple[float, float] = (8.0, 5.0),
    dpi: int = 200,
    title: str = "BIC Elbow Plot",
    xlabel: str = "Number of clusters (K)",
    ylabel: str = "BIC",
    best_k_color: str = "#e74c3c",
    line_color: str = "#2c3e50",
    marker_color: str = "#2c3e50",
) -> Figure:
    """BIC elbow plot for choosing the optimal number of clusters.

    BIC is drawn against K as a connected line with circular markers; the best K
    (lowest BIC) is highlighted with a distinct marker and annotation.
    """
    df = bic_results
    df_valid = df.dropna(subset=["BIC"]).copy()
    if df_valid.empty:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.text(0.5, 0.5, "No valid BIC results", ha="center", va="center",
                transform=ax.transAxes, fontproperties=font_prop)
        return fig

    ks = df_valid["K"].values.astype(int)
    bics = df_valid["BIC"].values

    if best_K is None and auto_select_best:
        converged = df_valid[df_valid["converged"] == True]
        if not converged.empty:
            best_K = int(converged.loc[converged["BIC"].idxmin(), "K"])
        else:
            best_K = int(ks[bics.argmin()])

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    ax.plot(ks, bics, "-o", color=line_color, markerfacecolor=marker_color,
            markersize=8, linewidth=2, zorder=2, label="BIC")

    candidate_set = {int(k) for k in (candidate_Ks or [])}
    candidate_x = [int(k) for k in ks if int(k) in candidate_set]
    candidate_y = [bics[ks.tolist().index(k)] for k in candidate_x]
    if candidate_x:
        ax.plot(
            candidate_x,
            candidate_y,
            "o",
            color=best_k_color,
            markersize=11,
            zorder=3,
            label="Checkmark candidates",
        )

    if best_K is not None and int(best_K) in ks.tolist():
        best_k_int = int(best_K)
        best_bic = bics[ks.tolist().index(best_k_int)]
        ax.plot([best_k_int], [best_bic], "o", color=best_k_color,
                markersize=14, zorder=4)
        ax.annotate(
            f"Best K={best_k_int}",
            xy=(best_k_int, best_bic),
            xytext=(best_k_int + 1.2, best_bic),
            fontsize=11,
            fontweight="bold",
            color=best_k_color,
            fontproperties=font_prop,
            arrowprops=dict(arrowstyle="->", color=best_k_color, lw=1.5),
        )

    ax.set_xlabel(xlabel, fontproperties=font_prop)
    ax.set_ylabel(ylabel, fontproperties=font_prop)
    ax.set_title(title, fontweight="bold", fontproperties=font_prop)
    ax.set_xticks(ks)
    ax.grid(True, linestyle=":", alpha=0.5)
    _set_chinese_axes(ax)

    fig.tight_layout()

    return fig
