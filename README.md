# idopNetwork

A toolkit for inferring directed network structure from multi-condition
biological and medical data.

This repository is the **algorithm-layer extraction** of the original
Streamlit application: the UI, authentication, configuration screens, log
popups and packaging are removed, leaving only the core algorithms split into
nine single-responsibility scripts.

> The algorithms, mathematical formulation, parameter defaults and computation
> steps are identical to the original application's implementation.

---

## Pipeline and scripts

```
data import -> quasi-dynamic (pseudo-time) analysis -> [functional clustering]
                                                       ^ two-layer mode only
   -> network reconstruction -> GLMY analysis -> save results and figures
```

| Step | Script | Responsibility |
| --- | --- | --- |
| Config | `config.py` | All tunable parameters: input files, preprocessing, fitting, clustering, network, GLMY, output directory |
| 1 Import | `DataPreprocessor.py` | Read CSV, subset rows/columns, drop columns, column transform |
| 1 Quasi-dynamic | `QuasiDynamicAnalyzer.py` | Column transforms, quasi-dynamic series, power-law fit `y = a*x^b`, SWT denoising, **power-law fit figures** |
| 2 Clustering | `FunctionalClusterer.py` | FunClu v4 EM clustering, BIC scan, **clustering figures** (profiles / per-module / BIC / initialisation) |
| 3 Network | `NetworkReconstructor.py` | Legendre basis expansion + constrained sparse regression (`IDOPRegressor` / `StaticIDOPRegressor`), edge selector, dynamic solver, **network / effect-decomposition / heatmap figures**, interactive HTML |
| 4 GLMY | `GlmyAnalyzer.py` | Graph construction, persistent path homology (including sign split), **barcode and Betti-number figures**; includes the pure-Python persistent path homology kernel `Digraph` |
| Visualisation | `NetworkVisualizer.py` | Unified entry point for all plotting functions plus `setup_matplotlib()` font setup |
| Results | `ResultManager.py` | `results/` tree, writing matrices / edge lists / figures |
| Entry point | `main.py` | Drives the whole pipeline from `config.py` (single or two layers) |

---

## Single-layer vs two-layer networks

Controlled by `config.NETWORK_LAYER`:

```python
NETWORK_LAYER = "single"   # one network over all variables (clustering skipped)
NETWORK_LAYER = "two"      # cluster first, then build macro + micro networks
```

Two-layer conventions (the layers are called `inter_cluster` and
`intra_cluster`):

- **Macro layer (`inter_cluster`)**: design matrix = the cluster centre curves
  from `FunClu.get_cluster_curves()` (columns `M1..Mk`); response = the
  arithmetic mean of the members' `quasi_dynamic` columns.
- **Micro layer (`intra_cluster`)**: for each module, take its member columns
  from `cluster_labels` (`labels[i]` aligned positionally to `common_cols[i]`)
  and fit a network on that condition's `curve_sample` / `quasi_dynamic`
  sub-table. Modules with fewer than two members are skipped.
- Both layers use the same network parameters and the same plotting and
  analysis pipeline.

---

## Installation and usage

```bash
pip install -r requirements.txt
python main.py
```

Common settings (`config.py`):

```python
DATA_FILES = [(r"data\example_static.csv", "example")]  # (path, condition name)
DATA_MODE = "quasi_dynamic"     # or "dynamic" (time series -> SWT denoising)
NETWORK_LAYER = "single"        # or "two"
CLUSTER_USE_BIC = True          # select K automatically by BIC scan
BASIS_MAX_ORDER = 1
LASSO_ALPHA = 0.05
ENABLE_GLMY = True
```

Using individual modules:

```python
from DataPreprocessor import DataPreprocessor
from QuasiDynamicAnalyzer import fit_uploaded_csv
from NetworkReconstructor import StaticIDOPRegressor
from GlmyAnalyzer import run_glmy

raw = DataPreprocessor.load_data("data/example_static.csv")
results = fit_uploaded_csv(open("data/example_static.csv", "rb").read())
model = StaticIDOPRegressor(max_order=1)
model.fit(results["curve_sample"], results["quasi_dynamic"],
          power_function_params=results["curve_params"])
adj = model.adjacency_matrix(results["curve_sample"])
homology = run_glmy(from_to_df)["homology"]
```

---

## Output layout

Every step writes its **result tables and figures**, together with the
**underlying plotting data**, so figures can be regenerated without re-running
the analysis:

```
results/<condition>/
├── 1_quasi_dynamic/   raw_data / quasi_dynamic / curve_params / curve_sample
│                      + power_law_fit.png
├── 2_clustering/      cluster_labels / cluster_sizes / bic_scores
│                      + curve_sample_common / quasi_dynamic_common
│                      + cluster_profiles.png / cluster_profiles_per_module.png
│                        / bic_elbow.png / cluster_initialization.png
├── 3_network/         design_matrix / response / predicted / adjacency_matrix
│   │                  / edge_list / edge_supports / effect_<var>.csv
│   │                  + network_topology.png / effect_decomposition.png
│   │                    / adjacency_heatmap.png
│   ├── inter_cluster/     two-layer: macro network (with macro_center_*.csv)
│   └── intra_cluster/Mk/  two-layer: per-module (micro) networks
├── 4_glmy/            from_to_*.csv / homology_*.json / betti_counts_*.json
│   │                  + barcode_*.png / barcode_combined_*.png
│   │                    / betti_numbers_*.png
│   └── modules/           two-layer: GLMY results per micro module
├── 5_interactive/     interactive HTML network figures
└── run_config.json    snapshot of all parameters used in this run
```

**Key artefacts**: power-law fit figure
`1_quasi_dynamic/power_law_fit.png`; clustering figures
`2_clustering/cluster_profiles*.png` (two-layer mode only); network topology
`3_network/**/network_topology.png`; effect decomposition
`3_network/**/effect_decomposition.png`; GLMY Betti-number and barcode figures
under `4_glmy/**/`.

> The original application displayed Betti numbers as JSON text rather than as
> a figure. `NetworkVisualizer.plot_betti_counts` renders the same counts
> (`bar_counts`, same convention) as a bar chart and introduces no additional
> homology computation.

---

## Implementation notes

Only software-form concerns were removed during extraction: Streamlit
rendering branches and the `show_in_streamlit` parameters, unused UI
placeholder functions, command-line entry points, dead imports, and the
non-pipeline utility scripts.
