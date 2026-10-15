# -*- coding: utf-8 -*-
"""Central configuration for the idopNetwork pipeline.

All tunable values live here: input files, preprocessing and fitting settings,
clustering, network reconstruction, GLMY analysis and output directories.
``main.py`` only drives the pipeline from this configuration.

Sparsity comparability
----------------------
All conditions share the same sparsity controls (``LASSO_ALPHA``,
``LASSO_WINDOWS``, ``LASSO_THRESHOLD``, ``MAX_INTERACTIONS``,
``BASIS_MAX_ORDER``), so the edge-selection rule is identical across
conditions and any difference in network density or topology comes from the
data rather than from the settings. ``MAX_INTERACTIONS`` is a hard budget on
the in-degree of each node and is the key switch that keeps the sparsity of
all conditions on the same scale, which is what makes GLMY results comparable
across conditions. Biological networks are sparse, hence the tight budget
(in-degree <= 3).
"""

# ---------------------------------------------------------------------------
# 1. Input files: each entry is (path, condition name).
#    Rows are samples; the first column is the row index. All conditions share
#    the same feature set.
# ---------------------------------------------------------------------------
DATA_FILES = [
    (r"data\Yachida_CRC_raw\Healthy.csv", "Healthy"),
    (r"data\Yachida_CRC_raw\Adenoma.csv", "Adenoma"),
    (r"data\Yachida_CRC_raw\Early_CRC.csv", "Early_CRC"),
    (r"data\Yachida_CRC_raw\Late_CRC.csv", "Late_CRC"),
]

# ---------------------------------------------------------------------------
# 2. Import and preprocessing
# ---------------------------------------------------------------------------
N_SAMPLES = None            # keep the first n rows; None keeps all
N_FEATURES = None           # keep the first n columns; None keeps all
DROP_FEATURES = None        # column names to drop; None or [] drops nothing

FIRST_TRANSFORM = "None"        # first column transformation
SECOND_TRANSFORM = "Log10_1p"   # second column transformation
# Supported: "None", "Log10_1p", "Minmax_0_1", "Z_min_add1".

# The choice of transform matters a great deal for zero-inflated microbiome
# data (~78% zeros). "Z_min_add1" shifts every zero to 1 (log10(2) = 0.30103),
# which flattens most curves (median R^2 ~ 0.008, no feature above R^2 0.25).
# "None" keeps zeros at zero so they are ignored by the fit (median R^2 ~ 0.04
# with 53 features above 0.25). Do not use "Z_min_add1" on zero-inflated data.

TRIM_PERCENT = 1.0          # percentage of leading rows removed from the quasi-dynamic series
N_CURVE_SAMPLES = 30        # number of samples on the power-law curve

# -- zero handling ----------------------------------------------------------
IGNORE_ZERO = True
# Ignore zeros (and negative values) when fitting the power law. With ~77%
# zeros in the raw data this is required, otherwise many features collapse to
# near-flat curves and clustering becomes meaningless. Has no extra effect when
# FIRST_TRANSFORM="Z_min_add1", because that transform already lifts zeros away
# from zero.

# -- variable filtering -----------------------------------------------------
MIN_POSITIVE_PCT = 0.20
# A feature must have a positive-observation fraction >= this value in *every*
# condition, otherwise it is dropped. The fraction (rather than an absolute
# count) is used because sample sizes differ widely; an absolute threshold
# above the smallest condition would remove every feature. 0 disables.
MIN_POSITIVE = 0
# Additional absolute lower bound (count); the stricter of the two applies.
MIN_FIT_R2_PCT = 40
# Relative goodness-of-fit filter: keep the (100 - value)% features with the
# highest global mean R^2. 40 drops the worst 40%. A quantile is used rather
# than an absolute threshold because power-law fits are weak overall here
# (median R^2 ~ 0.10). 0 disables.
MIN_FIT_R2_ABS = None
# Optional absolute R^2 lower bound, intersected with the quantile filter.
FILTER_CONSENSUS = True
# True: keep features that pass in *all* conditions simultaneously, so the
# module partition is identical across conditions. False: keep a feature if any
# condition passes it.

# -- module balancing (optional) --------------------------------------------
MODULE_BALANCE = True
# Reassign members of oversized modules to undersized ones by parameter
# similarity so that module sizes are comparable. False keeps the raw EM
# partition.
MODULE_MIN_SIZE = 8
MODULE_MAX_SHARE = None
# None: equal-size mode, target module sizes differ by at most one member.
# A number: cap mode, the maximum module size is value * total features.

# -- clustering diagnostic figure axes (plotting only) ----------------------
CLUSTER_FIG_SHAREY = "row"
# "row"  shared y-axis per cluster row -> directly comparable across conditions
# "all"  one shared y-axis for all panels
# "col"  shared y-axis per condition column
# False  independent axes
CLUSTER_FIG_NORMALIZE = "baseline"
# "none"          raw fitted values
# "baseline"      divide by each curve's own starting value (fold change)
# "mean"          divide by the time-averaged value
# "log1p"         compress large values, expand small ones
# "log_baseline"  baseline first, then log
# These transforms affect plotting only; clustering, network reconstruction and
# GLMY always use the raw fitted values.

# ---------------------------------------------------------------------------
# 3. Data type: selects the route through quasi-dynamic analysis.
#    "quasi_dynamic"  static data -> column transform -> order rows by row sum
#                     into a quasi-dynamic series -> power-law fit y = a * x^b
#    "dynamic"        time series -> per-channel SWT soft-threshold denoising
# ---------------------------------------------------------------------------
DATA_MODE = "quasi_dynamic"
SWT_WAVELET = "rbio3.9"     # dynamic mode: wavelet basis
SWT_LEVEL = 2               # dynamic mode: SWT decomposition level
SWT_ALPHA = 0.5             # dynamic mode: threshold alpha * sigma * sqrt(2 log N)

# ---------------------------------------------------------------------------
# 4. Network layers: single or two-layer.
#    "single"  one network over all variables
#    "two"     cluster first, then build an inter-module (macro) network plus
#              one intra-module (micro) network per module
# ---------------------------------------------------------------------------
NETWORK_LAYER = "two"
TWO_LAYER_MODULE_PREFIX = "M"   # module name prefix (M1, M2, ...)

# ---------------------------------------------------------------------------
# 5. Functional clustering (two-layer mode only).
#    A single joint clustering is performed, so the module partition is shared
#    by all conditions and therefore comparable.
# ---------------------------------------------------------------------------
CLUSTER_K = 6               # number of modules
CLUSTER_K_MIN = 2           # BIC scan lower bound
CLUSTER_K_MAX = 8           # BIC scan upper bound (capped by valid feature count)
CLUSTER_USE_BIC = False     # False uses CLUSTER_K directly, without a BIC scan
CLUSTER_MAX_ITER = 1000     # EM iterations
CLUSTER_TOL = 1e-4          # EM convergence tolerance
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 6. Network reconstruction (IDOP effect decomposition), shared by all
#    conditions. Biological networks are sparse: the mean in-degree is close
#    to MAX_INTERACTIONS.
# ---------------------------------------------------------------------------
BASIS_MAX_ORDER = 1         # highest order of the Legendre expansion
MAX_INTERACTIONS = 2        # hard in-degree budget per target (0 = unlimited)
NONNEG_SELF = True          # non-negativity constraint on self-effects
LASSO_ALPHA = 0.15          # edge-selector LASSO strength (larger = sparser)
LASSO_WINDOWS = 1           # number of edge-selector windows
LASSO_THRESHOLD = 0.9       # selection-frequency threshold (larger = sparser)

# ---------------------------------------------------------------------------
# 7. GLMY network analysis
# ---------------------------------------------------------------------------
ENABLE_GLMY = True          # compute GLMY persistent path homology (barcode + Betti)
GLMY_DIM = 3                # maximum homology dimension

# ---------------------------------------------------------------------------
# 8. Visualisation and output
# ---------------------------------------------------------------------------
ENABLE_INTERACTIVE_HTML = True  # also write interactive HTML network figures
PLOT_FIT_NCOLS = 4              # columns of the power-law fit grid
PLOT_FIT_MAX_ROWS = 7           # maximum rows per page

# -- publication-style power-law fit figure (plot_power_law_fit) ------------
PLOT_FIT_SCOPE = "retained"
# "retained" only variables kept by filtering (consistent with downstream steps)
# "all"      all variables
# "both"     both
PLOT_FIT_LOGY = True
# Log y-axis is recommended: retained features span roughly 16-fold in maximum
# value and low-abundance modules sit far below high-abundance ones. A linear
# axis flattens the low-abundance curves. Note that zeros cannot be shown on a
# log axis.
PLOT_FIT_SUMMARY = True         # also draw the parameter summary figure

# -- effect decomposition figure (_effect_label_mode) -----------------------
# Variable names are written at the end of their own curves instead of using a
# legend.
EFFECT_NCOLS = 3                # 3 columns give wider panels for end labels
EFFECT_PANEL_WIDTH = 5.5        # panel width in inches
EFFECT_LABEL_FONTSIZE = 7.0     # end-label font size (pt)
EFFECT_DPI = 150                # raster resolution; a vector PDF is not limited
EFFECT_VAR_LABEL_ALIGN = "left"
# "left"  the name starts at the curve end and extends rightwards
# "right" the name ends at the curve end and extends leftwards
FIGURE_DPI = 300                # figure resolution used by ResultManager
RESULT_BASE_DIR = "results"     # results/<condition>/<step>/...
FONT_CANDIDATES = (             # font search order for plotting
    "fonts/SimHei.ttf",
    "SimHei.ttf",
)
