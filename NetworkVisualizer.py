# -*- coding: utf-8 -*-
"""Unified visualisation entry point.

Re-exports the plotting functions of the curve-fitting, clustering, network and
GLMY modules, and provides ``setup_matplotlib()`` to initialise the bundled CJK
font. No computation or plotting is implemented here.
"""

import os
import warnings
from pathlib import Path

from QuasiDynamicAnalyzer import (
    plot_curve_fitting,
    plot_curve_fitting_compare,
    plot_power_law_fit,
    plot_power_law_summary,
)
from FunctionalClusterer import (
    plot_bic_elbow,
    plot_cluster_fit_grid,
    plot_cluster_profiles,
    plot_cluster_profiles_per_cluster,
    plot_initialization_grid,
    stitch_cluster_figures,
)
from NetworkReconstructor import (
    plot_adjusted_matrix_heatmap,
    plot_effect,
    _effect_label_mode,
    plot_network,
    plot_interactive_network_html,
    network_edges_from_adjacency,
)
from GlmyAnalyzer import (
    bar_counts,
    bar_counts_split,
    plot_glmy_barcode,
    plot_glmy_barcode_combined,
    plot_glmy_barcode_split,
)

__all__ = [
    "setup_matplotlib",
    "plot_curve_fitting",
    "plot_curve_fitting_compare",
    "plot_power_law_fit",
    "plot_power_law_summary",
    "plot_bic_elbow",
    "plot_cluster_profiles",
    "plot_cluster_profiles_per_cluster",
    "plot_initialization_grid",
    "plot_cluster_fit_grid",
    "stitch_cluster_figures",
    "plot_adjusted_matrix_heatmap",
    "plot_effect",
    "_effect_label_mode",
    "plot_network",
    "plot_interactive_network_html",
    "network_edges_from_adjacency",
    "plot_glmy_barcode",
    "plot_glmy_barcode_combined",
    "plot_glmy_barcode_split",
    "plot_betti_counts",
    "bar_counts",
    "bar_counts_split",
]


def setup_matplotlib(font_candidates=None) -> str:
    """Set the global plotting font so CJK text and minus signs render correctly.

    Candidates are probed in order. The first font found is applied both to the
    global rcParams and to the module-level ``font_prop`` of the plotting
    modules. Returns the resolved font name, or ``""`` if none was found.
    """
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    if font_candidates is None:
        font_candidates = (
            os.path.join("fonts", "SimHei.ttf"),
            "SimHei.ttf",
            os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts", "simhei.ttf"),
            os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts", "msyh.ttc"),
        )

    for candidate in font_candidates:
        path = Path(candidate)
        if not path.is_file():
            continue
        fm.fontManager.addfont(str(path))
        name = fm.FontProperties(fname=str(path)).get_name()
        plt.rcParams["font.sans-serif"] = [name, "Microsoft YaHei", "PingFang SC",
                                           "Heiti SC", "DejaVu Sans"]
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["axes.unicode_minus"] = False

        import QuasiDynamicAnalyzer as _qda
        import FunctionalClusterer as _fc
        import NetworkReconstructor as _nr
        import GlmyAnalyzer as _ga

        for module in (_qda, _fc, _nr, _ga):
            if hasattr(module, "font_prop"):
                module.font_prop = fm.FontProperties(fname=str(path))
        return name

    warnings.warn(
        "No CJK font found (fonts/SimHei.ttf). CJK glyphs will render as boxes; "
        "numerical results are unaffected."
    )
    return ""


def plot_betti_counts(homology, title: str = "Betti numbers"):
    """Bar chart of the number of bars per homology dimension (beta_d).

    Uses the same counts as ``bar_counts``; no additional homology computation.
    """
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure  # noqa: F401

    counts = bar_counts(homology)
    labels = [f"$\\beta_{{{d}}}$" for d in sorted(counts, key=int)]
    values = [counts[str(d)] for d in sorted(counts, key=int)]

    fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=200)
    bars = ax.bar(labels, values, color="#2F7ED8", width=0.55)
    for rect, value in zip(bars, values):
        ax.text(rect.get_x() + rect.get_width() / 2.0, value,
                str(int(value)), ha="center", va="bottom", fontsize=11)
    ax.set_xlabel("Homology dimension", fontproperties=_font_or_none())
    ax.set_ylabel("Betti number (number of bars)", fontproperties=_font_or_none())
    ax.set_title(title, fontproperties=_font_or_none())
    ax.set_ylim(0, max(values + [1]) * 1.25)
    ax.grid(True, axis="y", linestyle=":", alpha=0.45)
    fig.tight_layout()
    return fig


def _font_or_none():
    """Placeholder for reusing the global font; returns None (matplotlib default)."""
    import matplotlib.pyplot as plt

    if plt.rcParams.get("font.sans-serif"):
        return None
    return None
