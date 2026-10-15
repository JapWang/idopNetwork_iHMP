# -*- coding: utf-8 -*-
"""Result file management.

Owns the output layout: every pipeline step writes its tables and figures here,
together with the underlying plotting data, so figures can be regenerated later
without re-running the analysis.

Directory layout
----------------
    results/
    └── <condition>/
        ├── 1_quasi_dynamic/    curve fitting: quasi-dynamic table, parameters,
        │                       curve samples, power-law fit figures
        ├── 2_clustering/       functional clustering: labels, BIC scan,
        │                       initialisation / profile / BIC figures
        ├── 3_network/          network: adjacency, edge list, design matrix,
        │                       predictions, effect decomposition, network figures
        │   └── modules/        per-module (intra-cluster) networks in two-layer mode
        ├── 4_glmy/             GLMY: from_to edge list, homology intervals,
        │                       barcode figures, Betti-number figures
        │   └── modules/
        └── 5_interactive/      interactive HTML network figures

This module only decides where results go and what they are called.
"""

import os
import re
from typing import List, Optional

import numpy as np
import pandas as pd

# Step sub-directory names (one-to-one with the steps in main.py).
STEP_CURVE = "1_quasi_dynamic"
STEP_CLUSTER = "2_clustering"
STEP_NETWORK = "3_network"
STEP_GLMY = "4_glmy"
STEP_INTERACTIVE = "5_interactive"


class ResultManager:
    """Builds the result tree and writes data / figures to disk."""

    def __init__(self, base_dir: str = "results", dpi: int = 300):
        self.base_dir = base_dir
        self.dpi = int(dpi)
        os.makedirs(self.base_dir, exist_ok=True)

    # -- paths ---------------------------------------------------------------
    @staticmethod
    def _sanitize(name: str) -> str:
        """Strip characters that are not allowed in file or folder names."""
        if not name:
            return ""
        return re.sub(r'[\\/*?:"<>|]', "_", str(name))

    def get_dir(self, condition: str = "", step: str = "") -> str:
        """Build (and create) a result directory."""
        parts = [self.base_dir]
        if condition:
            parts.append(self._sanitize(condition))
        if step:
            parts.append(step)
        path = os.path.join(*parts)
        os.makedirs(path, exist_ok=True)
        return path

    def get_path(self, filename: str, condition: str = "", step: str = "") -> str:
        return os.path.join(self.get_dir(condition, step), self._sanitize(filename))

    # -- data ----------------------------------------------------------------
    def save_dataframe(
        self,
        df: pd.DataFrame,
        filename: str,
        condition: str = "",
        step: str = "",
        index: bool = True,
    ) -> str:
        """Write a DataFrame as CSV (utf-8-sig, readable by Excel)."""
        path = self.get_path(filename, condition, step)
        pd.DataFrame(df).to_csv(path, index=index, encoding="utf-8-sig")
        return path

    def save_matrix(
        self,
        matrix: np.ndarray,
        filename: str,
        index_names: Optional[List[str]] = None,
        col_names: Optional[List[str]] = None,
        condition: str = "",
        step: str = "",
    ) -> str:
        """Write a numpy matrix as CSV."""
        return self.save_dataframe(
            pd.DataFrame(matrix, index=index_names, columns=col_names),
            filename, condition, step,
        )

    def save_edge_list(
        self,
        matrix: np.ndarray,
        node_names: List[str],
        filename: str,
        condition: str = "",
        step: str = "",
    ) -> str:
        """Convert a directed adjacency matrix to a from_to edge list.

        ``matrix[i, j]`` is the effect of j on i, so j is the source.
        The output is compatible with Cytoscape / Gephi.
        """
        edges = []
        n = matrix.shape[0]
        for i in range(n):
            for j in range(n):
                weight = matrix[i, j]
                if i != j and not np.isnan(weight) and weight != 0:
                    edges.append({
                        "Source": node_names[j],
                        "Target": node_names[i],
                        "Weight": weight,
                        "AbsWeight": abs(weight),
                        "Sign": "Positive" if weight > 0 else "Negative",
                    })
        df = pd.DataFrame(edges, columns=["Source", "Target", "Weight", "AbsWeight", "Sign"])
        return self.save_dataframe(df, filename, condition, step, index=False)

    def save_json(self, obj, filename: str, condition: str = "", step: str = "") -> str:
        """Write a JSON-serialisable object (e.g. GLMY homology intervals)."""
        import json

        path = self.get_path(filename, condition, step)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=2, default=str)
        return path

    # -- figures -------------------------------------------------------------
    def save_figure(
        self,
        fig,
        filename: str,
        condition: str = "",
        step: str = "",
        dpi: Optional[int] = None,
    ) -> str:
        """Save a matplotlib figure and close it.

        ``axes.unicode_minus`` is reset afterwards: some plotting functions
        update global rcParams, which can restore the Unicode minus sign
        (U+2212) that the bundled CJK font lacks.
        """
        import matplotlib.pyplot as plt

        path = self.get_path(filename, condition, step)
        fig.savefig(path, dpi=self.dpi if dpi is None else dpi, bbox_inches="tight")
        plt.close(fig)
        self._restore_rcparams()
        return path

    @staticmethod
    def _restore_rcparams() -> None:
        """Reset ``axes.unicode_minus`` (the font family is set by setup_matplotlib)."""
        import matplotlib.pyplot as plt

        plt.rcParams["axes.unicode_minus"] = False
