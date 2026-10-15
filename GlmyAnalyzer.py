# -*- coding: utf-8 -*-
"""GLMY persistent path homology.

Graph construction (from/to/weight normalisation, weight offset), homology
computation (including positive/negative weight splitting), from_to.csv
loading and member enumeration, plus barcode and Betti-number visualisation.

The homology kernel is a pure-Python persistent path homology implementation.
Class names, function names, parameter defaults and all computations are
unchanged from the original implementation.
"""
from __future__ import annotations

from typing import Any

import pandas as pd


DIMENSION: int = 5

DIMENSION_KEYS: tuple[str, ...] = (
    "3",
    "2",
    "1",
    "0",
)
DIMENSION_COLORS: tuple[str, ...] = (
    "purple",
    "g",
    "r",
    "b",
)
DIMENSION_LABELS: tuple[str, ...] = (
    r"$\beta_3$",
    r"$\beta_2$",
    r"$\beta_1$",
    r"$\beta_0$",
)

Bar = list
Homology = dict

# Backwards-compatible aliases.
DEFAULT_DIMENSION: int = DIMENSION
DEFAULT_WEIGHT_OFFSET: float | None = None

# Column-name aliases, all mapped to source / target / weight.
_COLUMN_ALIASES = {
    "from": "source",
    "source": "source",
    "to": "target",
    "target": "target",
    "weight": "weight",
    "effect": "weight",
    "w": "weight",
}


# -- persistent path homology kernel for directed graphs --------------------
class Digraph():
    False_value = 99999999
    @classmethod
    def get_face(cls,path,index):
        res = []
        for i in range(len(path)):
            if i!=index:
                res.append(path[i])
        return res
    @classmethod
    def get_max(cls,m):
        if len(m)>0:
            return max(m)
        else:
            return -1
    @classmethod
    def add_two_column(cls,a,b):
        for item in a:
            if item in b:
                b.remove(item)
            else:
                b.add(item)
        return b
    @classmethod
    def combine(cls,a,b):
        res = []
        for i in range(len(a)-1):
            res.append(a[i])
        for i in range(len(b)):
            res.append(b[i])
        return res


    def __init__(self,V,WG,dim):
        self.V = V
        self.dim = dim
        self.P = []
        self.PW = []
        self.set_PW(WG)
        self.boundary_matrix = []
        self.set_boundary_matrix()
        self.diagram = {}

    def set_PW(self,WG):
        v_out = {} # arrows from the vertex
        for i in range(len(self.V)):
            v_out[self.V[i]] = []
        for item in WG:
            one = item[0]
            two = item[1]
            weight = item[2]
            v_out[one].append(item)
        # all the paths and associated weights
        P = []
        W = []
        for i in range(self.dim):
            P.append([])
            W.append([])

        # P0 is all the vertices
        for i in self.V:
            P[0].append([i])
            W[0].append(0)
        # P1 is all the arrows
        for item in WG:
            one = item[0]
            two = item[1]
            weight = item[2]
            P[1].append([one,two])
            W[1].append(weight)

        # Pn(n>1)
        for n in range(2,self.dim):
            for i in range(len(P[n-1])):
                item = P[n-1][i]
                for temp1 in v_out[item[-1]]:
                    new = Digraph.combine(item,temp1[0:2])
                    P[n].append(new)
                    W[n].append( max(temp1[2],W[n-1][i]) )


        # dictionary is used to fast search
        dict1 = dict([])
        for i in range(self.dim):
            for j in range(len(P[i])):
                self.P.append(P[i][j])
                self.PW.append(W[i][j])
                dict1[str(P[i][j])] = 1

        # get the associated complex of the path complex of digraph.
        for n in range(2,self.dim):
            for i in range(len(P[n])):
                temp1 = P[n][i]
                for no in range(len(temp1)):
                    face = Digraph.get_face(temp1,no)
                    if str(face) not in dict1:
                        self.P.append(face)
                        self.PW.append(Digraph.False_value)
                        dict1[str(face)] = 1


    def set_boundary_matrix(self):
        complex_path = []
        for i in range(len(self.P)):
            temp = [ self.P[i],self.PW[i] ]
            complex_path.append(temp)
        complex_path = sorted(complex_path,key=lambda x:(x[1]+len(x[0])*0.00001))
        graph = []
        complex1 = []
        graph_weight = []
        complex_weight = []
        for i in range(len(complex_path)):
            item = complex_path[i]
            complex1.append(item[0])
            complex_weight.append(item[1])
            if item[1]!=Digraph.False_value:
                graph.append(item[0])
                graph_weight.append(item[1])
        N = len(graph)
        self.P = complex1
        self.PW = complex_weight
        # dict1 is used for fast search
        dict1 = dict([])
        for i in range(len(complex1)):
            item = complex1[i]
            dict1[str(item)] = i

        for j in range(N):
            temp = set()
            temp1 = graph[j]
            if len(temp1)>1:
                for i in range(len(temp1)):
                    face = Digraph.get_face(temp1, i)
                    index = dict1[str(face)]
                    temp.add(index)
            self.boundary_matrix.append(temp)


    def get_persistence(self):
        inf_bar = dict([])
        finite_bar = []
        L = []
        for i in range(len(self.P)):
            L.append(-1)
        for j in range(len(self.boundary_matrix)):
            N = Digraph.get_max(self.boundary_matrix[j])
            while N!=-1 and L[N]!=-1:
                self.boundary_matrix[j] = Digraph.add_two_column(self.boundary_matrix[L[N]], self.boundary_matrix[j])
                N = Digraph.get_max(self.boundary_matrix[j])
            if N==-1:
                inf_bar[ str([self.P[j],self.PW[j]]) ] = 1
            elif L[N]==-1:
                L[N] = j

        # find all finite bars
        for j in range(len(self.boundary_matrix)):
            low_j = Digraph.get_max(self.boundary_matrix[j])
            if low_j!=-1 and (str( [self.P[low_j],self.PW[low_j]] ) in inf_bar):
                del inf_bar[ str( [self.P[low_j],self.PW[low_j]] ) ]
                if low_j<j:
                    finite_bar.append( [len(self.P[low_j])-1,self.PW[low_j],self.PW[j]] )

        for i in range(self.dim):
            self.diagram[str(i)] = []
        for item in inf_bar.items():
            temp = eval(item[0])
            self.diagram[str(len(temp[0])-1)].append([temp[1],-1])
        for item in finite_bar:
            temp = [item[1],item[2]]
            if item[2]>item[1]:
                self.diagram[str(item[0])].append(temp)


    '''

V = [ 1,2,3 ]
weighted_digraph = [ [ 1,2,1 ], [ 2,3,2 ], [ 3,1,3 ] ]
dim = 3
dg = Digraph(V,weighted_digraph,dim)
dg.get_persistence()
path_homology = dg.diagram

'''

def _vertex_sort_key(name: str) -> tuple[int, float | str]:
    """Numeric vertex names sort before string names."""
    try:
        return (0, float(name))
    except ValueError:
        return (1, str(name))


def normalize_network(network: pd.DataFrame) -> pd.DataFrame:
    """Normalise an edge table to the columns ``source`` / ``target`` / ``weight``.

    Accepts ``source`` / ``target`` (upstream scripts) as well as ``from`` /
    ``to`` (exported ``from_to.csv``); the weight column may be ``weight`` /
    ``Effect``. Non-numeric weights are dropped and self-loops are removed.
    """
    frame = pd.DataFrame(network).copy()
    rename = {}
    for column in frame.columns:
        key = str(column).strip().lower()
        if key in _COLUMN_ALIASES:
            rename[column] = _COLUMN_ALIASES[key]
    frame = frame.rename(columns=rename)

    required = {"source", "target", "weight"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"edge table is missing required columns: {sorted(missing)}; "
            f"expected source/target/weight or from/to/weight, "
            f"got {list(frame.columns)}"
        )

    frame = frame[["source", "target", "weight"]].copy()
    frame["source"] = frame["source"].astype(str).str.strip()
    frame["target"] = frame["target"].astype(str).str.strip()
    frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce")
    frame = frame.dropna(subset=["weight"])

    if frame.empty:
        raise ValueError("no valid numeric weights in the edge table; cannot run GLMY.")

    self_loop_mask = frame["source"] == frame["target"]
    if self_loop_mask.any():
        frame = frame.loc[~self_loop_mask]
    if frame.empty:
        raise ValueError("no usable edges after removing self-loops; cannot run GLMY.")

    return frame.reset_index(drop=True)


def build_weighted_digraph(
    network: pd.DataFrame,
    weight_offset: float | None = None,
) -> tuple[list[int], list[list[float | int]], float]:
    """Build the ``Digraph`` input: integer vertices ``1..n`` and shifted weights.

    Parameters
    ----------
    network:
        Edge table with ``source`` / ``target`` / ``weight``.
    weight_offset:
        Value added to every weight. When ``None`` it is computed as
        ``max(0.0, 1.0 - min(weight))`` so all shifted weights are strictly
        positive. Pass ``0.0`` explicitly for weights that are already
        non-negative (e.g. after splitting by sign).

    Returns
    -------
    vertices, weighted_edges, weight_offset
    """
    clean = normalize_network(network)

    node_names = sorted(
        set(clean["source"]).union(set(clean["target"])),
        key=_vertex_sort_key,
    )
    node_map = {
        name: index + 1
        for index, name in enumerate(node_names)
    }

    if weight_offset is None:
        minimum_weight = float(clean["weight"].min())
        weight_offset = max(0.0, 1.0 - minimum_weight)

    weighted_edges: list[list[float | int]] = []
    for row in clean.itertuples(index=False):
        weighted_edges.append(
            [
                node_map[row.source],
                node_map[row.target],
                float(row.weight) + float(weight_offset),
            ]
        )

    vertices = list(range(1, len(node_names) + 1))
    return vertices, weighted_edges, float(weight_offset)


def vertex_id_map(network: pd.DataFrame) -> dict[str, int]:
    """Map node names to integer vertex ids (same numbering as above)."""
    clean = normalize_network(network)
    node_names = sorted(
        set(clean["source"]).union(set(clean["target"])),
        key=_vertex_sort_key,
    )
    return {name: index + 1 for index, name in enumerate(node_names)}


def _restore_bar_endpoint(
    value: Any,
    *,
    weight_offset: float,
) -> float | None:
    """Map a barcode endpoint back to the user weight scale.

    Both ``-1`` and ``None`` are infinite sentinels and normalise to ``None``;
    other values have ``weight_offset`` subtracted.
    """
    if value is None or value == -1:
        return None
    return float(value) - float(weight_offset)


def compute_glmy_homology(
    network: pd.DataFrame,
    weight_offset: float | None = None,
    *,
    dimension: int = DIMENSION,
) -> Homology:
    """Compute path homology in dimensions 0-3 as ``{dim: [[birth, death], ...]}``.

    Infinite intervals are ``None``. ``dimension`` is passed to ``Digraph``
    (default 5, i.e. one extra dimension is computed so that dimension 3 is
    correct); the returned dict always has ``len(DIMENSION_KEYS)`` entries.
    """
    vertices, weighted_edges, weight_offset = build_weighted_digraph(
        network, weight_offset,
    )

    digraph = Digraph(vertices, weighted_edges, dimension)
    digraph.get_persistence()

    homology: Homology = {}
    for index, key in enumerate(DIMENSION_KEYS):
        dimension_number = int(key)
        bars: list[Bar] = []

        for birth, death in digraph.diagram.get(key, []):
            if dimension_number == 0 and birth == 0:
                restored_birth = None
            else:
                restored_birth = _restore_bar_endpoint(
                    birth, weight_offset=weight_offset,
                )

            bars.append(
                [
                    restored_birth,
                    _restore_bar_endpoint(death, weight_offset=weight_offset),
                ]
            )

        homology[key] = bars

    return homology


def _split_network_by_sign(
    network: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into positive and negative weight subgraphs.

    Zero-weight edges are dropped (they only produce degenerate bars); the
    negative group uses absolute values. Both returned tables are already
    normalised.
    """
    clean = normalize_network(network)

    positive = clean.loc[clean["weight"] > 0].reset_index(drop=True)
    negative = clean.loc[clean["weight"] < 0].reset_index(drop=True)
    negative = negative.copy()
    negative["weight"] = negative["weight"].abs()

    return positive, negative


def compute_glmy_homology_split(
    network: pd.DataFrame,
    *,
    dimension: int = DIMENSION,
) -> dict[str, Homology]:
    """Compute homology separately for the positive and negative subgraphs.

    Returns
    -------
    dict[str, Homology]
        Keys ``"positive"`` / ``"negative"``, each usable with
        :func:`plot_glmy_barcode_split`. A side without usable edges yields an
        empty homology rather than raising.
    """
    positive, negative = _split_network_by_sign(network)

    homologies: dict[str, Homology] = {}
    for name, subgraph in (("positive", positive), ("negative", negative)):
        if subgraph.empty:
            homologies[name] = {key: [] for key in DIMENSION_KEYS}
        else:
            homologies[name] = compute_glmy_homology(
                subgraph,
                weight_offset=0.0,
                dimension=dimension,
            )
    return homologies


def bar_counts(homology: Homology) -> dict[str, int]:
    """Number of bars per dimension, used as the shared y-axis height."""
    return {key: len(homology.get(key, [])) for key in DIMENSION_KEYS}


def bar_counts_split(
    homologies: dict[str, Homology],
) -> dict[str, dict[str, int]]:
    """Bar counts for both signs, used by :func:`plot_glmy_barcode_split`."""
    return {
        sign: bar_counts(homologies.get(sign, {}))
        for sign in ("positive", "negative")
    }


"""Network analysis backend helpers.

The GLMY barcode workflow consumes ``from_to.csv`` files exported by the
network reconstruction step and computes persistent path homology with the
bundled pure-Python implementation.
"""

import io
import re
import zipfile


# -- ZIP parsing ------------------------------------------------------------

def _read_zip_member_csv(zf: zipfile.ZipFile, name: str) -> pd.DataFrame:
    """Read a CSV member from a ZIP archive (utf-8-sig BOM tolerated)."""
    with zf.open(name) as fh:
        return pd.read_csv(io.BytesIO(fh.read()))


def list_from_to_members(zip_bytes: bytes) -> list[str]:
    """Enumerate all ``from_to.csv`` paths inside a ZIP archive.

    Single-layer exports return ``["from_to.csv"]``; two-layer exports return
    paths such as ``"inter_cluster/<cond>/from_to.csv"`` and
    ``"intra_cluster/<cond>/<cluster>/from_to.csv"``.
    """
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = [n for n in zf.namelist() if n.endswith("from_to.csv")]
    names.sort()
    return names


def member_display_label(member_path: str) -> str:
    """Turn a from_to.csv path inside a ZIP into a short display label."""
    if member_path == "from_to.csv":
        return "single_layer"
    if member_path.endswith("/from_to.csv"):
        return member_path[: -len("/from_to.csv")]
    return member_path


def load_from_to_from_zip(zip_bytes: bytes, member_path: str) -> pd.DataFrame:
    """Read a specific ``from_to.csv`` from a ZIP archive and validate columns."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        df = _read_zip_member_csv(zf, member_path)
    required = {"from", "to", "weight"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{member_path} is missing required columns: {sorted(missing)}; "
            f"got {list(df.columns)}"
        )
    df = df.copy()
    df["from"] = df["from"].astype(str)
    df["to"] = df["to"].astype(str)
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    df = df.dropna(subset=["weight"]).reset_index(drop=True)
    return df


# -- GLMY computation -------------------------------------------------------

def run_glmy(
    from_to_df: pd.DataFrame,
    *,
    dim: int = DEFAULT_DIMENSION,
    weight_offset: float | None = DEFAULT_WEIGHT_OFFSET,
) -> dict[str, Any]:
    """Compute GLMY path homology with the bundled Python implementation.

    Parameters
    ----------
    from_to_df:
        Edge table, accepting ``from`` / ``to`` / ``weight`` or
        ``source`` / ``target`` / ``weight``.
    dim:
        Dimension passed to ``Digraph`` (default 5, one extra dimension so that
        dimension 3 is correct); the returned dict always covers beta_3..beta_0.
    weight_offset:
        Value added to every weight, used only for numerical separation of the
        filtration -- this is not a normalisation. When ``None`` (default) it
        becomes ``max(0, 1 - min(weight))``.

    Returns
    -------
    dict
        ``homology`` is ``{dim: [[birth, death], ...]}`` with ``None`` for
        infinite intervals; ``weight_offset`` is the offset actually used.
    """
    clean = normalize_network(from_to_df)

    # Resolve the offset first (None -> automatic) so the caller sees the real value.
    _, _, resolved_offset = build_weighted_digraph(clean, weight_offset)

    homology = compute_glmy_homology(
        clean,
        resolved_offset,
        dimension=dim,
    )
    return {
        "homology": homology,
        "vertex_id_map": vertex_id_map(clean),
        "dimension": dim,
        "weight_offset": resolved_offset,
        "backend": "python",
    }


def run_glmy_split(
    from_to_df: pd.DataFrame,
    *,
    dim: int = DEFAULT_DIMENSION,
) -> dict[str, Any]:
    """Compute homology separately for the positive and negative subgraphs.

    Returns
    -------
    dict
        ``homologies`` is ``{"positive": Homology, "negative": Homology}``;
        ``edge_counts`` reports how many edges each side used.
    """
    clean = normalize_network(from_to_df)
    homologies = compute_glmy_homology_split(clean, dimension=dim)

    positive_count = int((clean["weight"] > 0).sum())
    negative_count = int((clean["weight"] < 0).sum())

    return {
        "homologies": homologies,
        "edge_counts": {
            "positive": positive_count,
            "negative": negative_count,
        },
        "dimension": dim,
        "weight_offset": 0.0,
        "backend": "python",
    }


# -- adaptive max_x ---------------------------------------------------------

def suggest_max_x(
    from_to_df: pd.DataFrame,
    *,
    buffer_ratio: float = 0.1,
    floor: float = 1e-9,
) -> float:
    """Suggest an adaptive right-hand limit for the barcode x-axis.

    Returns ``|weight|.max() * (1 + buffer_ratio)``. The barcode values
    themselves are unchanged; this only sets a plotting ``xlim`` and never
    rewrites ``weight`` in ``from_to.csv``. ``floor`` is only a fallback for
    empty or all-zero tables.
    """
    if from_to_df.empty:
        return floor
    abs_max = float(from_to_df["weight"].abs().max())
    suggested = abs_max * (1.0 + max(0.0, buffer_ratio))
    return max(floor, suggested)


# -- file-name helpers ------------------------------------------------------

def sanitize_name(name: str) -> str:
    """Reduce an arbitrary string to a file-name-safe fragment."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
    return cleaned or "glmy"


from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Legacy aliases kept for backwards compatibility.
_DIM_KEYS = DIMENSION_KEYS
_DIM_COLORS = DIMENSION_COLORS
_DIM_LABELS = DIMENSION_LABELS


def _sorted_bars(
    homology: dict[str, list[list[Any]]],
    key: str,
) -> list[list[float | None]]:
    """Fetch the bars of one dimension, normalise infinite sentinels and sort.

    Both ``-1`` and ``None`` become ``None``. Bars are sorted by birth
    ascending (infinite lowest) and then death descending.
    """
    bars: list[list[float | None]] = []
    for bar in homology.get(key, []):
        if not bar:
            continue
        birth, death = bar[0], bar[1]
        bars.append(
            [
                None if birth is None or birth == -1 else float(birth),
                None if death is None or death == -1 else float(death),
            ]
        )

    bars.sort(
        key=lambda bar: (
            float("-inf") if bar[0] is None else bar[0],
            float("-inf") if bar[1] is None else -bar[1],
        )
    )
    return bars


def _finite_endpoints(bars: list[list[float | None]]) -> list[float]:
    return [
        value
        for birth, death in bars
        for value in (birth, death)
        if value is not None
    ]


def _draw_bars(
    ax: plt.Axes,
    bars: list[list[float | None]],
    *,
    color: str,
    start_for_infinite: float,
    infinite_x: float,
) -> None:
    """Draw one dimension of bars; infinite bars end with a black ``>`` arrow."""
    for index, (birth, death) in enumerate(bars):
        start_x = start_for_infinite if birth is None else birth
        if death is None:
            ax.plot(
                [start_x, infinite_x],
                [index, index],
                color=color,
                linewidth=3,
                alpha=0.8,
            )
            ax.plot(
                infinite_x,
                index,
                marker=">",
                markersize=12,
                color="black",
                markeredgecolor="black",
            )
        else:
            ax.plot(
                [start_x, death],
                [index, index],
                color=color,
                linewidth=2.5,
            )


def plot_glmy_barcode(
    homology: dict[str, list[list[Any]]],
    shared_bar_counts: dict[str, int] | None = None,
    *,
    max_x: float | None = None,
) -> plt.Figure:
    """Draw a GLMY barcode (four stacked panels beta_3..beta_0, x centred on 0).

    Parameters
    ----------
    homology: ``{dim: [[birth, death], ...]}``; ``None`` or ``-1`` is infinite.
    shared_bar_counts: y-axis height per dimension; inferred when ``None``.
    max_x: optional symmetric half-span; when given the x-axis becomes
        ``[-max_x*1.1, max_x*1.1]``, otherwise it is derived from the data.

    Returns
    -------
    matplotlib.figure.Figure
        The caller is responsible for ``plt.close(fig)``.
    """
    if shared_bar_counts is None:
        shared_bar_counts = bar_counts(homology)

    bars_per_dimension = [
        _sorted_bars(homology, key) for key in DIMENSION_KEYS
    ]

    if max_x is not None and float(max_x) > 0:
        half_span = float(max_x)
    else:
        half_span = max(
            (abs(value) for bars in bars_per_dimension
             for value in _finite_endpoints(bars)),
            default=0.0,
        )
        if half_span == 0.0:
            half_span = 1.0

    left_x = -half_span * 1.10
    right_x = half_span * 1.10
    infinite_x = half_span * 1.07

    fig, axes = plt.subplots(
        len(DIMENSION_KEYS),
        1,
        figsize=(7, 10),
        sharex=True,
    )
    if len(DIMENSION_KEYS) == 1:
        axes = [axes]

    for key, bars, ax, color, label in zip(
        DIMENSION_KEYS,
        bars_per_dimension,
        axes,
        DIMENSION_COLORS,
        DIMENSION_LABELS,
    ):
        maximum_y = max(shared_bar_counts.get(key, len(bars)), 5)
        y_margin = max(1.0, maximum_y * 0.05)
        ax.set_ylim(-y_margin, maximum_y - 1 + y_margin)
        ax.set_ylabel(label, fontsize=24, labelpad=20)
        ax.yaxis.set_major_locator(
            ticker.MaxNLocator(integer=True, nbins=3)
        )
        ax.tick_params(axis="both", which="major", labelsize=20)

        _draw_bars(
            ax,
            bars,
            color=color,
            start_for_infinite=left_x,
            infinite_x=infinite_x,
        )

    axes[-1].set_xlim(left_x, right_x)
    fig.tight_layout()
    return fig


def plot_glmy_barcode_split(
    homologies: dict[str, dict[str, list[list[Any]]]],
    shared_bar_counts: dict[str, dict[str, int]] | None = None,
) -> plt.Figure:
    """Draw the positive and negative barcodes side by side (4 rows x 2 columns).

    Parameters
    ----------
    homologies: keys ``"positive"`` / ``"negative"``.
    shared_bar_counts: ``{sign: {dim: count}}``; inferred when ``None``.
    """
    if shared_bar_counts is None:
        shared_bar_counts = bar_counts_split(homologies)

    figure, axes = plt.subplots(
        len(DIMENSION_KEYS),
        2,
        figsize=(7, 10),
        sharex=False,
        constrained_layout=True,
    )
    if len(DIMENSION_KEYS) == 1:
        axes = [axes]

    for row, key in enumerate(DIMENSION_KEYS):
        for column, sign in enumerate(("positive", "negative")):
            ax = axes[row][column]
            color = DIMENSION_COLORS[row]

            bars = _sorted_bars(homologies.get(sign, {}), key)

            finite_endpoints = _finite_endpoints(bars)
            max_x = max(finite_endpoints, default=1.0)
            if max_x == 0.0:
                max_x = 1.0

            counts = shared_bar_counts.get(sign, {}).get(key, len(bars))
            maximum_y = max(counts, 5)
            y_margin = max(1.0, maximum_y * 0.05)
            ax.set_ylim(-y_margin, maximum_y - 1 + y_margin)
            ax.yaxis.set_major_locator(
                ticker.MaxNLocator(integer=True, nbins=3)
            )
            ax.tick_params(axis="both", which="major", labelsize=20)

            if column == 0:
                ax.set_ylabel(
                    DIMENSION_LABELS[row],
                    fontsize=24,
                    labelpad=20,
                )
            if row == 0:
                ax.set_title(sign, fontsize=28)

            # Both sides use the same convention: 0 at the left, |w| max at the right.
            ax.set_xlim(0.0, max_x * 1.10)

            _draw_bars(
                ax,
                bars,
                color=color,
                start_for_infinite=0.0,
                infinite_x=max_x * 1.07,
            )

    return figure


def plot_glmy_barcode_combined(
    homology: dict[str, list[list[Any]]],
    split_homologies: dict[str, dict[str, list[list[Any]]]],
    *,
    max_x: float | None = None,
    figsize: tuple[float, float] = (13.0, 10.0),
) -> plt.Figure:
    """Draw overall / positive / negative barcodes in one figure.

    Layout: 4 rows (beta_3..beta_0) x 3 columns (``overall`` | ``positive`` |
    ``negative``). The three panels of a row share the same y-axis height, so
    the columns are directly comparable.

    Parameters
    ----------
    homology: overall (signed) homology.
    split_homologies: ``{"positive": Homology, "negative": Homology}``.
    max_x: symmetric half-span of the ``overall`` column; inferred when ``None``.
    figsize: figure size in inches.
    """
    overall_bars = {
        key: _sorted_bars(homology, key) for key in DIMENSION_KEYS
    }
    split_bars = {
        sign: {
            key: _sorted_bars(split_homologies.get(sign, {}), key)
            for key in DIMENSION_KEYS
        }
        for sign in ("positive", "negative")
    }

    # The three panels of one dimension share the largest bar count.
    row_max_y = {
        key: max(
            len(overall_bars[key]),
            len(split_bars["positive"][key]),
            len(split_bars["negative"][key]),
            5,
        )
        for key in DIMENSION_KEYS
    }

    # Overall column: x-axis centred on 0.
    if max_x is not None and float(max_x) > 0:
        half_span = float(max_x)
    else:
        half_span = max(
            (
                abs(value)
                for bars in overall_bars.values()
                for value in _finite_endpoints(bars)
            ),
            default=0.0,
        )
        if half_span == 0.0:
            half_span = 1.0
    overall_left = -half_span * 1.10
    overall_right = half_span * 1.10
    overall_infinite = half_span * 1.07

    figure, axes = plt.subplots(
        len(DIMENSION_KEYS),
        3,
        figsize=figsize,
        sharex=False,
        constrained_layout=True,
    )
    if len(DIMENSION_KEYS) == 1:
        axes = [axes]

    for row, key in enumerate(DIMENSION_KEYS):
        maximum_y = row_max_y[key]
        y_margin = max(1.0, maximum_y * 0.05)

        for column in range(3):
            ax = axes[row][column]
            ax.set_ylim(-y_margin, maximum_y - 1 + y_margin)
            ax.yaxis.set_major_locator(
                ticker.MaxNLocator(integer=True, nbins=3)
            )
            ax.tick_params(axis="both", which="major", labelsize=20)
            if column == 0:
                ax.set_ylabel(
                    DIMENSION_LABELS[row],
                    fontsize=24,
                    labelpad=20,
                )

        axes[row][0].set_xlim(overall_left, overall_right)
        _draw_bars(
            axes[row][0],
            overall_bars[key],
            color=DIMENSION_COLORS[row],
            start_for_infinite=overall_left,
            infinite_x=overall_infinite,
        )

        for column, sign in ((1, "positive"), (2, "negative")):
            bars = split_bars[sign][key]
            column_max_x = max(_finite_endpoints(bars), default=1.0)
            if column_max_x == 0.0:
                column_max_x = 1.0
            axes[row][column].set_xlim(0.0, column_max_x * 1.10)
            _draw_bars(
                axes[row][column],
                bars,
                color=DIMENSION_COLORS[row],
                start_for_infinite=0.0,
                infinite_x=column_max_x * 1.07,
            )

    for column, name in enumerate(("overall", "positive", "negative")):
        axes[0][column].set_title(name, fontsize=28)

    return figure
