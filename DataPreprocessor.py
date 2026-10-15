# -*- coding: utf-8 -*-
"""Data loading and preprocessing (pipeline step 1: import data)."""

from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from QuasiDynamicAnalyzer import data_transformation, load_csv

__all__ = ["DataPreprocessor"]


class DataPreprocessor:
    """Data loading and column transformation."""

    @staticmethod
    def load_data(
        file_path,
        n_samples: Optional[int] = None,
        n_features: Optional[int] = None,
        scaler_type: str = "None",
        drop_features: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        """Load a CSV, optionally subset rows/columns, drop columns, then transform.

        Parameters
        ----------
        file_path:
            CSV path; the first column is used as the row index.
        n_samples:
            Number of leading rows to keep; ``None`` keeps all.
        n_features:
            Number of leading columns to keep; ``None`` keeps all.
        scaler_type:
            One of ``"None"``, ``"Log10_1p"``, ``"Minmax_0_1"``, ``"Z_min_add1"``.
        drop_features:
            Column names to drop (matched by intersection).
        """
        data = load_csv(file_path)

        if drop_features:
            cols_to_drop = data.columns.intersection(list(drop_features))
            if not cols_to_drop.empty:
                data = data.drop(columns=cols_to_drop)

        # Drop all-zero / constant (zero-variance) columns: they cannot be fitted
        # and would only add meaningless cluster members.
        numeric = data.apply(pd.to_numeric, errors="coerce")
        nunique = numeric.nunique(dropna=True)
        dead = list(nunique.index[nunique <= 1])
        if dead:
            data = data.drop(columns=dead)
            print(f"      dropped {len(dead)} all-zero/constant variables")

        if n_samples is not None:
            data = data.iloc[: int(n_samples), :]
        if n_features is not None:
            data = data.iloc[:, : int(n_features)]

        return data_transformation(
            data, "None" if scaler_type is None else str(scaler_type)
        )

    @classmethod
    def load_many(
        cls,
        data_files: Iterable[Tuple[str, str]],
        n_samples: Optional[int] = None,
        n_features: Optional[int] = None,
        scaler_type: str = "None",
        drop_features: Optional[Iterable[str]] = None,
    ) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
        """Load several conditions. Returns ``({condition: DataFrame}, names)``."""
        data_map: Dict[str, pd.DataFrame] = {}
        condition_names: List[str] = []
        for file_path, condition in data_files:
            data_map[condition] = cls.load_data(
                file_path=file_path,
                n_samples=n_samples,
                n_features=n_features,
                scaler_type=scaler_type,
                drop_features=drop_features,
            )
            condition_names.append(condition)
        return data_map, condition_names

    @staticmethod
    def read_raw_bytes(file_path) -> bytes:
        """Read the raw bytes of a file (for fitting APIs that accept bytes)."""
        with open(file_path, "rb") as fh:
            return fh.read()
