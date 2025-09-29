from __future__ import annotations

import pandas as pd
from typing import List, Tuple, Dict


def load_sales_history(
    sales_csv_path: str,
    store_col: str,
    product_col: str,
    sales_qty_col: str,
    date_col: str | None = None,
) -> pd.DataFrame:
    """
    Load historical weekly sales.

    Returns a DataFrame indexed by (store, product) with columns:
      - mean_demand
      - std_demand
      - observations
    """
    df = pd.read_csv(sales_csv_path)

    required_cols = {store_col, product_col, sales_qty_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in sales file: {missing}")

    # Robust numeric conversion
    df[sales_qty_col] = pd.to_numeric(df[sales_qty_col], errors="coerce").fillna(0.0)

    # Group by week if date provided, otherwise assume rows are already weekly
    group_cols = [store_col, product_col]
    if date_col and date_col in df.columns:
        group_cols.append(date_col)
    weekly = df.groupby(group_cols, dropna=False)[sales_qty_col].sum().reset_index()

    # Aggregate across time to get mean/std per (store, product)
    agg = (
        weekly.groupby([store_col, product_col], dropna=False)[sales_qty_col]
        .agg(["mean", "std", "count"])  # std is sample std (ddof=1)
        .rename(columns={"mean": "mean_demand", "std": "std_demand", "count": "observations"})
    )

    # If std is NA (e.g., only one observation), fall back to sqrt(mean)
    agg["std_demand"] = agg["std_demand"].fillna((agg["mean_demand"].clip(lower=0.0)).pow(0.5))
    return agg


def load_current_state(
    current_state_csv_path: str,
    store_col: str,
    product_col: str,
    on_hand_col: str,
    in_transit_cols: List[str] | None = None,
) -> pd.DataFrame:
    """
    Load current inventory state. Expected columns:
      - on_hand_col: on-hand units at end of week
      - in_transit_cols: list of columns containing units scheduled to arrive in the next k weeks
    Returns indexed DataFrame with columns: on_hand, on_order
    """
    df = pd.read_csv(current_state_csv_path)
    required_cols = {store_col, product_col, on_hand_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in current state file: {missing}")

    df[on_hand_col] = pd.to_numeric(df[on_hand_col], errors="coerce").fillna(0.0)
    on_order = 0.0
    if in_transit_cols:
        for c in in_transit_cols:
            if c not in df.columns:
                raise ValueError(f"Missing in_transit column '{c}' in current state file")
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        on_order = df[in_transit_cols].sum(axis=1)
    state = pd.DataFrame(
        {
            store_col: df[store_col],
            product_col: df[product_col],
            "on_hand": df[on_hand_col],
            "on_order": on_order,
        }
    )
    state = state.set_index([store_col, product_col])
    return state


def load_index(index_csv_path: str, store_col: str, product_col: str) -> pd.DataFrame:
    """Load index ordering (store-product pairs) and return a DataFrame with that index."""
    idx = pd.read_csv(index_csv_path)
    missing = {store_col, product_col} - set(idx.columns)
    if missing:
        raise ValueError(f"Missing columns in index file: {missing}")
    return idx.set_index([store_col, product_col])


def align_frames(
    index_df: pd.DataFrame,
    *frames: Tuple[pd.DataFrame, Dict[str, float] | None]
) -> List[pd.DataFrame]:
    """
    Reindex frames to the same (store, product) index as index_df. Fill missing numeric
    with zeros. Returns list of aligned frames.
    """
    aligned: List[pd.DataFrame] = []
    index = index_df.index
    for f in frames:
        if isinstance(f, tuple):
            df, fill = f
        else:
            df, fill = f, None
        a = df.reindex(index)
        if fill:
            a = a.fillna(fill)
        else:
            a = a.fillna(0.0)
        aligned.append(a)
    return aligned
