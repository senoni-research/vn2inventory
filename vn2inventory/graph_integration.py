"""
Graph Integration Module

Loads and merges graph model features (KumoRFM) into HB pipeline.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional


def load_graph_features(
    graph_features_path: Path,
    col_mapping: Dict[str, str],
    verbose: bool = True
) -> Optional[pd.DataFrame]:
    """
    Load graph model features from CSV.
    
    Returns:
        DataFrame with graph features or None if file not found
    """
    if not graph_features_path.exists():
        if verbose:
            print(f"⚠️ Graph features not found: {graph_features_path}")
            print("   Proceeding with BASELINE HB only")
        return None
    
    graph_feat = pd.read_csv(graph_features_path)
    graph_feat = graph_feat.rename(columns={
        "store_id": col_mapping["store"], 
        "product_id": col_mapping["product"]
    })
    
    # Compute derived features
    # p_mean: average activation probability across the horizon if available
    if all(c in graph_feat.columns for c in ["p_t1", "p_t2", "p_t3"]):
        graph_feat["p_mean"] = graph_feat[["p_t1", "p_t2", "p_t3"]].mean(axis=1)
    elif "p_t3" in graph_feat.columns:
        graph_feat["p_mean"] = graph_feat["p_t3"].astype(float)
    else:
        graph_feat["p_mean"] = 0.5
    
    # Prefer weekly moments if available; fallback to horizon moments
    has_weekly_mu = all(c in graph_feat.columns for c in ["mu_w1", "mu_w2", "mu_w3"])
    has_weekly_sigma = all(c in graph_feat.columns for c in ["sigma_w1", "sigma_w2", "sigma_w3"])
    
    if has_weekly_mu:
        graph_feat["mu_weekly_graph"] = graph_feat[["mu_w1", "mu_w2", "mu_w3"]].astype(float).mean(axis=1)
    elif "mu_H" in graph_feat.columns:
        graph_feat["mu_weekly_graph"] = graph_feat["mu_H"].astype(float) / 3.0
    else:
        graph_feat["mu_weekly_graph"] = np.nan
    
    if has_weekly_sigma:
        # Use average weekly sigma (pragmatic; avoids i.i.d. assumption)
        graph_feat["sigma_weekly_graph"] = graph_feat[["sigma_w1", "sigma_w2", "sigma_w3"]].astype(float).mean(axis=1)
    elif "sigma_H" in graph_feat.columns:
        graph_feat["sigma_weekly_graph"] = graph_feat["sigma_H"].astype(float) / np.sqrt(3.0)
    else:
        graph_feat["sigma_weekly_graph"] = np.nan
    
    if verbose:
        print(f"✅ Graph features loaded: {len(graph_feat)} SKUs from {graph_features_path.name}")
    
    return graph_feat


def merge_graph_features(
    sl: pd.DataFrame,
    graph_feat: pd.DataFrame,
    col_mapping: Dict[str, str]
) -> pd.DataFrame:
    """
    Merge graph features into design matrix.
    
    Fills missing values with neutral defaults.
    """
    # Base and optional columns to merge
    base_cols = [col_mapping["store"], col_mapping["product"], "p_t3", "p_mean", "mu_weekly_graph", "sigma_weekly_graph"]
    optional_cols = [c for c in [
        "mu_H", "sigma_H", "mu_hat_plus", "sigma_hat_plus",
        "p_t1", "p_t2",
        "mu_w1", "mu_w2", "mu_w3", "sigma_w1", "sigma_w2", "sigma_w3"
    ] if c in graph_feat.columns]
    graph_cols = base_cols + optional_cols
    
    sl_before = len(sl)
    sl = sl.merge(graph_feat[graph_cols], on=[col_mapping["store"], col_mapping["product"]], how="left")
    
    # Fill NaNs with neutral values
    if "p_t3" in sl.columns:
        sl["p_t3"] = sl["p_t3"].fillna(0.5)
    if "p_mean" in sl.columns:
        sl["p_mean"] = sl["p_mean"].fillna(0.5)
    
    sku_grp_local = [col_mapping["store"], col_mapping["product"]]
    sl["mu_weekly_graph"] = sl["mu_weekly_graph"].fillna(sl.groupby(sku_grp_local)["y"].transform("mean"))
    sl["sigma_weekly_graph"] = sl["sigma_weekly_graph"].fillna(sl.groupby(sku_grp_local)["y"].transform("std"))
    
    print("="*70)
    print("📊 GRAPH FEATURES MERGED INTO DESIGN MATRIX")
    print("="*70)
    print(f"   Rows before merge: {sl_before}")
    print(f"   Rows after merge: {len(sl)}")
    print(f"\n📈 Feature Ranges:")
    print(f"   p_t3 (activation prob): [{sl['p_t3'].min():.3f}, {sl['p_t3'].max():.3f}]")
    print(f"   mu_weekly_graph: [{sl['mu_weekly_graph'].min():.2f}, {sl['mu_weekly_graph'].max():.2f}]")
    print(f"\n🔗 Correlations with observed demand (y):")
    if "p_t3" in sl.columns:
        print(f"   corr(y, p_t3): {sl[['y','p_t3']].corr().iloc[0,1]:.3f}")
    print(f"   corr(y, mu_weekly_graph): {sl[['y','mu_weekly_graph']].corr().iloc[0,1]:.3f}")
    print("="*70)
    
    return sl


def build_graph_enriched_formulas(
    hier_terms: list,
    include_graph: bool
) -> list:
    """
    Build formula candidates with optional graph features.
    
    Args:
        hier_terms: List of hierarchy terms like ["C(Department)", "C(ProductGroup)"]
        include_graph: Whether to include graph features
        
    Returns:
        List of formula strings in order of richness
    """
    if include_graph:
        graph_terms = ["mu_weekly_graph", "p_mean"]
        richest = "y ~ sin52 + cos52 + " + " + ".join(hier_terms + graph_terms) if hier_terms else "y ~ sin52 + cos52 + mu_weekly_graph + p_mean"
        
        formula_candidates = [
            richest,  # Full: hierarchy + seasonality + graph
            "y ~ sin52 + cos52 + mu_weekly_graph + p_mean",  # Graph-only
            "y ~ sin52 + cos52 + C(ProductGroup) + mu_weekly_graph",  # Graph + ProductGroup
            "y ~ sin52 + cos52 + C(Department) + C(ProductGroup)",  # Baseline fallback
        ]
    else:
        richest = "y ~ sin52 + cos52 + " + " + ".join(hier_terms) if hier_terms else "y ~ sin52 + cos52"
        formula_candidates = [
            richest,
            "y ~ sin52 + cos52 + C(ProductGroup)",
            "y ~ sin52 + cos52 + C(Department)",
            "y ~ sin52 + cos52",
        ]
    
    return formula_candidates


def enhance_params_with_graph(
    params: pd.DataFrame,
    sl: pd.DataFrame,
    sku_grp: list
) -> pd.DataFrame:
    """
    Enhance CV parameters with graph activation probabilities.
    
    Replaces empirical p0 with graph p_t3.
    """
    if 'p_t3' not in sl.columns:
        return params
    
    graph_agg = sl.groupby(sku_grp).agg({
        "p_t3": "mean",
        "p_mean": "mean",
        "mu_weekly_graph": "mean"
    })
    
    params_enhanced = params.join(graph_agg, how="left")
    params_enhanced["p_t3"] = params_enhanced["p_t3"].fillna(0.5)
    params_enhanced["p_mean"] = params_enhanced["p_mean"].fillna(0.5)
    
    # Replace p0 with graph's learned zero-probability
    params_enhanced["p0_graph"] = 1.0 - params_enhanced["p_t3"]
    
    # Update intermittency flag using graph
    params_enhanced["intermittent_graph"] = (params_enhanced["p_t3"] < 0.5) & (params_enhanced["mu"] < 2.0)
    
    # Use graph-enhanced routing
    params_enhanced["p0"] = params_enhanced["p0_graph"]
    params_enhanced["intermittent"] = params_enhanced["intermittent_graph"]
    
    return params_enhanced

