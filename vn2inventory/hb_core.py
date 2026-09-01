"""
Hierarchical Empirical Bayes Core Module

Implements the core HB pipeline: GLM fitting, EB shrinkage, tau² sweep,
per-segment dispersion estimation.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from typing import Dict, Tuple, Optional
from sklearn.metrics import mean_squared_error


# ======================================================================
# GLM FITTING WITH FALLBACKS
# ======================================================================

def fit_nb_glm_with_fallback(
    data: pd.DataFrame,
    formula_candidates: list,
    verbose: bool = True
) -> Tuple:
    """
    Fit NB GLM with automatic fallbacks for collinearity/convergence issues.
    
    Returns:
        (fitted_model, family, metadata)
    """
    last_err = None
    
    for f in formula_candidates:
        try:
            nb_fam = sm.families.NegativeBinomial(alpha=1.0)
            model = smf.glm(formula=f, data=data, family=nb_fam)
            
            # Try Poisson starts to stabilize IRLS
            try:
                pois = smf.glm(formula=f, data=data, family=sm.families.Poisson()).fit(maxiter=100)
                start = pois.params
            except Exception:
                start = None
            
            res = model.fit(maxiter=200, start_params=start)
            if getattr(res, "converged", True):
                if verbose:
                    print(f"✅ Fitted: {f} (NB-IRLS)")
                return res, nb_fam, {"formula": f, "method": "NB-IRLS"}
            raise RuntimeError("NB IRLS did not converge")
            
        except Exception as e:
            last_err = e
            # L2-regularized fallback
            try:
                nb_fam = sm.families.NegativeBinomial(alpha=1.0)
                model = smf.glm(formula=f, data=data, family=nb_fam)
                res = model.fit_regularized(alpha=1e-4, L1_wt=0.0, maxiter=300)
                if verbose:
                    print(f"✅ Fitted: {f} (NB-L2-regularized)")
                return res, nb_fam, {"formula": f, "method": "NB-L2-regularized"}
            except Exception as e2:
                last_err = e2
                continue
    
    # Final fallback: Poisson with L2
    f = "y ~ sin52 + cos52"
    model = smf.glm(formula=f, data=data, family=sm.families.Poisson())
    res = model.fit_regularized(alpha=1e-4, L1_wt=0.0, maxiter=300)
    if verbose:
        print(f"⚠️ Fallback: {f} (Poisson-L2-regularized)")
    return res, sm.families.Poisson(), {"formula": f, "method": "Poisson-L2-regularized"}


# ======================================================================
# EMPIRICAL BAYES SHRINKAGE
# ======================================================================

def apply_eb_shrinkage(
    sl: pd.DataFrame,
    res,
    nb_fam,
    sku_grp: list,
    tau2: Optional[float] = None
) -> pd.DataFrame:
    """
    Apply Empirical Bayes shrinkage to SKU-specific effects.

    Residuals and tau² live on the log-mean scale. The within-SKU term
    is the sampling variance of the mean log-residual (var / n_weeks),
    not the NB count variance. Mixing those units collapses every SKU
    onto the department mean.
    """
    eta_global = res.model.predict(res.params, linear=True)
    mu_global = np.exp(eta_global)
    sl["mu_global"] = mu_global

    alpha = getattr(nb_fam, "alpha", 1.0)

    eps = 1e-6
    sl["log_residual"] = np.log(sl["y"] + eps) - np.log(sl["mu_global"] + eps)
    sku_key = sl[sku_grp].apply(tuple, axis=1)
    g = sl.groupby(sku_key)["log_residual"]
    n_i = g.size().clip(lower=1)
    log_resid_mean_by_sku = g.mean()
    log_resid_var = g.var(ddof=1)
    log_resid_var = log_resid_var.fillna(g.var(ddof=0)).fillna(0.0)
    sigma2_mean = (log_resid_var / n_i).clip(lower=1e-8)

    if tau2 is None:
        between = float(log_resid_mean_by_sku.var(ddof=1)) if len(log_resid_mean_by_sku) > 1 else 1e-3
        tau2 = max(between - float(sigma2_mean.mean()), 1e-4)

    w = tau2 / (tau2 + sigma2_mean)
    shrunk_delta = (w * log_resid_mean_by_sku).rename("delta_sku")

    idx_tuples = sku_key
    sl = sl.join(shrunk_delta, on=idx_tuples)
    sl["delta_sku"] = sl["delta_sku"].fillna(0.0)
    sl["mu_eb"] = sl["mu_global"] * np.exp(sl["delta_sku"])

    mse_glob = mean_squared_error(sl["y"], sl["mu_global"])
    mse_eb = mean_squared_error(sl["y"], sl["mu_eb"])
    rmse_glob = float(np.sqrt(mse_glob))
    rmse_eb = float(np.sqrt(mse_eb))
    w_vals = w.to_numpy()

    print(f"✅ EB shrinkage applied (tau²={tau2:.6f}, log-scale)")
    print(f"   RMSE: global={rmse_glob:.2f}, EB={rmse_eb:.2f}")
    print(
        f"   SKU weight w: median={float(np.median(w_vals)):.3f} "
        f"p10={float(np.quantile(w_vals, 0.10)):.3f} "
        f"p90={float(np.quantile(w_vals, 0.90)):.3f}"
    )

    return sl, tau2, alpha, log_resid_mean_by_sku, sigma2_mean


# ======================================================================
# TAU² SWEEP WITH TEMPORAL VALIDATION
# ======================================================================

def tau2_sweep(
    sl: pd.DataFrame,
    sku_grp: list,
    log_resid_mean_by_sku: pd.Series,
    sigma2_mean_by_sku: pd.Series,
    tau2_grid: list,
    val_frac: float = 0.2,
    val_min: int = 8,
    min_sku_history: int = 16
) -> Tuple[float, pd.DataFrame]:
    """
    Sweep tau² on the log-mean scale. `sigma2_mean_by_sku` must be the
    sampling variance of each SKU's mean log-residual (var / n), not
    the NB count variance.
    """
    # Build validation mask for temporal holdout
    max_t = sl.groupby(sku_grp)["t"].transform("max")
    len_t = sl.groupby(sku_grp)["t"].transform("size")
    val_len = np.maximum(val_min, (val_frac * len_t).astype(int))
    sl["t_cut"] = max_t - val_len
    sl["is_valid"] = (sl["t"] > sl["t_cut"]) & (len_t >= min_sku_history)
    
    # Sweep
    results = []
    best = {"tau2": None, "rmse": np.inf, "median_w": None}
    
    delta_raw_full = log_resid_mean_by_sku
    sigma2_i_full = sigma2_mean_by_sku.reindex(log_resid_mean_by_sku.index).fillna(sigma2_mean_by_sku.median())
    
    for t2 in tau2_grid:
        w_g = t2 / (t2 + sigma2_i_full.replace(0.0, 1e-6))
        delta_g = (w_g * delta_raw_full)
        delta_g_dict = delta_g.to_dict()
        
        sl_tmp = sl.copy()
        sl_tmp["delta_sku_grid"] = sl_tmp[sku_grp].apply(tuple, axis=1).map(delta_g_dict).fillna(0.0)
        mu_eb_g = sl_tmp["mu_global"] * np.exp(sl_tmp["delta_sku_grid"])
        
        y_val = sl_tmp.loc[sl_tmp["is_valid"], "y"].values
        mu_val = mu_eb_g.loc[sl_tmp["is_valid"]].values
        
        rmse = float(np.sqrt(np.mean((y_val - mu_val) ** 2))) if len(y_val) > 0 else np.inf
        med_w = float(np.median(w_g.values if hasattr(w_g, 'values') else w_g))
        results.append({"tau2": t2, "rmse": rmse, "median_w": med_w})
        
        if rmse < best["rmse"]:
            best = {"tau2": t2, "rmse": rmse, "median_w": med_w}
    
    # Apply best tau2
    w_best = best["tau2"] / (best["tau2"] + sigma2_i_full.replace(0.0, 1e-6))
    shrunk_delta_best = (w_best * delta_raw_full).rename("delta_sku")
    sl["delta_sku"] = sl[sku_grp].apply(tuple, axis=1).map(shrunk_delta_best.to_dict()).fillna(0.0)
    sl["mu_eb"] = sl["mu_global"] * np.exp(sl["delta_sku"])
    
    print(f"✅ Tau² sweep complete: best={best['tau2']:.4f}, RMSE={best['rmse']:.2f}")
    
    return best["tau2"], sl


# ======================================================================
# PER-SEGMENT DISPERSION ESTIMATION
# ======================================================================

def estimate_segment_dispersion(
    sl: pd.DataFrame,
    sku_grp: list,
    segment_col: str = "Department",
    alpha_cap: float = 10.0,
    alpha_k: int = 8
) -> pd.DataFrame:
    """
    Estimate per-segment NB dispersion with SKU-level shrinkage.
    
    Returns:
        sl with alpha_seg and alpha_eff columns added
    """
    # Get baseline alpha from GLM
    alpha = sl.get("alpha_baseline", pd.Series(1.0, index=sl.index)).iloc[0] if "alpha_baseline" in sl.columns else 1.0
    
    # Per-segment alpha via method-of-moments
    seg_alpha = {}
    for seg, g in sl.groupby(segment_col):
        mu = g["mu_eb"].values
        y = g["y"].values
        if len(mu) < 10:
            continue
        var = np.var(y - mu) + np.mean(mu + (1.0 * mu**2)) - np.mean(mu)
        denom = np.mean(mu**2) + 1e-6
        a = max((var - np.mean(mu)) / denom, 0.0)
        seg_alpha[seg] = a
    
    sl["alpha_seg"] = sl[segment_col].map(seg_alpha).fillna(alpha)
    
    # Per-SKU alpha with shrinkage toward segment
    sku_counts = sl.groupby(sku_grp)["y"].count().rename("n_obs")
    sku_mean = sl.groupby(sku_grp)["y"].mean().rename("mean")
    sku_var = sl.groupby(sku_grp)["y"].var().rename("var")
    alpha_raw = ((sku_var - sku_mean) / (sku_mean**2 + 1e-6)).clip(lower=0.0).rename("alpha_raw")
    
    seg_alpha_map = sl.groupby(sku_grp)["alpha_seg"].mean()
    
    alpha_df = pd.concat([sku_counts, alpha_raw, seg_alpha_map], axis=1)
    alpha_df["alpha_seg"] = alpha_df["alpha_seg"].fillna(alpha)
    alpha_df["w"] = alpha_df["n_obs"] / (alpha_df["n_obs"] + alpha_k)
    alpha_df["alpha_shrunk"] = (
        alpha_df["w"] * alpha_df["alpha_raw"] + (1.0 - alpha_df["w"]) * alpha_df["alpha_seg"]
    ).clip(upper=alpha_cap)
    
    # Map back to rows
    alpha_by_sku_shrunk = alpha_df["alpha_shrunk"]
    sl["alpha_eff"] = sl[sku_grp].apply(tuple, axis=1).map(alpha_by_sku_shrunk.to_dict()).fillna(alpha)
    
    print(f"✅ Segment dispersion estimated: {len(seg_alpha)} segments, median alpha={alpha_df['alpha_shrunk'].median():.3f}")
    
    return sl


# ======================================================================
# BUILD DESIGN MATRIX
# ======================================================================

def build_design_matrix(
    sales_long: pd.DataFrame,
    master: pd.DataFrame,
    col_mapping: Dict[str, str],
    p0_series: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Build design matrix with hierarchy, seasonality, and optional p0.
    
    Returns:
        sl DataFrame with columns: Store, Product, Division, Department, 
        ProductGroup, week_monday, t, sin52, cos52, y, [p0]
    """
    # Detect optional store-hierarchy columns
    store_level_candidates = [
        "Region", "Cluster", "District", "StoreGroup", "StoreCluster", "City", "Zone", "Area"
    ]
    present_store_levels = [c for c in store_level_candidates if c in master.columns]
    
    required_master = {col_mapping["store"], col_mapping["product"], "Division", "Department", "ProductGroup"}
    cols_to_merge = sorted(list(required_master | set(present_store_levels)))
    
    # Merge hierarchy onto sales
    sl = sales_long.merge(master[cols_to_merge], on=[col_mapping["store"], col_mapping["product"]], how="left")
    
    if sl[["Division", "Department", "ProductGroup"]].isna().any().any():
        raise ValueError("Master join produced missing hierarchy labels")
    
    # Add time index and Fourier terms
    sl["week_monday"] = sl[col_mapping["week"]].dt.to_period("W-MON").dt.to_timestamp()
    sl = sl.sort_values([col_mapping["store"], col_mapping["product"], "week_monday"])
    
    sl["t"] = sl.groupby([col_mapping["store"], col_mapping["product"]])["week_monday"].rank(method="first").astype(int)
    period = 52
    sl["sin52"] = np.sin(2 * np.pi * sl["t"] / period)
    sl["cos52"] = np.cos(2 * np.pi * sl["t"] / period)
    
    # Keep necessary columns
    keep_cols = [col_mapping["store"], col_mapping["product"]] + present_store_levels + [
        "Division", "Department", "ProductGroup", "week_monday", "t", "sin52", "cos52", col_mapping["qty"]
    ]
    sl = sl[keep_cols].rename(columns={col_mapping["qty"]: "y"})
    
    # Merge p0 if provided
    if p0_series is not None:
        sl = sl.merge(p0_series.reset_index(), on=[col_mapping["store"], col_mapping["product"]], how="left")
        sl["p0"] = sl["p0"].fillna(0.0)
    
    print(f"✅ Design matrix built: {len(sl)} rows, {sl.groupby([col_mapping['store'], col_mapping['product']]).ngroups} SKUs")
    print(f"   Store hierarchy used: {present_store_levels or 'None'}")
    
    return sl


# ======================================================================
# IDENTIFY INTERMITTENT SKUS
# ======================================================================

def identify_intermittent_skus(
    sl: pd.DataFrame,
    sku_grp: list,
    p0_threshold: float = 0.5,
    mu_threshold: float = 2.0
) -> pd.Series:
    """
    Identify intermittent SKUs for Hurdle model routing.
    
    Returns:
        Boolean Series indexed by (Store, Product)
    """
    sku_stats = sl.groupby(sku_grp).agg({
        "y": ["mean"],
        "p0": "first"
    })
    sku_stats.columns = ["mu_weekly", "p0"]
    
    intermittent = (sku_stats["p0"] >= p0_threshold) & (sku_stats["mu_weekly"] < mu_threshold)
    
    n_intermittent = intermittent.sum()
    print(f"✅ Intermittent SKUs identified: {n_intermittent} / {len(sku_stats)} ({n_intermittent/len(sku_stats)*100:.1f}%)")
    
    return intermittent

