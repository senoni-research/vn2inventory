"""
Friend's Recommendations: Core Utilities

Implements canonical NB2 samplers, availability-aware p0, ABC/XYZ segmentation,
and segmented service targets.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict


# ======================================================================
# CANONICAL NB2 SAMPLERS
# ======================================================================

def draw_nb2_canonical(mu: float, phi: float, size, rng) -> np.ndarray:
    """
    Single canonical NB2 drawer with correct parameterization.
    
    NB2: Var = mu + phi*mu^2
    phi: overdispersion parameter (renamed from 'alpha' to avoid confusion)
    """
    n = 1.0 / max(phi, 1e-12)
    p = n / (n + max(mu, 0.0))  # == 1/(1 + phi*mu) ✅
    return rng.negative_binomial(n, p, size=size)


def sample_D3_nb_canonical(
    mu: float, 
    phi: float, 
    weeks: int = 3, 
    K: int = 2000, 
    seed: int = 42
) -> np.ndarray:
    """
    3-week NB2 demand sampler.
    
    Returns K samples of total demand over 'weeks' periods.
    """
    rng = np.random.default_rng(seed)
    if mu <= 0 or phi <= 0:
        return np.zeros(K, dtype=int)
    draws = draw_nb2_canonical(mu, phi, size=(K, weeks), rng=rng)
    return draws.sum(axis=1).astype(int)


def sample_D3_hurdle_canonical(
    mu: float, 
    phi: float, 
    p0: float, 
    weeks: int = 3, 
    K: int = 2000, 
    seed: int = 42
) -> np.ndarray:
    """
    3-week hurdle (Bernoulli × NB2) sampler.

    `mu` is the unconditional weekly mean E[Y], zeros included.
    The positive component is therefore mu / (1 - p0). Passing
    unconditional mu straight into the NB gate would apply p0 twice.
    """
    rng = np.random.default_rng(seed)
    if mu <= 0:
        return np.zeros(K, dtype=int)

    p0 = float(np.clip(p0, 0.0, 1.0 - 1e-6))
    mu_plus = float(mu) / (1.0 - p0)

    dsum = np.zeros(K, dtype=int)
    for _ in range(weeks):
        active = rng.random(K) < (1.0 - p0)
        nb = draw_nb2_canonical(mu_plus, phi, size=K, rng=rng)
        dsum += (nb * active).astype(int)
    return dsum


# ======================================================================
# AVAILABILITY-AWARE P0
# ======================================================================

def compute_p0_available_only(
    sales_long: pd.DataFrame,
    data_dir: Path,
    col_mapping: Dict[str, str]
) -> pd.Series:
    """
    Compute zero-rate P(y=0) on available weeks only.
    
    If availability data exists, filters to in-stock weeks first.
    Prevents bias in intermittency classification from OOS periods.
    """
    avail_path = data_dir / "Week 0 - In Stock.csv"
    
    if avail_path.exists():
        try:
            avail_wide = pd.read_csv(avail_path).set_index([col_mapping["store"], col_mapping["product"]])
            avail_wide.columns = pd.to_datetime(avail_wide.columns)
            
            avail_long = (avail_wide.stack()
                          .rename("available")
                          .reset_index()
                          .rename(columns={"level_2": col_mapping["week"]}))
            avail_long["available"] = avail_long["available"].astype(bool)
            
            s = sales_long.merge(
                avail_long, 
                on=[col_mapping["store"], col_mapping["product"], col_mapping["week"]], 
                how="left"
            )
            s["available"] = s["available"].fillna(True)
            
            # Compute p0 ONLY on available weeks
            p0 = (s.loc[s["available"]]
                    .groupby([col_mapping["store"], col_mapping["product"]])[col_mapping["qty"]]
                    .apply(lambda x: (x.fillna(0) == 0).mean()))
            
            print(f"✅ P0 computed on available weeks only ({avail_path.name} found)")
            return p0.rename("p0")
            
        except Exception as e:
            print(f"⚠️ Availability file exists but failed to load ({e}); using all weeks")
    
    # Fallback: compute on all weeks if availability not present
    print("⚠️ No availability data; computing p0 on all weeks (may bias intermittency)")
    return sales_long.groupby([col_mapping["store"], col_mapping["product"]])[col_mapping["qty"]].apply(
        lambda x: (x==0).mean()
    ).rename("p0")


# ======================================================================
# ABC/XYZ SEGMENTATION
# ======================================================================

def classify_xyz(row: pd.Series) -> str:
    """Classify SKU variability: X=stable, Y=variable, Z=intermittent."""
    if row["zero_rate"] >= 0.60 or row["cv"] >= 2.0:
        return "Z"
    if row["zero_rate"] >= 0.30 or row["cv"] >= 1.0:
        return "Y"
    return "X"


def compute_abc_xyz_segmentation(
    sl: pd.DataFrame,
    col_mapping: Dict[str, str]
) -> pd.DataFrame:
    """
    Segment SKUs by volume (ABC) and variability (XYZ).
    
    Returns sku_meta DataFrame with ABC, XYZ, and velocity stats.
    """
    # Compute SKU-level stats
    sku_stats = sl.groupby([col_mapping["store"], col_mapping["product"]]).agg(
        mu_weekly=("y", "mean"),
        sd_weekly=("y", "std"),
        zero_rate=("p0", "first")  # Use availability-aware p0
    ).reset_index()
    
    sku_stats["cv"] = (sku_stats["sd_weekly"] / sku_stats["mu_weekly"].replace(0, np.nan)).fillna(np.inf)
    
    # ABC: Volume-based (A=high, B=mid, C=low)
    qA, qB = sku_stats["mu_weekly"].quantile([0.90, 0.60])
    sku_stats["ABC"] = np.where(sku_stats["mu_weekly"] >= qA, "A",
                                np.where(sku_stats["mu_weekly"] >= qB, "B", "C"))
    
    # XYZ: Variability-based
    sku_stats["XYZ"] = sku_stats.apply(classify_xyz, axis=1)
    
    # Additional velocity stats for caps
    wk = sl[[col_mapping["store"], col_mapping["product"], "week_monday", "y"]].copy()
    p95_weekly = wk.groupby([col_mapping["store"], col_mapping["product"]])["y"].quantile(0.95).rename("p95_weekly")
    hist_max_weekly = wk.groupby([col_mapping["store"], col_mapping["product"]])["y"].max().rename("hist_max_weekly")
    
    # Combine into sku_meta
    sku_meta = (sku_stats
                .merge(p95_weekly.reset_index(), on=[col_mapping["store"], col_mapping["product"]], how="left")
                .merge(hist_max_weekly.reset_index(), on=[col_mapping["store"], col_mapping["product"]], how="left"))
    
    # Display segmentation summary
    print("\n📊 ABC/XYZ SEGMENTATION SUMMARY:")
    seg_counts = sku_meta.groupby(["ABC", "XYZ"]).size().unstack(fill_value=0)
    print(seg_counts)
    print(f"\n   Total SKUs: {len(sku_meta)}")
    
    return sku_meta


# ======================================================================
# SEGMENTED SERVICE TARGETS
# ======================================================================

def q_target_for_segment(abc: str, xyz: str) -> float:
    """
    Return target quantile based on ABC/XYZ segment.
    
    A/X: Priority items, avoid stockouts → high q
    C/Z: Intermittent, low priority → lower q
    
    Friend's recommendation: Segment-specific targets hedge model uncertainty
    and balance holding vs. shortage costs optimally across SKU types.
    """
    table = {
        ('A', 'X'): 0.89, ('A', 'Y'): 0.86, ('A', 'Z'): 0.83,
        ('B', 'X'): 0.85, ('B', 'Y'): 0.83, ('B', 'Z'): 0.81,
        ('C', 'X'): 0.82, ('C', 'Y'): 0.80, ('C', 'Z'): 0.79,
    }
    return table.get((abc, xyz), 0.83)  # Default: cost-consistent baseline


# ======================================================================
# ORDER COMPUTATION WITH GUARDRAILS
# ======================================================================

def compute_order_with_guardrails(
    need: int,
    inv_pos: int,
    d3_samples: np.ndarray,
    hist_max: float,
    p95: float,
    abc: str,
    xyz: str,
    mu: float,
    protection_weeks: int = 3
) -> int:
    """
    Compute order with multiple guardrails.
    
    Implements:
    - Statistical cap (q99.5)
    - Empirical cap (2× hist_max)
    - Velocity cap (p95 × weeks)
    - Floor for priority items (A/X)
    - Triage (prevent obvious misses)
    """
    # MULTIPLE CAPS (statistical, empirical, velocity)
    q995 = int(np.quantile(d3_samples, 0.995))
    cap_stat = max(0, q995 - inv_pos)
    
    cap_emp = max(10, int(2.0 * float(hist_max or 0) * protection_weeks) - inv_pos)
    
    cap_vel = int(float(p95 or 0) * protection_weeks) if p95 else cap_emp
    
    # Final cap (most restrictive, but ignore vel if zero)
    cap = max(0, min(cap_stat, cap_emp, cap_vel) if cap_vel > 0 else min(cap_stat, cap_emp))
    
    # Floor: small hedge for high-priority items (A/X)
    floor_units = 2 if (abc == "A" and xyz == "X" and need > 0) else 0
    
    order = max(floor_units, min(max(0, need), cap))
    
    # TRIAGE: if model says zero but demand looks positive, order at least 1
    if order == 0 and need > 0 and mu > 0.2:
        order = 1
    
    return int(order)


# ======================================================================
# SUBMISSION QA
# ======================================================================

def validate_submission(sub: pd.DataFrame) -> Tuple[bool, list]:
    """
    Run 6 automated QA checks on submission DataFrame.
    
    Returns:
        (all_pass, check_results)
    """
    checks = [
        (len(sub) == 599, f"Row count: {len(sub)} == 599"),
        (list(sub.columns) == ["Store", "Product", "0"], f"Columns: {list(sub.columns)}"),
        (sub["0"].ge(0).all(), f"Non-negative: min={sub['0'].min()}"),
        ((sub["0"] == sub["0"].astype(int)).all(), f"Integer orders"),
        (sub["0"].notna().all(), f"No NaN values"),
        (sub[["Store", "Product"]].duplicated().sum() == 0, f"No duplicate SKUs"),
    ]
    
    all_pass = all(passed for passed, _ in checks)
    return all_pass, checks


def export_submission_safe(sub: pd.DataFrame, output_path: Path) -> None:
    """
    Export submission with QA checks and proper formatting.
    
    CRITICAL: Uses reset_index(drop=True).to_csv(..., index=False)
    to ensure Store/Product appear as columns.
    """
    # Ensure correct column order
    sub = sub[["Store", "Product", "0"]].copy()
    
    # Coerce to non-negative integers per submission spec
    sub["0"] = np.maximum(0, np.rint(sub["0"]).astype(int))
    
    # Run QA
    all_pass, checks = validate_submission(sub)
    
    print("\n🔍 SUBMISSION QA CHECKS:")
    print("=" * 70)
    for passed, msg in checks:
        status = "✅" if passed else "❌"
        print(f"{status} {msg}")
    print("=" * 70)
    
    if not all_pass:
        raise ValueError("❌ SUBMISSION QA FAILED - fix errors before export")
    
    # Export with proper reset_index (CRITICAL FIX)
    sub.reset_index(drop=True).to_csv(output_path, index=False)
    
    print(f"\n✅ SUBMISSION VALIDATED & EXPORTED")
    print(f"   Path: {output_path}")
    print(f"   Size: {output_path.stat().st_size:,} bytes")
    print(f"   Total units: {sub['0'].sum():,}")
    print(f"   Mean: {sub['0'].mean():.2f}")
    print(f"   Zeros: {(sub['0'] == 0).sum()} ({(sub['0'] == 0).sum()/len(sub)*100:.1f}%)")
    print(f"   Max: {sub['0'].max()}")
    print("=" * 70)

