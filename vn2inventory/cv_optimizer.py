"""
Cross-Validation Optimization Module

Implements Optuna-based CV tuning for policy parameters.
"""

import numpy as np
import pandas as pd
import optuna
from dataclasses import dataclass
from typing import Callable, Tuple, Dict


@dataclass
class PolicyKnobs:
    """Tunable policy parameters."""
    k_safety: float       # Scales target quantile (1.0 = exact cost-consistent α)
    calib_sigma: float    # Adjusts overdispersion (alpha -> alpha/calib_sigma)


def validate_cv_cutoffs(
    sales_long: pd.DataFrame,
    cutoffs: list,
    col_week: str,
    min_future: int = 4
) -> bool:
    """
    Validate that each CV cutoff has enough future weeks for evaluation.
    
    Returns:
        True if all cutoffs valid, False otherwise
    """
    weeks = pd.to_datetime(sales_long[col_week].unique())
    results = {}
    
    print("\n📊 CV CUTOFF VALIDATION:")
    print("="*70)
    
    for c in cutoffs:
        future_weeks = (weeks > pd.to_datetime(c)).sum()
        results[c] = future_weeks >= min_future
        status = "✅" if results[c] else "❌"
        print(f"{status} {c}: {future_weeks} weeks available (need >={min_future})")
    
    all_valid = all(results.values())
    
    if not all_valid:
        print("\n⚠️ WARNING: Some CV cutoffs don't have enough future data!")
        print("   Suggested fix: Use earlier cutoffs like:")
        print("   ['2024-02-05', '2024-02-12', '2024-02-19', '2024-02-26']")
        print("\n   OR: Skip CV and use segmented q_target for safer submission")
    
    print("="*70)
    
    return all_valid


def make_cv_objective(
    cv_cutoffs: list,
    compute_orders_fn: Callable,
    simulate_cost_fn: Callable,
    horizon_weeks: int = 4
):
    """
    Create Optuna objective that evaluates across CV folds.
    
    Args:
        cv_cutoffs: List of historical cutoff dates
        compute_orders_fn: Function(cutoff, knobs) -> orders Series
        simulate_cost_fn: Function(cutoff, orders, weeks_ahead) -> cost
        horizon_weeks: Number of weeks to simulate per fold
        
    Returns:
        objective function for Optuna
    """
    def objective(trial):
        # Sample knobs
        k_safety = trial.suggest_float("k_safety", 0.80, 1.20)
        calib_sigma = trial.suggest_float("calib_sigma", 0.70, 1.30)
        
        knobs = PolicyKnobs(k_safety=k_safety, calib_sigma=calib_sigma)
        
        # Evaluate on each CV fold
        fold_costs = []
        for cutoff in cv_cutoffs:
            orders = compute_orders_fn(cutoff, knobs)
            cost = simulate_cost_fn(cutoff, orders, horizon_weeks)
            
            if not np.isfinite(cost):
                return float('inf')  # Skip invalid trials
            
            fold_costs.append(cost)
        
        # Return median cost (robust to outlier folds)
        return float(np.median(fold_costs))
    
    return objective


def run_cv_optimization(
    cv_cutoffs: list,
    compute_orders_fn: Callable,
    simulate_cost_fn: Callable,
    n_trials: int = 60,
    timeout: int = 300,
    horizon_weeks: int = 4,
    seed: int = 42
) -> optuna.Study:
    """
    Run Optuna CV optimization to find best policy knobs.
    
    Returns:
        Optuna study with best parameters
    """
    print("\n" + "=" * 70)
    print(f"RUNNING CV TUNING: {len(cv_cutoffs)} folds × {n_trials} trials")
    print(f"CV folds: {cv_cutoffs}")
    print("=" * 70)
    
    # Suppress Optuna verbosity
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # Create study
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=seed,
            multivariate=True,
            group=True
        ),
        pruner=optuna.pruners.MedianPruner(
            n_warmup_steps=max(5, len(cv_cutoffs)//2)
        )
    )
    
    # Optimize
    study.optimize(
        make_cv_objective(cv_cutoffs, compute_orders_fn, simulate_cost_fn, horizon_weeks),
        n_trials=n_trials,
        timeout=timeout,
        gc_after_trial=True,
        show_progress_bar=True
    )
    
    print("\n" + "=" * 70)
    print("CV TUNING COMPLETE")
    print(f"Best params: {study.best_params}")
    print(f"Best value (median CV cost): {study.best_value:.1f}")
    print("=" * 70)
    
    return study

