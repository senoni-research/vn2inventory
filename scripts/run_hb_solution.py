#!/usr/bin/env python3
"""
Hierarchical Bayes Solution CLI

Usage:
    python scripts/run_hb_solution.py --model baseline
    python scripts/run_hb_solution.py --model graph-enhanced
    python scripts/run_hb_solution.py --model baseline --skip-cv
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from vn2inventory.hb_pipeline import HBPipeline, HBConfig
from vn2inventory.cv_optimizer import run_cv_optimization, validate_cv_cutoffs, PolicyKnobs
from vn2inventory.friend_recommendations import export_submission_safe


def main():
    parser = argparse.ArgumentParser(description="Run Hierarchical Bayes solution")
    parser.add_argument(
        "--model",
        choices=["baseline", "graph-enhanced"],
        default="baseline",
        help="Which model variant to run"
    )
    parser.add_argument(
        "--skip-cv",
        action="store_true",
        help="Skip CV optimization and use default knobs (faster)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to data directory (default: ../data)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Path to output directory (default: ../submissions)"
    )
    parser.add_argument(
        "--apply-cap",
        action="store_true",
        help="Apply targeted cap vs baseline HB (graph runs only, requires baseline to exist)"
    )
    parser.add_argument(
        "--run-both",
        action="store_true",
        help="Run baseline and graph sequentially, then pick the winner via weekly judge"
    )
    parser.add_argument(
        "--features-599",
        type=str,
        default=None,
        help="Path to 599-row graph features CSV (required for --model graph-enhanced)"
    )
    parser.add_argument(
        "--cutoff",
        type=str,
        default="2024-04-08",
        help="Cutoff week date (YYYY-MM-DD), e.g., 2024-04-15"
    )
    parser.add_argument(
        "--state-csv",
        type=str,
        default=None,
        help="Override current state CSV (downloaded dashboard state file)"
    )
    parser.add_argument(
        "--sales-wide",
        type=str,
        default=None,
        help="Override sales wide CSV (e.g., Week 1 - 2024-04-15 - Sales.csv)"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    data_dir = Path(args.data_dir) if args.data_dir else ROOT / "data"
    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "submissions"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Graph features path (only for graph-enhanced model)
    if args.model == "graph-enhanced":
        # Prefer CLI-provided features-599 path
        graph_features_path = Path(args.features_599) if args.features_599 else None
        if graph_features_path and not graph_features_path.exists():
            print(f"⚠️ Graph features not found: {graph_features_path}")
            print("   Falling back to baseline model")
            graph_features_path = None
    else:
        graph_features_path = None
    
    # Create config
    config = HBConfig(
        data_dir=data_dir,
        output_dir=output_dir,
        graph_features_path=graph_features_path,
        state_csv_override=Path(args.state_csv) if args.state_csv else None,
        sales_wide_override=Path(args.sales_wide) if args.sales_wide else None,
    )
    
    print("="*70)
    print(f"🚀 RUNNING HIERARCHICAL BAYES: {args.model.upper()}")
    print("="*70)
    print(f"Data dir: {data_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Graph features: {'Yes' if graph_features_path else 'No'}")
    print(f"CV: {'Disabled' if args.skip_cv else 'Enabled'}")
    print("="*70)
    
    # Initialize pipeline
    pipeline = HBPipeline(config)
    
    # Step 1: Load data
    print("\n📂 STEP 1: Loading data...")
    pipeline.load_data()
    
    # Validate state reconstruction (if CV will be used)
    if not args.skip_cv and pipeline.state_reconstructor is not None:
        from vn2inventory.state_reconstruction import validate_state_reconstruction
        test_cutoffs = ["2024-02-05", "2024-02-12", "2024-02-19", "2024-02-26", "2024-04-08"]
        validate_state_reconstruction(
            pipeline.state_reconstructor,
            "2024-04-08",
            pipeline.state_df,
            test_cutoffs
        )
    
    # Step 2: Load graph features (if applicable)
    if args.model == "graph-enhanced":
        print("\n📊 STEP 2: Loading graph features...")
        pipeline.load_graph_features_if_available()
    
    # Step 3: Fit model
    print("\n🔬 STEP 3: Fitting hierarchical model...")
    pipeline.fit_model()
    
    # Step 4: CV optimization or use defaults
    if args.skip_cv:
        print("\n⏭️  STEP 4: Skipping CV (using default knobs)")
        # Safer defaults (service-tilted, calibrated dispersion)
        best_knobs = PolicyKnobs(k_safety=1.00, calib_sigma=1.10)
        print(f"   Using: k_safety={best_knobs.k_safety}, calib_sigma={best_knobs.calib_sigma}")
    else:
        print("\n🎯 STEP 4: Running CV optimization...")
        
        # Define CV cutoffs
        cv_cutoffs = [
            "2024-02-05",
            "2024-02-12",
            "2024-02-19",
            "2024-02-26",
        ]
        
        # Validate cutoffs
        cv_valid = validate_cv_cutoffs(
            pipeline.sales_long,
            cv_cutoffs,
            config.col_week,
            config.cv_horizon_weeks
        )
        
        if not cv_valid:
            print("\n⚠️ CV cutoffs invalid; using default knobs instead")
            best_knobs = PolicyKnobs(k_safety=1.00, calib_sigma=1.10)
        else:
            # Run CV
            from vn2inventory.sim_env import InventorySim, Costs
            
            def simulate_cost(cutoff, orders, weeks_ahead):
                """Simplified simulator for CV."""
                try:
                    df_orders = orders.reset_index()
                    df_orders.columns = [config.col_store, config.col_product, "0"]
                    
                    cutoff_dt = pd.to_datetime(cutoff)
                    future_weeks = pipeline.sales_long[pipeline.sales_long[config.col_week] > cutoff_dt].copy()
                    unique_weeks = sorted(future_weeks[config.col_week].unique())[:weeks_ahead]
                    
                    if len(unique_weeks) == 0:
                        return float('inf')
                    
                    future_subset = future_weeks[future_weeks[config.col_week].isin(unique_weeks)]
                    sales_cv = future_subset.pivot_table(
                        index=[config.col_store, config.col_product],
                        columns=config.col_week,
                        values=config.col_qty,
                        fill_value=0.0
                    )
                    
                    costs = Costs(shortage_per_unit=config.shortage_cost)
                    state_cv = pipeline._get_state_at(cutoff)
                    
                    init_state = state_cv.reset_index()
                    init_state.columns = [config.col_store, config.col_product, 
                                         "End Inventory", "In Transit W+1", "In Transit W+2"]
                    
                    sim = InventorySim(
                        sales_wide=sales_cv.reset_index(),
                        initial_state=init_state,
                        costs=costs
                    )
                    sim.t = 0
                    
                    total_cost = 0.0
                    orders_placed = False
                    
                    for week_idx in range(len(unique_weeks)):
                        if not orders_placed:
                            orders_week = df_orders.set_index([config.col_store, config.col_product])["0"].reindex(sales_cv.index).fillna(0)
                            orders_placed = True
                        else:
                            orders_week = pd.Series(0, index=sales_cv.index)
                        
                        week_costs = sim.step(orders=orders_week)
                        total_cost += week_costs["holding_cost"] + week_costs["shortage_cost"]
                    
                    return float(total_cost)
                except Exception as e:
                    print(f"⚠️ CV simulation error: {e}")
                    return float('inf')
            
            study = run_cv_optimization(
                cv_cutoffs,
                pipeline.compute_orders_for_cutoff,
                simulate_cost,
                n_trials=config.n_cv_trials,
                timeout=config.cv_timeout,
                horizon_weeks=config.cv_horizon_weeks,
                seed=config.mc_seed
            )
            
            best_knobs = PolicyKnobs(**study.best_params)
            
            print(f"\n📈 BEST KNOBS FROM CV:")
            print(f"   k_safety: {best_knobs.k_safety:.4f}")
            print(f"   calib_sigma: {best_knobs.calib_sigma:.4f}")
            print(f"   Effective alpha: {config.cost_consistent_alpha * best_knobs.k_safety:.4f}")
    
    # Helper: weekly expected cost judge (uses weekly moments)
    def weekly_expected_cost(features_path: Path, submission_path: Path, qty_col: str | None = None) -> float:
        import math
        feat = pd.read_csv(features_path)
        feat["Store"] = feat.get("store_id", feat.get("Store")).astype(str)
        feat["Product"] = feat.get("product_id", feat.get("Product")).astype(str)
        need = ["Store","Product","mu_w1","mu_w2","mu_w3","sigma_w1","sigma_w2","sigma_w3"]
        for c in need:
            if c not in feat.columns:
                raise ValueError(f"weekly judge requires column {c} in {features_path}")
        feat = feat[need]
        state = pd.read_csv(data_dir/"Week 0 - 2024-04-08 - Initial State.csv")
        state["Store"] = state["Store"].astype(str); state["Product"] = state["Product"].astype(str)
        state["onhand"] = pd.to_numeric(state["End Inventory"], errors="coerce").fillna(0)
        state["onorder_le2"] = (
            pd.to_numeric(state.get("In Transit W+1"), errors="coerce").fillna(0) +
            pd.to_numeric(state.get("In Transit W+2"), errors="coerce").fillna(0)
        )
        st = state[["Store","Product","onhand","onorder_le2"]]
        sub = pd.read_csv(submission_path)
        if "Store" not in sub.columns or "Product" not in sub.columns:
            sub = sub.rename(columns={"store_id":"Store","product_id":"Product"})
        sub["Store"] = sub["Store"].astype(str); sub["Product"] = sub["Product"].astype(str)
        if qty_col is None:
            for c in ["0","order_qty","qty","orders"]:
                if c in sub.columns:
                    qty_col = c; break
            if qty_col is None:
                qty_col = sub.columns[-1]
        sub = sub[["Store","Product", qty_col]].rename(columns={qty_col:"qty"})
        base = feat.merge(sub, on=["Store","Product"], how="inner").merge(st, on=["Store","Product"], how="left")
        base = base.fillna({"onhand":0,"onorder_le2":0})
        Q = pd.to_numeric(base["qty"], errors="coerce").fillna(0).to_numpy()
        a1 = (base["onhand"] + base["onorder_le2"]).to_numpy() + Q
        mu1 = pd.to_numeric(base["mu_w1"], errors="coerce").fillna(0).to_numpy()
        mu2 = pd.to_numeric(base["mu_w2"], errors="coerce").fillna(0).to_numpy()
        mu3 = pd.to_numeric(base["mu_w3"], errors="coerce").fillna(0).to_numpy()
        s1  = pd.to_numeric(base["sigma_w1"], errors="coerce").fillna(1e-6).to_numpy()
        s2  = pd.to_numeric(base["sigma_w2"], errors="coerce").fillna(1e-6).to_numpy()
        s3  = pd.to_numeric(base["sigma_w3"], errors="coerce").fillna(1e-6).to_numpy()
        SQRT2PI = math.sqrt(2.0 * math.pi)
        def week_cost(mu, sigma, a, c_s=1.0, c_h=0.2):
            sigma = max(float(sigma), 1e-6)
            z = (a - mu)/sigma
            phi = math.exp(-0.5*z*z)/SQRT2PI
            Phi = 0.5*(1.0 + math.erf(z/math.sqrt(2)))
            over = sigma*phi + (a - mu)*Phi
            under = sigma*phi + (mu - a)*(1 - Phi)
            return c_s*under + c_h*over
        c1 = np.array([week_cost(mu1[i], s1[i], a1[i]) for i in range(len(a1))])
        a2 = np.maximum(0.0, a1 - mu1)
        c2 = np.array([week_cost(mu2[i], s2[i], a2[i]) for i in range(len(a2))])
        a3 = np.maximum(0.0, a2 - mu2)
        c3 = np.array([week_cost(mu3[i], s3[i], a3[i]) for i in range(len(a3))])
        return float(c1.sum() + c2.sum() + c3.sum())

    # Step 5: Generate final submission
    print("\n📦 STEP 5: Generating final submission...")
    
    # Load baseline orders if needed for capping
    baseline_orders_df = None
    if args.apply_cap and args.model == "graph-enhanced":
        baseline_path = ROOT / "submissions" / "baseline_cv" / "orders_hierarchical_final_store_cv.csv"
        if baseline_path.exists():
            baseline_orders_df = pd.read_csv(baseline_path)
            print(f"   Loaded baseline for capping: {baseline_path}")
        else:
            print(f"   ⚠️ Baseline not found at {baseline_path}, skipping cap")
            args.apply_cap = False
    
    if args.run_both:
        # Run baseline first into a temp folder
        base_out = ROOT / "submissions" / "_auto_baseline"
        base_out.mkdir(parents=True, exist_ok=True)
        pipe_base = HBPipeline(HBConfig(data_dir=data_dir, output_dir=base_out, graph_features_path=None))
        pipe_base.load_data(); pipe_base.fit_model()
        # Run a short CV or reuse knobs from above; use best_knobs if available
        sub_base = pipe_base.generate_submission(best_knobs, cutoff_week=args.cutoff)
        bp = base_out / "orders_hierarchical_final_store_cv.csv"
        export_submission_safe(sub_base, bp)

        # Run graph into a temp folder
        graph_out = ROOT / "submissions" / "_auto_graph"
        graph_out.mkdir(parents=True, exist_ok=True)
        pipe_graph = HBPipeline(HBConfig(data_dir=data_dir, output_dir=graph_out, graph_features_path=Path(args.features_599)))
        pipe_graph.load_data(); pipe_graph.load_graph_features_if_available(); pipe_graph.fit_model()
        sub_graph = pipe_graph.generate_submission(best_knobs, cutoff_week=args.cutoff, apply_hb_cap=True, baseline_orders=sub_base)
        gp = graph_out / "orders_hierarchical_graph_enhanced_cv.csv"
        export_submission_safe(sub_graph, gp)

        # Weekly judge
        cost_base = weekly_expected_cost(Path(args.features_599), bp)
        cost_graph = weekly_expected_cost(Path(args.features_599), gp)
        print(f"\n🧮 WEEKLY JUDGE:")
        print(f"   Baseline CV: {cost_base:.2f}")
        print(f"   Graph CV (capped): {cost_graph:.2f}")
        winner_path = gp if cost_graph < cost_base else bp
        submission = pd.read_csv(winner_path)
        print(f"   Winner: {'graph' if cost_graph < cost_base else 'baseline'} → {winner_path}")
    else:
        submission = pipeline.generate_submission(
            best_knobs,
            cutoff_week=args.cutoff,
            apply_hb_cap=args.apply_cap,
            baseline_orders=baseline_orders_df
        )
    
    # Step 6: Export with QA
    print("\n💾 STEP 6: Exporting submission...")
    
    if args.skip_cv:
        if args.model == "graph-enhanced" and pipeline.graph_available:
            filename = "orders_hierarchical_graph_enhanced.csv"
        else:
            filename = "orders_hierarchical_final.csv"
    else:
        if args.model == "graph-enhanced" and pipeline.graph_available:
            filename = "orders_hierarchical_graph_enhanced_cv.csv"
        else:
            filename = "orders_hierarchical_final_store_cv.csv"
    
    output_path = output_dir / filename
    export_submission_safe(submission, output_path)
    
    print(f"\n🎉 PIPELINE COMPLETE!")
    print(f"   Model: {args.model}")
    print(f"   Output: {output_path}")
    print("="*70)


if __name__ == "__main__":
    main()

