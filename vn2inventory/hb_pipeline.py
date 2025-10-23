"""
Hierarchical Bayes Pipeline

Main pipeline orchestrator that can run with or without graph features.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

# Local imports
from .hb_core import (
    fit_nb_glm_with_fallback,
    apply_eb_shrinkage,
    estimate_segment_dispersion,
    build_design_matrix,
    identify_intermittent_skus
)
from .friend_recommendations import (
    compute_p0_available_only,
    compute_abc_xyz_segmentation,
    q_target_for_segment,
    sample_D3_nb_canonical,
    sample_D3_hurdle_canonical,
    compute_order_with_guardrails,
    validate_submission,
    export_submission_safe
)
from .graph_integration import (
    load_graph_features,
    merge_graph_features,
    build_graph_enriched_formulas,
    enhance_params_with_graph
)
from .cv_optimizer import PolicyKnobs, validate_cv_cutoffs
from .data_io import load_index, load_current_state
from .state_reconstruction import StateReconstructor, validate_state_reconstruction


@dataclass
class HBConfig:
    """Configuration for HB pipeline."""
    # Paths
    data_dir: Path
    output_dir: Path
    graph_features_path: Optional[Path] = None
    
    # Column mapping
    col_store: str = "Store"
    col_product: str = "Product"
    col_week: str = "Week"
    col_qty: str = "SalesQty"
    col_onhand: str = "End Inventory"
    col_intransit: list = None
    
    # HB params
    tau2_grid: list = None
    alpha_cap: float = 10.0
    alpha_k: int = 8
    
    # CV params
    cv_horizon_weeks: int = 4
    n_cv_trials: int = 60
    cv_timeout: int = 300
    
    # Policy params
    protection_weeks: int = 3
    shortage_cost: float = 1.0
    holding_cost: float = 0.2
    mc_samples: int = 2000
    mc_seed: int = 42
    
    def __post_init__(self):
        if self.col_intransit is None:
            self.col_intransit = ["In Transit W+1", "In Transit W+2"]
        if self.tau2_grid is None:
            self.tau2_grid = list(np.geomspace(0.1, 5.0, 9))
    
    @property
    def col_mapping(self) -> Dict[str, str]:
        return {
            "store": self.col_store,
            "product": self.col_product,
            "week": self.col_week,
            "qty": self.col_qty,
            "on_hand": self.col_onhand,
            "in_transit": self.col_intransit,
        }
    
    @property
    def cost_consistent_alpha(self) -> float:
        """Newsvendor critical fractile."""
        return self.shortage_cost / (self.shortage_cost + self.holding_cost)


class HBPipeline:
    """
    Main Hierarchical Bayes pipeline.
    
    Can run with or without graph features.
    """
    
    def __init__(self, config: HBConfig):
        self.config = config
        self.col = config.col_mapping
        self.sku_grp = [config.col_store, config.col_product]
        
        # State
        self.sales_long = None
        self.sl = None
        self.sku_meta = None
        self.idx_df = None
        self.state_df = None
        self.graph_available = False
        self.fit_meta = {}
        self.state_reconstructor = None  # For historical CV states
        
    def load_data(self) -> None:
        """Load all required data files."""
        # Load sales_long
        sales_candidates = [
            self.config.data_dir / "Sales_long.csv",
            self.config.data_dir.parent / "artifacts" / "hierarchical" / "sales_long_clean.csv",
        ]
        
        sales_path = None
        for p in sales_candidates:
            if p.exists():
                sales_path = p
                break
        
        if sales_path is None:
            raise FileNotFoundError(f"Could not find sales_long CSV in {sales_candidates}")
        
        self.sales_long = pd.read_csv(sales_path)
        self.sales_long[self.col["qty"]] = pd.to_numeric(self.sales_long[self.col["qty"]], errors="coerce").fillna(0.0)
        self.sales_long[self.col["week"]] = pd.to_datetime(self.sales_long[self.col["week"]])
        
        # Group to weekly sums
        self.sales_long = (
            self.sales_long[[self.col["store"], self.col["product"], self.col["week"], self.col["qty"]]]
            .groupby([self.col["store"], self.col["product"], self.col["week"]], dropna=False)
            .sum()
            .reset_index()
        )
        
        # Load master
        master_path = self.config.data_dir / "Week 0 - Master.csv"
        master = pd.read_csv(master_path)
        
        # Load index and state
        idx_path = self.config.data_dir / "Week 0 - Submission Template.csv"
        state_path = self.config.data_dir / "Week 0 - 2024-04-08 - Initial State.csv"
        
        self.idx_df = load_index(str(idx_path), self.col["store"], self.col["product"])
        self.state_df = load_current_state(
            str(state_path),
            self.col["store"], self.col["product"], 
            self.col["on_hand"], self.col["in_transit"]
        )
        
        print(f"✅ Data loaded: {len(self.sales_long)} sales records, {len(self.idx_df)} SKUs")
        
        # Compute availability-aware p0
        p0_series = compute_p0_available_only(self.sales_long, self.config.data_dir, self.col)
        
        # Build design matrix with p0
        self.sl = build_design_matrix(self.sales_long, master, self.col, p0_series)
        
        # Compute ABC/XYZ segmentation
        self.sku_meta = compute_abc_xyz_segmentation(self.sl, self.col)
        
        # Initialize state reconstructor for accurate CV
        self.state_reconstructor = StateReconstructor(
            week0_date="2024-04-08",
            week0_state=self.state_df,
            sales_long=self.sales_long,
            col_mapping=self.col,
            lead_weeks=2
        )
        
        print("="*70)
        
    def load_graph_features_if_available(self) -> None:
        """Attempt to load graph features (optional)."""
        if self.config.graph_features_path is None:
            print("⚠️ Graph features path not specified; using baseline HB")
            return
        
        graph_feat = load_graph_features(
            self.config.graph_features_path,
            self.col,
            verbose=True
        )
        
        if graph_feat is not None:
            self.sl = merge_graph_features(self.sl, graph_feat, self.col)
            self.graph_available = True
        
    def fit_model(self) -> None:
        """Fit hierarchical GLM with optional graph features."""
        # Build hierarchy terms
        store_terms = []
        for c in ["Region", "Cluster", "District"]:
            if c in self.sl.columns and self.sl[c].notna().any():
                store_terms.append(f"C({c})")
        
        hier_terms = ["C(Department)", "C(ProductGroup)"] + store_terms
        
        # Build formulas
        formula_candidates = build_graph_enriched_formulas(
            hier_terms,
            include_graph=self.graph_available and "mu_weekly_graph" in self.sl.columns
        )
        
        # Fit GLM
        res, nb_fam, fit_meta = fit_nb_glm_with_fallback(
            self.sl,
            formula_candidates,
            verbose=True
        )
        
        self.fit_meta = fit_meta
        
        # Apply EB shrinkage (freeze tau² to stable default)
        self.sl, tau2_used, alpha, log_resid, nb_var = apply_eb_shrinkage(
            self.sl, res, nb_fam, self.sku_grp, tau2=0.1
        )
        
        # Per-segment dispersion
        self.sl = estimate_segment_dispersion(
            self.sl,
            self.sku_grp,
            segment_col="Department",
            alpha_cap=self.config.alpha_cap,
            alpha_k=self.config.alpha_k
        )
        
        print("="*70)
        
    def compute_orders_for_cutoff(
        self,
        cutoff_week: str,
        knobs: PolicyKnobs
    ) -> pd.Series:
        """
        Compute orders for a given cutoff with segmented targets and guardrails.
        """
        # Get template
        template = self.idx_df.reset_index().set_index([self.col["store"], self.col["product"]])
        
        # Fit params up to cutoff (simplified for CV)
        params = self._fit_params_up_to(cutoff_week).reindex(template.index)
        
        # Get state
        state = self._get_state_at(cutoff_week).reindex(template.index)
        
        # Join metadata
        meta = self.sku_meta.set_index([self.col["store"], self.col["product"]]).reindex(template.index)
        
        # Inventory position
        inv_pos = (
            state["OnHand"].fillna(0).astype(float) +
            state["InTransitW+1"].fillna(0).astype(float) +
            state.get("InTransitW+2", pd.Series(0, index=state.index)).fillna(0).astype(float)
        ).values
        
        # Extract parameters
        mu = params["mu"].fillna(0).values
        phi_raw = params["alpha"].fillna(1.0).values
        p0 = params["p0"].fillna(0).values
        intermittent = params["intermittent"].fillna(False).values
        
        # Apply calibration
        # Treat calib_sigma as a multiplicative dispersion calibrator (phi_cal)
        phi_eff = phi_raw * float(knobs.calib_sigma)
        
        # Compute orders with SEGMENTED TARGETS
        orders = np.zeros(len(template), dtype=int)
        
        for i in range(len(template)):
            if mu[i] <= 0:
                orders[i] = 0
                continue
            
            # Sample D3 using CANONICAL samplers
            seed_i = self.config.mc_seed + i
            # Adaptive MC samples: more for A or intermittent, fewer otherwise
            abc_i = str(meta.ABC.iloc[i]) if "ABC" in meta.columns else "B"
            xyz_i = str(meta.XYZ.iloc[i]) if "XYZ" in meta.columns else "Y"
            K_i = 3000 if (abc_i == "A" or bool(intermittent[i])) else 1500
            if intermittent[i]:
                d3_samples = sample_D3_hurdle_canonical(
                    mu[i], phi_eff[i], p0[i], 
                    self.config.protection_weeks, K_i, seed_i
                )
            else:
                d3_samples = sample_D3_nb_canonical(
                    mu[i], phi_eff[i], 
                    self.config.protection_weeks, K_i, seed_i
                )
            
            # SEGMENTED target quantile
            q_seg = q_target_for_segment(abc_i, xyz_i)
            q_target_i = float(np.clip(q_seg * knobs.k_safety, 0.78, 0.92))
            
            # Raw need
            need = int(np.quantile(d3_samples, q_target_i)) - int(inv_pos[i])
            
            # Apply guardrails
            hist_max = meta.hist_max_weekly.iloc[i] if "hist_max_weekly" in meta.columns else 0
            p95 = meta.p95_weekly.iloc[i] if "p95_weekly" in meta.columns else 0
            
            mu13_recent = meta.mu13_recent.iloc[i] if "mu13_recent" in meta.columns else np.nan
            order = compute_order_with_guardrails(
                need=need,
                inv_pos=int(inv_pos[i]),
                d3_samples=d3_samples,
                hist_max=hist_max,
                p95=p95,
                abc=abc_i,
                xyz=xyz_i,
                mu=mu[i],
                mu13_recent=float(mu13_recent) if pd.notna(mu13_recent) else None,
                protection_weeks=self.config.protection_weeks
            )
            
            orders[i] = order
        
        return pd.Series(orders, index=template.index, name="0")
    
    def _fit_params_up_to(self, cutoff_week: str) -> pd.DataFrame:
        """Simplified EB refit on historical subset for CV."""
        # Exclude the cutoff week itself to avoid leakage
        sl_subset = self.sl[self.sl["week_monday"] < pd.to_datetime(cutoff_week)].copy()
        
        mu_hist = sl_subset.groupby(self.sku_grp)["mu_eb"].mean()
        alpha_hist = sl_subset.groupby(self.sku_grp)["alpha_eff"].mean()
        
        # Intermittency stats (availability-aware p0 from sl)
        y_stats = sl_subset.groupby(self.sku_grp).agg(mu_check=("y", "mean"), p0=("p0", "mean"))
        
        params = pd.DataFrame({
            "mu": mu_hist,
            "alpha": alpha_hist.fillna(1.0),
            "p0": y_stats["p0"].fillna(0.0),
            "intermittent": (y_stats["p0"] >= 0.5) & (y_stats["mu_check"] < 2.0)
        })
        
        # Enhance with graph if available
        if self.graph_available:
            params = enhance_params_with_graph(params, self.sl, self.sku_grp)
        
        return params.fillna({"mu": 0, "alpha": 1.0, "p0": 0, "intermittent": False})
    
    def _get_state_at(self, cutoff_week: str) -> pd.DataFrame:
        """
        Get inventory state at cutoff.
        
        Uses historical state reconstruction if available, otherwise falls back
        to Week 0 state approximation.
        """
        cutoff_dt = pd.to_datetime(cutoff_week)
        week0_dt = pd.to_datetime("2024-04-08")

        # If we have a reconstructor, use it for dates before Week 0
        if self.state_reconstructor is not None and cutoff_dt < week0_dt:
            return self.state_reconstructor.reconstruct_state_at(cutoff_week)

        # For cutoffs at/after Week 0, prefer an explicit Initial State file if present
        try:
            weeks_ahead = int(max(0, (cutoff_dt - week0_dt).days // 7))
        except Exception:
            weeks_ahead = 0
        if weeks_ahead >= 1:
            # Try to load e.g., "Week 1 - 2024-04-15 - Initial State.csv"
            fname = f"Week {weeks_ahead} - {cutoff_week} - Initial State.csv"
            path = self.config.data_dir / fname
            if path.exists():
                st = pd.read_csv(path)
                st = st.rename(columns={
                    self.config.col_onhand: "OnHand",
                })
                # Map transit columns if available
                w1 = "In Transit W+1" if "In Transit W+1" in st.columns else None
                w2 = "In Transit W+2" if "In Transit W+2" in st.columns else None
                st_out = pd.DataFrame({
                    "OnHand": pd.to_numeric(st.get("OnHand", st.get(self.config.col_onhand, 0)), errors="coerce").fillna(0.0),
                    "InTransitW+1": pd.to_numeric(st.get(w1, 0), errors="coerce").fillna(0.0),
                    "InTransitW+2": pd.to_numeric(st.get(w2, 0), errors="coerce").fillna(0.0),
                })
                st_out.index = st[[self.config.col_store, self.config.col_product]].apply(tuple, axis=1)
                # Align to template index if available
                try:
                    idx = self.idx_df.index
                    st_out = st_out.reindex(idx).fillna(0.0)
                except Exception:
                    pass
                return st_out

        # Default: Week 0 state approximation
        return self.state_df.rename(columns={
            "on_hand": "OnHand",
            "on_order": "InTransitW+1"
        }).assign(**{"InTransitW+2": 0})
    
    def generate_submission(
        self,
        knobs: PolicyKnobs,
        cutoff_week: str = "2024-04-08",
        apply_hb_cap: bool = False,
        baseline_orders: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Generate final submission using fitted model and policy knobs.
        
        Args:
            knobs: Policy knobs from CV or defaults
            cutoff_week: Cutoff date for submission
            apply_hb_cap: If True and graph_available, cap graph orders vs baseline HB
            baseline_orders: Baseline HB orders for capping (required if apply_hb_cap=True)
        """
        orders = self.compute_orders_for_cutoff(cutoff_week, knobs)
        
        # Build submission DataFrame
        template = self.idx_df.reset_index()
        submission = template.copy()
        submission["0"] = orders.reindex(
            template.set_index([self.col["store"], self.col["product"]]).index
        ).values
        
        # Optional: apply targeted cap vs baseline HB (graph runs only)
        if apply_hb_cap and self.graph_available and baseline_orders is not None:
            print("\n🛡️ APPLYING TARGETED CAP VS BASELINE HB")
            print("="*70)
            
            hb_aligned = baseline_orders.set_index([self.col["store"], self.col["product"]])["0"]
            hb_aligned = hb_aligned.reindex(submission.set_index([self.col["store"], self.col["product"]]).index).fillna(0)
            
            graph_orders = submission.set_index([self.col["store"], self.col["product"]])["0"]
            
            # Flag extreme outliers: graph > 4×HB AND graph > 220
            flag = (graph_orders > (hb_aligned * 4)) & (graph_orders > 220)
            
            # Cap: max(hb×3.5 + 65, 270)
            cap_upper = np.maximum(hb_aligned * 3.5 + 65, 270)
            
            # Apply cap only to flagged rows
            capped_orders = np.where(flag, np.minimum(graph_orders, cap_upper), graph_orders)
            capped_orders = np.rint(capped_orders).astype(int)
            
            original_max = int(graph_orders.max())
            original_sum = int(graph_orders.sum())
            capped_max = int(capped_orders.max())
            capped_sum = int(capped_orders.sum())
            
            submission["0"] = capped_orders
            
            print(f"   Flagged outliers: {int(flag.sum())} SKUs")
            print(f"   Original max: {original_max} → Capped max: {capped_max}")
            print(f"   Original sum: {original_sum} → Capped sum: {capped_sum}")
            print(f"   Volume retained: {capped_sum/original_sum*100:.1f}%")
            print("="*70)
        
        print(f"\n✅ ORDERS GENERATED")
        print(f"   Total units: {submission['0'].sum():,}")
        print(f"   Mean: {submission['0'].mean():.2f}")
        print(f"   Zeros: {(submission['0'] == 0).sum()}")
        print(f"   Max: {submission['0'].max()}")
        print("=" * 70)
        
        return submission

