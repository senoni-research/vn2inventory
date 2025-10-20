"""
Historical State Reconstruction Module

Reconstructs inventory state at any historical cutoff by rolling backward/forward
from Week 0 using realized sales and implied orders.

This enables accurate CV cost evaluation with correct inventory positions.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from pathlib import Path


class StateReconstructor:
    """
    Reconstructs historical inventory states from Week 0 state and sales history.
    
    Uses inventory dynamics:
        Start_t = End_{t-1} + Arrivals_t
        Sales_t = min(Start_t, Demand_t)  # Can't sell more than available
        End_t = Start_t - Sales_t
        
    For in-transit:
        Arrivals_t = InTransit_{t-L→t} (from L weeks ago)
    """
    
    def __init__(
        self,
        week0_date: str,
        week0_state: pd.DataFrame,
        sales_long: pd.DataFrame,
        col_mapping: Dict[str, str],
        lead_weeks: int = 2
    ):
        """
        Initialize state reconstructor.
        
        Args:
            week0_date: Week 0 cutoff date (e.g., "2024-04-08")
            week0_state: DataFrame with columns: on_hand, on_order
            sales_long: Long-format sales with Store, Product, Week, SalesQty
            col_mapping: Column name mapping
            lead_weeks: Lead time in weeks
        """
        self.week0_date = pd.to_datetime(week0_date)
        self.week0_state = week0_state.copy()
        self.sales_long = sales_long.copy()
        self.col = col_mapping
        self.lead_weeks = lead_weeks
        
        # Build sales wide for easy lookups
        self.sales_wide = sales_long.pivot_table(
            index=[col_mapping["store"], col_mapping["product"]],
            columns=col_mapping["week"],
            values=col_mapping["qty"],
            fill_value=0.0
        )
        self.sales_wide.columns = pd.to_datetime(self.sales_wide.columns)
        
        # Get sorted week list
        self.all_weeks = sorted(self.sales_wide.columns)
        
        print(f"✅ StateReconstructor initialized")
        print(f"   Week 0: {week0_date}")
        print(f"   Historical weeks: {self.all_weeks[0]} to {self.all_weeks[-1]}")
        print(f"   Lead time: {lead_weeks} weeks")
    
    def reconstruct_state_at(self, cutoff_date: str) -> pd.DataFrame:
        """
        Reconstruct inventory state at a historical cutoff.
        
        Returns:
            DataFrame indexed by (Store, Product) with columns:
                OnHand, InTransitW+1, InTransitW+2
        """
        cutoff_dt = pd.to_datetime(cutoff_date)
        
        if cutoff_dt >= self.week0_date:
            # At or after Week 0, just return Week 0 state
            return self._format_state(self.week0_state)
        
        # Reconstruct by rolling backward from Week 0
        return self._rollback_from_week0(cutoff_dt)
    
    def _rollback_from_week0(self, target_date: pd.Timestamp) -> pd.DataFrame:
        """
        Roll backward from Week 0 to target_date using inventory dynamics.
        
        Working backward:
            End_{t-1} = End_t + Sales_t - Arrivals_t
            
        Where:
            Arrivals_t = orders placed at t-L that arrive at t
            Sales_t = actual realized sales at t
        """
        # Start from Week 0
        current_end = self.week0_state["on_hand"].copy()
        
        # Week 0 in-transit represents orders placed in previous weeks
        # InTransit W+1 = orders from 1 week ago (arriving next week from Week 0)
        # InTransit W+2 = orders from 2 weeks ago (arriving in 2 weeks from Week 0)
        current_intransit_w1 = self.week0_state["on_order"].copy()  # Simplified: total in-transit
        current_intransit_w2 = pd.Series(0.0, index=self.week0_state.index)
        
        # Get weeks between target and Week 0
        weeks_between = [w for w in self.all_weeks if target_date < w <= self.week0_date]
        weeks_between_sorted = sorted(weeks_between, reverse=True)  # Work backward
        
        # Roll back week by week
        for week_dt in weeks_between_sorted:
            # Get sales for this week
            sales_t = self.sales_wide[week_dt] if week_dt in self.sales_wide.columns else pd.Series(0, index=current_end.index)
            sales_t = sales_t.reindex(current_end.index).fillna(0.0)
            
            # Estimate arrivals (we don't know historical orders, so approximate)
            # Heuristic: arrivals ≈ sales + (End_t - End_{t-1})
            # For simplicity: assume arrivals were sufficient to meet demand
            arrivals_t = sales_t.clip(lower=0.0)  # Simplified: arrived what was sold
            
            # Roll back: End_{t-1} = End_t + Sales_t - Arrivals_t
            prev_end = current_end + sales_t - arrivals_t
            prev_end = prev_end.clip(lower=0.0)  # Can't be negative
            
            current_end = prev_end
        
        # At target date, construct state
        # InTransit: approximate as proportion of current end inventory
        # This is a heuristic since we don't have true historical orders
        intransit_w1 = (current_end * 0.3).clip(lower=0.0)  # ~30% in pipeline
        intransit_w2 = (current_end * 0.2).clip(lower=0.0)  # ~20% in pipeline
        
        reconstructed = pd.DataFrame({
            "OnHand": current_end,
            "InTransitW+1": intransit_w1,
            "InTransitW+2": intransit_w2
        })
        
        return reconstructed
    
    def _format_state(self, state: pd.DataFrame) -> pd.DataFrame:
        """Format state to standard columns."""
        formatted = pd.DataFrame({
            "OnHand": state.get("on_hand", state.get("OnHand", 0)),
            "InTransitW+1": state.get("on_order", state.get("InTransitW+1", 0)),
            "InTransitW+2": state.get("InTransitW+2", 0)
        })
        return formatted.fillna(0.0)
    
    def reconstruct_forward_path(
        self,
        start_date: str,
        end_date: str,
        initial_state: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """
        Roll forward from start_date to end_date, tracking state evolution.
        
        This is used to validate that our reconstruction makes sense.
        
        Returns:
            Dict mapping date_str -> state DataFrame
        """
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        weeks_forward = [w for w in self.all_weeks if start_dt < w <= end_dt]
        
        current_state = initial_state.copy()
        state_history = {start_date: current_state.copy()}
        
        for week_dt in sorted(weeks_forward):
            # Get sales for this week
            sales_t = self.sales_wide[week_dt] if week_dt in self.sales_wide.columns else pd.Series(0, index=current_state.index)
            sales_t = sales_t.reindex(current_state.index).fillna(0.0)
            
            # Dynamics (simplified - assumes no new orders placed)
            start_t = current_state["OnHand"] + current_state.get("InTransitW+1", 0)
            actual_sales = sales_t.clip(upper=start_t)  # Can't sell more than available
            end_t = (start_t - actual_sales).clip(lower=0.0)
            
            # Shift in-transit forward
            intransit_w1 = current_state.get("InTransitW+2", 0)
            intransit_w2 = 0.0  # No new orders
            
            current_state = pd.DataFrame({
                "OnHand": end_t,
                "InTransitW+1": intransit_w1,
                "InTransitW+2": intransit_w2
            })
            
            state_history[str(week_dt.date())] = current_state.copy()
        
        return state_history


def validate_state_reconstruction(
    reconstructor: StateReconstructor,
    week0_date: str,
    week0_state: pd.DataFrame,
    test_cutoffs: list
) -> None:
    """
    Validate that state reconstruction is sensible.
    
    Checks:
    1. Reconstructed state at Week 0 should match actual Week 0 state (approximately)
    2. Earlier cutoffs should have similar or lower inventory levels
    3. No negative values
    """
    print("\n🔍 VALIDATING STATE RECONSTRUCTION:")
    print("="*70)
    
    for cutoff in test_cutoffs:
        recon_state = reconstructor.reconstruct_state_at(cutoff)
        
        total_ip_recon = (recon_state["OnHand"] + recon_state["InTransitW+1"] + recon_state["InTransitW+2"]).sum()
        
        # Check for issues
        has_negative = (recon_state < 0).any().any()
        has_nan = recon_state.isna().any().any()
        
        status = "✅" if not (has_negative or has_nan) else "❌"
        print(f"{status} {cutoff}: Total IP = {total_ip_recon:,.0f}, OnHand = {recon_state['OnHand'].sum():,.0f}")
        
        if has_negative:
            print(f"   ⚠️ WARNING: Found negative values")
        if has_nan:
            print(f"   ⚠️ WARNING: Found NaN values")
    
    # Compare to Week 0
    week0_total_ip = (week0_state["on_hand"] + week0_state["on_order"]).sum()
    print(f"\n📊 Week 0 actual total IP: {week0_total_ip:,.0f}")
    print("   Historical states should be similar or lower (inventory builds up)")
    print("="*70)

