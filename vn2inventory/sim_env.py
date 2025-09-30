from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd


@dataclass
class Costs:
    holding_per_unit: float = 0.2
    shortage_per_unit: float = 1.0


@dataclass
class SimState:
    # Indexed by (Store, Product)
    end_inventory: pd.Series
    in_transit_w1: pd.Series
    in_transit_w2: pd.Series
    cumulative_holding: float = 0.0
    cumulative_shortage: float = 0.0


class InventorySim:
    """
    Minimal simulator reproducing the organizer's weekly state transition.

    API (per step):
      - input: orders Series (>=0 ints) indexed like sales
      - demand: from sales DataFrame column for current week
      - transition:
          StartInv = prev.EndInv + prev.InTransit_W+1
          Sales    = min(StartInv, Demand)
          Missed   = Demand - Sales
          EndInv   = StartInv - Sales
          W+1      = prev.W+2
          W+2      = Orders
      - costs: holding on EndInv; shortage on Missed
    """

    def __init__(
        self,
        sales_wide: pd.DataFrame,
        initial_state: pd.DataFrame,
        costs: Costs = Costs(),
        index_cols: Tuple[str, str] = ("Store", "Product"),
        demand_dates: Optional[list[str]] = None,
    ) -> None:
        self.index_cols = index_cols
        self.costs = costs

        self.sales = sales_wide.set_index(list(index_cols))
        self.dates = demand_dates or list(self.sales.columns)
        if len(self.dates) == 0:
            raise ValueError("No demand dates found in sales_wide")

        st = initial_state.set_index(list(index_cols))
        self.state = SimState(
            end_inventory=st["End Inventory"].astype(float).copy(),
            in_transit_w1=st.get("In Transit W+1", pd.Series(0.0, index=st.index)).astype(float).copy(),
            in_transit_w2=st.get("In Transit W+2", pd.Series(0.0, index=st.index)).astype(float).copy(),
            cumulative_holding=float(st.get("Cumulative Holding Cost", pd.Series(0.0)).sum()),
            cumulative_shortage=float(st.get("Cumulative Shortage Cost", pd.Series(0.0)).sum()),
        )
        self.t = 0
        self.index = self.sales.index

    def current_demand(self) -> pd.Series:
        return self.sales[self.dates[self.t]].astype(float)

    def step(self, orders: pd.Series) -> Dict[str, float]:
        orders = orders.reindex(self.index).fillna(0).astype(float).clip(lower=0.0)
        demand = self.current_demand()

        start_inventory = self.state.end_inventory + self.state.in_transit_w1
        sales = np.minimum(start_inventory, demand)
        missed = demand - sales
        end_inventory = start_inventory - sales

        new_w1 = self.state.in_transit_w2
        new_w2 = orders

        holding_cost = float((end_inventory * self.costs.holding_per_unit).sum())
        shortage_cost = float((missed * self.costs.shortage_per_unit).sum())

        self.state = SimState(
            end_inventory=end_inventory,
            in_transit_w1=new_w1,
            in_transit_w2=new_w2,
            cumulative_holding=self.state.cumulative_holding + holding_cost,
            cumulative_shortage=self.state.cumulative_shortage + shortage_cost,
        )
        self.t += 1

        done = self.t >= len(self.dates)
        return {
            "holding_cost": holding_cost,
            "shortage_cost": shortage_cost,
            "round_cost": holding_cost + shortage_cost,
            "cumulative_cost": self.state.cumulative_holding + self.state.cumulative_shortage,
            "t": self.t,
            "done": float(done),
        }

    def inventory_position(self) -> pd.Series:
        return self.state.end_inventory + self.state.in_transit_w1 + self.state.in_transit_w2

    def reset_to(self, initial_state: pd.DataFrame) -> None:
        st = initial_state.set_index(list(self.index_cols))
        self.state.end_inventory = st["End Inventory"].astype(float).copy()
        self.state.in_transit_w1 = st.get("In Transit W+1", pd.Series(0.0, index=st.index)).astype(float).copy()
        self.state.in_transit_w2 = st.get("In Transit W+2", pd.Series(0.0, index=st.index)).astype(float).copy()
        self.state.cumulative_holding = float(st.get("Cumulative Holding Cost", pd.Series(0.0)).sum())
        self.state.cumulative_shortage = float(st.get("Cumulative Shortage Cost", pd.Series(0.0)).sum())
        self.t = 0
