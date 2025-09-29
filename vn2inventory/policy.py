from __future__ import annotations

import math
import numpy as np
import pandas as pd


def _inv_normal_cdf(p: float) -> float:
    """Approximate inverse CDF for standard normal (Acklam's method)."""
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    # Coefficients in rational approximations
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    plow = 0.02425
    phigh = 1 - plow

    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )
    if phigh < p:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )

    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
    ) / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1
    )


def compute_base_stock_levels(
    demand_stats: pd.DataFrame,
    lead_time_weeks: int,
    review_period_weeks: int,
    shortage_cost_per_unit: float,
    holding_cost_per_unit_per_week: float,
    min_service_level: float | None = None,
) -> pd.Series:
    """
    Compute base stock S for each (store, product).

    demand_stats: index (store, product) with columns mean_demand, std_demand
    Returns pd.Series named 'base_stock'
    """
    required = {"mean_demand", "std_demand"}
    if not required.issubset(set(demand_stats.columns)):
        raise ValueError("demand_stats must have mean_demand and std_demand")

    protection_weeks = lead_time_weeks + review_period_weeks

    # Effective overage cost approximates holding during the cycle: average age ~ P/2
    co_effective = holding_cost_per_unit_per_week * max(protection_weeks, 0) / 2.0
    cu = max(shortage_cost_per_unit, 1e-9)
    co = max(co_effective, 1e-12)
    service_level = cu / (cu + co)
    if min_service_level is not None:
        service_level = max(service_level, min_service_level)
    service_level = min(max(service_level, 1e-6), 1 - 1e-6)

    z = _inv_normal_cdf(service_level)

    mean_p = demand_stats["mean_demand"].clip(lower=0.0) * protection_weeks
    std_p = demand_stats["std_demand"].clip(lower=0.0) * math.sqrt(protection_weeks)
    safety = z * std_p
    base_stock = (mean_p + safety).clip(lower=0.0)
    base_stock.name = "base_stock"
    return base_stock


def compute_orders(
    index_df: pd.DataFrame,
    demand_stats: pd.DataFrame,
    current_state: pd.DataFrame,
    lead_time_weeks: int = 2,
    review_period_weeks: int = 1,
    shortage_cost_per_unit: float = 1.0,
    holding_cost_per_unit_per_week: float = 0.2,
    min_service_level: float | None = None,
    max_order_per_item: float | None = None,
) -> pd.Series:
    """Compute order quantities using a base-stock policy.

    Inventory position = on_hand + on_order.
    Order = max(0, round(base_stock - inventory_position)).
    Returns a Series aligned to index_df index named 'order_qty'.
    """
    base_stock = compute_base_stock_levels(
        demand_stats,
        lead_time_weeks=lead_time_weeks,
        review_period_weeks=review_period_weeks,
        shortage_cost_per_unit=shortage_cost_per_unit,
        holding_cost_per_unit_per_week=holding_cost_per_unit_per_week,
        min_service_level=min_service_level,
    )

    # Align
    base_stock = base_stock.reindex(index_df.index).fillna(0.0)
    state = current_state.reindex(index_df.index).fillna({"on_hand": 0.0, "on_order": 0.0})
    inventory_position = state["on_hand"].clip(lower=0.0) + state["on_order"].clip(lower=0.0)

    raw = (base_stock - inventory_position).clip(lower=0.0)
    if max_order_per_item is not None:
        raw = raw.clip(upper=max_order_per_item)

    orders = np.rint(raw).astype(int)
    orders.name = "order_qty"
    return orders


def compute_orders_for_week(
    sales_history: pd.DataFrame,
    current_state: pd.DataFrame,
    index_df: pd.DataFrame,
    lead_time_weeks: int = 2,
    review_period_weeks: int = 1,
    shortage_cost_per_unit: float = 1.0,
    holding_cost_per_unit_per_week: float = 0.2,
    min_service_level: float | None = None,
    max_order_per_item: float | None = None,
) -> pd.Series:
    """
    Convenience wrapper that takes raw sales history and state, computes demand stats,
    and then orders.
    """
    # Ensure proper columns exist
    if not {"mean_demand", "std_demand"}.issubset(set(sales_history.columns)):
        raise ValueError(
            "sales_history must be aggregated to (mean_demand, std_demand) per (store, product)."
        )
    return compute_orders(
        index_df=index_df,
        demand_stats=sales_history,
        current_state=current_state,
        lead_time_weeks=lead_time_weeks,
        review_period_weeks=review_period_weeks,
        shortage_cost_per_unit=shortage_cost_per_unit,
        holding_cost_per_unit_per_week=holding_cost_per_unit_per_week,
        min_service_level=min_service_level,
        max_order_per_item=max_order_per_item,
    )
