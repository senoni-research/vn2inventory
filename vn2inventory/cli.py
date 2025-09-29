from __future__ import annotations

import sys
import json
import click
import yaml
import pandas as pd

from .data_io import load_sales_history, load_current_state, load_index
from .policy import compute_orders


@click.group()
def order_cli() -> None:
    """VN2 Inventory - CLI tools"""


@order_cli.command("order")
@click.option("--sales", "sales_csv", type=click.Path(exists=True, dir_okay=False), required=True, help="Historical weekly sales CSV path")
@click.option("--current", "current_csv", type=click.Path(exists=True, dir_okay=False), required=True, help="Current state CSV path")
@click.option("--index", "index_csv", type=click.Path(exists=True, dir_okay=False), required=True, help="Index CSV in the required row order")
@click.option("--out", "out_csv", type=click.Path(dir_okay=False), required=True, help="Output submission CSV path")
@click.option("--config", "config_yml", type=click.Path(exists=True, dir_okay=False), help="YAML config for column names and policy params")
# Column mappings
@click.option("--store-col", type=str)
@click.option("--product-col", type=str)
@click.option("--sales-qty-col", type=str)
@click.option("--sales-date-col", type=str)
@click.option("--on-hand-col", type=str)
@click.option("--in-transit-cols", type=str, help="Comma-separated list of in-transit columns")
# Policy overrides
@click.option("--lead", "lead_time_weeks", type=int)
@click.option("--review", "review_period_weeks", type=int)
@click.option("--shortage-cost", "shortage_cost", type=float)
@click.option("--holding-cost", "holding_cost", type=float)
@click.option("--min-service", "min_service_level", type=float)
@click.option("--max-order", "max_order", type=float)
@click.option("--submission-col", "submission_col", type=str)
def order_command(
    sales_csv: str,
    current_csv: str,
    index_csv: str,
    out_csv: str,
    config_yml: str | None,
    store_col: str | None,
    product_col: str | None,
    sales_qty_col: str | None,
    sales_date_col: str | None,
    on_hand_col: str | None,
    in_transit_cols: str | None,
    lead_time_weeks: int | None,
    review_period_weeks: int | None,
    shortage_cost: float | None,
    holding_cost: float | None,
    min_service_level: float | None,
    max_order: float | None,
    submission_col: str | None,
) -> None:
    """Generate orders.csv following the given index ordering."""

    cfg = {}
    if config_yml:
        with open(config_yml, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    def cfg_get(*keys, default=None):
        node = cfg
        for k in keys:
            if node is None:
                return default
            node = node.get(k)
        return default if node is None else node

    # Resolve column names
    store_col = store_col or cfg_get("columns", "store_id", default="store")
    product_col = product_col or cfg_get("columns", "product_id", default="product")
    sales_qty_col = sales_qty_col or cfg_get("columns", "sales_qty", default="qty")
    sales_date_col = sales_date_col or cfg_get("columns", "sales_date", default=None)
    on_hand_col = on_hand_col or cfg_get("columns", "on_hand", default="on_hand")

    in_transit_list = None
    if in_transit_cols:
        in_transit_list = [c.strip() for c in in_transit_cols.split(",") if c.strip()]
    else:
        in_transit_list = cfg_get("columns", "in_transit_cols", default=None)

    # Policy params
    lead_time_weeks = lead_time_weeks or int(cfg_get("policy", "lead_time_weeks", default=2))
    review_period_weeks = review_period_weeks or int(cfg_get("policy", "review_period_weeks", default=1))
    shortage_cost = shortage_cost or float(cfg_get("policy", "shortage_cost_per_unit", default=1.0))
    holding_cost = holding_cost or float(cfg_get("policy", "holding_cost_per_unit_per_week", default=0.2))
    min_service_level = min_service_level or cfg_get("policy", "min_service_level", default=None)
    max_order = max_order or cfg_get("policy", "max_order_per_item", default=None)
    submission_col = submission_col or cfg_get("submission", "column_name", default="order_qty")

    # Load inputs
    sales_stats = load_sales_history(
        sales_csv_path=sales_csv,
        store_col=store_col,
        product_col=product_col,
        sales_qty_col=sales_qty_col,
        date_col=sales_date_col,
    )
    current_state = load_current_state(
        current_state_csv_path=current_csv,
        store_col=store_col,
        product_col=product_col,
        on_hand_col=on_hand_col,
        in_transit_cols=in_transit_list,
    )
    index_df = load_index(index_csv, store_col=store_col, product_col=product_col)

    # Compute orders
    orders = compute_orders(
        index_df=index_df,
        demand_stats=sales_stats,
        current_state=current_state,
        lead_time_weeks=lead_time_weeks,
        review_period_weeks=review_period_weeks,
        shortage_cost_per_unit=shortage_cost,
        holding_cost_per_unit_per_week=holding_cost,
        min_service_level=min_service_level,
        max_order_per_item=max_order,
    )

    # Emit submission
    submission = index_df.copy()
    submission[submission_col] = orders.values
    submission.to_csv(out_csv, index=True)

    total_units = int(orders.sum())
    click.echo(json.dumps({
        "items": int(len(orders)),
        "total_units": total_units,
        "lead_time_weeks": lead_time_weeks,
        "review_period_weeks": review_period_weeks,
        "submission": out_csv,
    }, indent=2))


if __name__ == "__main__":
    order_cli()
