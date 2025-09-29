VN2 Inventory Planning Starter

Quickstart to compute weekly orders using a simple base-stock policy with 2-week lead time.

Setup

1. Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Inputs

You need three CSVs:
- Index CSV: row order to match the platform (columns: `Store`, `Product` or as configured).
- Historical Sales CSV: weekly sales history. Columns include index columns and a quantity column; a week/date column is optional.
- Current State CSV: contains on-hand and in-transit quantities for each `(Store, Product)` pair.

See `config.example.yml` for column names. You can also override any column via CLI flags.

Run

```bash
python -m vn2inventory order \
  --sales path/to/sales_history.csv \
  --current path/to/current_state.csv \
  --index path/to/index.csv \
  --out orders.csv \
  --config config.example.yml
```

Common overrides (if your columns differ):

```bash
python -m vn2inventory order \
  --sales HISTORY.csv --current STATE.csv --index INDEX.csv --out orders.csv \
  --store-col Store --product-col Product --sales-qty-col Qty --sales-date-col Week \
  --on-hand-col OnHand --in-transit-cols InTransit_W1,InTransit_W2
```

Order #1 (competition data) mapping

For the initial dataset, use `End Inventory` as Week 1 on-hand, and pass the in-transit columns as shown:

```bash
python -m vn2inventory order \
  --sales artifacts/order1/sales_long.csv \
  --current "data/Week 0 - 2024-04-08 - Initial State.csv" \
  --index "data/Week 0 - Submission Template.csv" \
  --out submissions/orders_round1_cli.csv \
  --store-col Store --product-col Product \
  --sales-qty-col SalesQty --sales-date-col Week \
  --on-hand-col "End Inventory" \
  --in-transit-cols "In Transit W+1,In Transit W+2"
```

Policy

- Base-stock for protection period `P = lead_time + review_period`.
- `S = mean_demand * P + z * std_demand * sqrt(P)`.
- `z` is chosen from a newsvendor-like critical ratio using shortage vs. effective holding cost during the cycle.

Output

- Produces `orders.csv` with the same `(Store, Product)` index and a single column (default `order_qty`). All quantities are non-negative integers.

Notes

- No backorders are modeled; lost sales are lost.
- Holding cost is applied to end-of-week on-hand stock; goods in transit have no holding cost.
