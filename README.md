# Inventory planning starter

**Senoni Research** CLI for weekly store–product orders with a simple base-stock policy (2-week lead time by default).

Bring your own sales, on-hand and in-transit CSVs. This repository does not include competition data.

Companion repos:

- [senoni-research/relational-graph](https://github.com/senoni-research/relational-graph) — temporal graph scorer and gated order policies
- [senoni-research/timesfm](https://github.com/senoni-research/timesfm) — TimesFM 2.5 quantile notes for the same setting

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Inputs

Three CSVs:

- **Index** — row order for the output (`Store`, `Product`, or as configured)
- **Sales history** — weekly quantities; a week/date column is optional
- **Current state** — on-hand and in-transit per `(Store, Product)`

See `config.example.yml` for default column names. Any column can be overridden on the CLI.

## Run

```bash
python -m vn2inventory order \
  --sales path/to/sales_history.csv \
  --current path/to/current_state.csv \
  --index path/to/index.csv \
  --out orders.csv \
  --config config.example.yml
```

Column overrides:

```bash
python -m vn2inventory order \
  --sales HISTORY.csv --current STATE.csv --index INDEX.csv --out orders.csv \
  --store-col Store --product-col Product --sales-qty-col Qty --sales-date-col Week \
  --on-hand-col OnHand --in-transit-cols InTransit_W1,InTransit_W2
```

If you have a local VN2-shaped extract (not in git), a typical mapping is:

```bash
python -m vn2inventory order \
  --sales artifacts/order1/sales_long.csv \
  --current "data/Week 0 - 2024-04-08 - Initial State.csv" \
  --index "data/Week 0 - Submission Template.csv" \
  --out orders.csv \
  --store-col Store --product-col Product \
  --sales-qty-col SalesQty --sales-date-col Week \
  --on-hand-col "End Inventory" \
  --in-transit-cols "In Transit W+1,In Transit W+2"
```

`data/`, `artifacts/` and `submissions/` are gitignored.

## Policy

- Base-stock for protection period `P = lead_time + review_period`
- `S = mean_demand * P + z * std_demand * sqrt(P)`
- `z` comes from a newsvendor-like critical ratio (shortage vs holding)

Output is a non-negative integer `order_qty` per index row. Lost sales are lost; holding applies to end-of-week on-hand only.

## License

MIT. See [LICENSE](LICENSE).
