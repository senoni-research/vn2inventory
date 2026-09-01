# Inventory planning

**Senoni Research** code for weekly store–product orders.

Two layers:

1. **Base-stock CLI** — mean/std demand, newsvendor `z`, order = `S −` inventory position.
2. **Hierarchical Bayes pipeline** — department GLM, SKU-level empirical-Bayes shrinkage on the log-mean scale, optional hurdle for intermittent SKUs, optional graph features from [relational-graph](https://github.com/senoni-research/relational-graph).

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

`data/`, `artifacts/` and `submissions/` are gitignored.

## Base-stock CLI

```bash
python -m vn2inventory order \
  --sales path/to/sales_history.csv \
  --current path/to/current_state.csv \
  --index path/to/index.csv \
  --out orders.csv \
  --config config.example.yml
```

- Protection period `P = lead_time + review_period`
- `S = mean_demand * P + z * std_demand * sqrt(P)`
- `z` from a newsvendor critical ratio (shortage vs holding)

Output is a non-negative integer `order_qty` per index row.

## Hierarchical Bayes

```bash
python scripts/run_hb_solution.py \
  --model baseline \
  --data-dir path/to/csvs \
  --output-dir path/to/out
```

Graph-enhanced (features from the companion repo, not in this tree):

```bash
python scripts/run_hb_solution.py \
  --model graph-enhanced \
  --features-599 path/to/orders_features_599.csv \
  --data-dir path/to/csvs \
  --output-dir path/to/out
```

SKU effects are shrunk on the **log-mean** scale. The within-SKU term is `Var(log residual) / n_weeks`. Mixing that with NB count variance (`μ + αμ²`) drives every weight to zero and forecasts the department average — that was a production bug and is fixed here.

The hurdle sampler takes an unconditional weekly mean. The positive NB component is `μ / (1 − p0)`, so the zero process is applied once.

See [scripts/README_HB_CLI.md](scripts/README_HB_CLI.md).

## License

MIT. See [LICENSE](LICENSE).
