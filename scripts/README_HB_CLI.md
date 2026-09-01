# Hierarchical Bayes CLI - Complete Reference

## 🚀 Quick Start (Production)

```bash
python scripts/run_hb_solution.py --model baseline --data-dir path/to/csvs --output-dir path/to/out
```

Graph-enhanced needs a features file from [relational-graph](https://github.com/senoni-research/relational-graph):

```bash
python scripts/run_hb_solution.py \
  --model graph-enhanced \
  --features-599 path/to/orders_features_599.csv \
  --data-dir path/to/csvs \
  --output-dir path/to/out
```

---

## 📋 All Command Options

```bash
python scripts/run_hb_solution.py \
  --model {baseline|graph-enhanced} \
  [--apply-cap] \
  [--skip-cv] \
  [--data-dir PATH] \
  [--output-dir PATH]
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | choice | `baseline` | Model variant (`baseline` or `graph-enhanced`) |
| `--apply-cap` | flag | False | Apply targeted cap vs baseline (graph only) ✨ |
| `--skip-cv` | flag | False | Use default knobs (faster, less optimal) |
| `--data-dir` | path | `../data` | Data directory |
| `--output-dir` | path | `../submissions` | Output directory |

---

## 🏗️ Pipeline Steps

### Step 1: Load Data
- Sales history (`Sales_long.csv`)
- Master hierarchy (`Week 0 - Master.csv`)
- Availability data (`Week 0 - In Stock.csv`)
- Submission template
- Initial state (`Week 0 - 2024-04-08 - Initial State.csv`)
- **Computes availability-aware p0** ✅
- **Computes ABC/XYZ segmentation** ✅
- **Initializes state reconstructor** ✨

### Step 2: Load Graph Features (if --model graph-enhanced)
- Loads `orders_features_599_idemb_asof.csv` ✨
- Prefers weekly moments (`mu_w1-w3`, `sigma_w1-w3`)
- Falls back to horizon moments if needed
- Merges into design matrix

### Step 3: Fit Hierarchical Model
- GLM with fallbacks (handles collinearity)
- EB shrinkage (log-scale, tau²=0.1)
- Per-segment dispersion estimation
- **Graph covariates** (if graph-enhanced): `mu_weekly_graph`, `p_mean`

### Step 4: CV Optimization (unless --skip-cv)
- **Validates cutoffs** (Feb 05-26, all ≥6 weeks future) ✨
- **Reconstructs historical states** per fold ✨
- Optuna optimization (60 trials, TPE sampler)
- Finds best `k_safety`, `calib_sigma`

### Step 5: Generate Orders
- Canonical NB2 samplers (correct parameterization)
- **Segmented targets** (A/X: 0.89, C/Z: 0.79)
- Multiple guardrails (statistical, empirical, velocity)
- **Targeted cap** (if --apply-cap): flags graph > 4×HB AND >220 ✨

### Step 6: Export with QA
- 6 automated checks
- Integer coercion
- Proper CSV format
- Saves to output directory

---

## 📊 Expected Output

```
======================================================================
🚀 RUNNING HIERARCHICAL BAYES: GRAPH-ENHANCED
======================================================================
Data dir: /Users/senoni/noni/vn2inventory/data
Output dir: submissions/final
Graph features: Yes
CV: Enabled
======================================================================

📂 STEP 1: Loading data...
✅ P0 computed on available weeks only
✅ StateReconstructor initialized
   Week 0: 2024-04-08
   Historical weeks: 2021-04-12 to 2024-04-08

🔍 VALIDATING STATE RECONSTRUCTION:
✅ 2024-02-05: Total IP = 3,338
✅ 2024-02-12: Total IP = 3,338
...

📊 STEP 2: Loading graph features...
✅ Graph features loaded: 599 SKUs from orders_features_599_idemb_asof.csv

📊 GRAPH FEATURES MERGED:
   Rows: 94043
   p_t3: [0.000, 1.000]
   mu_weekly_graph: [0.00, 85.81]
   corr(y, mu_weekly_graph): 0.634

🔬 STEP 3: Fitting hierarchical model...
✅ Fitted: y ~ ... + mu_weekly_graph + p_mean
✅ EB shrinkage applied (tau²=0.100000)
   RMSE: global=8.42, EB=8.42
✅ Segment dispersion: 26 segments, median alpha=0.818

🎯 STEP 4: Running CV optimization...
✅ CV cutoffs validated (all ≥6 weeks future)
[Optuna progress bar...]

======================================================================
CV TUNING COMPLETE
Best params: {'k_safety': 0.8730, 'calib_sigma': 0.7047}
Best value (median CV cost): 3204.5
======================================================================

📦 STEP 5: Generating final submission...

🛡️ APPLYING TARGETED CAP VS BASELINE HB
   Flagged outliers: 1 SKUs
   Original max: 380 → Capped max: 270
   Original sum: 2862 → Capped sum: 2752
   Volume retained: 96.2%

✅ ORDERS GENERATED
   Total units: 2,752
   Mean: 4.59
   Zeros: 311
   Max: 270

💾 STEP 6: Exporting submission...
[All QA checks ✅]

✅ SUBMISSION VALIDATED & EXPORTED
   Path: submissions/final/orders_hierarchical_graph_enhanced_cv.csv
======================================================================
```

---

## 🎯 Command Examples

### Production (Recommended):
```bash
# Graph with cap
python scripts/run_hb_solution.py --model graph-enhanced --output-dir submissions/final --apply-cap
```

### Baseline (Fallback):
```bash
# If graph has issues
python scripts/run_hb_solution.py --model baseline --output-dir submissions/baseline_final
```

### Quick Test:
```bash
# Fast iteration (2 min, uses defaults)
python scripts/run_hb_solution.py --model graph-enhanced --skip-cv --apply-cap
```

---

## 🏆 Key Improvements Over Notebooks

| Feature | Notebooks | Final Scripts |
|---------|-----------|---------------|
| **State reconstruction** | ❌ Week 0 approximation | ✅ Historical rollback |
| **CV cutoffs** | ❌ March (broken) | ✅ February (≥6 weeks) |
| **Graph features** | ⚠️ Horizon only | ✅ Weekly moments |
| **Spike prevention** | ❌ Manual | ✅ Targeted auto-cap |
| **p0 computation** | ❌ All weeks | ✅ Available weeks only |
| **Service targets** | ❌ Global | ✅ ABC/XYZ segmented |
| **Export QA** | ❌ Manual | ✅ 6 automated checks |
| **Cell order bugs** | ❌ Frequent | ✅ Impossible |

---

## 📈 Performance Comparison

### Old Notebook (`hierarchical_bayes_final_store.ipynb`):
- CV cost: 2411 (but with broken March cutoffs)
- Actual expected: ~3300-3400
- Issues: Wrong CV, global targets, no caps

### New Scripts:
- **Baseline**: CV 3274 (corrected)
- **Graph (uncapped)**: CV 3204 (corrected, weekly features)
- **Graph (capped)**: CV 3204, max 270 (safe to submit)

**Net improvement**: ~100-200 points vs old approach

---

## 🔧 Module Reference

```
vn2inventory/
├── hb_core.py                    # GLM, EB, tau², dispersion
├── friend_recommendations.py     # NB2, p0, ABC/XYZ, QA
├── graph_integration.py          # Graph load/merge (weekly features)
├── cv_optimizer.py               # Optuna CV
├── state_reconstruction.py       # Historical state rollback ✨
├── hb_pipeline.py                # Orchestrator + cap logic
├── data_io.py                    # Existing utilities
└── sim_env.py                    # Inventory simulator

scripts/
└── run_hb_solution.py            # CLI entry point
```

---

## 🐛 Troubleshooting

### Graph Features Not Found:
```
⚠️ Graph features not found: .../orders_features_599_idemb_asof.csv
```
→ Check path or fallback to baseline automatically

### Baseline Not Found (for --apply-cap):
```
⚠️ Baseline not found at submissions/baseline_cv/..., skipping cap
```
→ Run baseline first: `python scripts/run_hb_solution.py --model baseline`

### CV Validation Fails:
```
❌ 2024-02-26: 4 weeks available (need >=4)
```
→ Scripts use corrected cutoffs (Feb 05-26); this shouldn't happen

---

## 🎁 For Your Team

**Reproduce a graph-enhanced run**:
1. Clone repo
2. Install: `pip install -r requirements.txt`
3. Run: `python scripts/run_hb_solution.py --model graph-enhanced --features-599 path/to/orders_features_599.csv --output-dir submissions/final --apply-cap`

**Runtime**: ~1-2 minutes (CV cached in pipeline)
