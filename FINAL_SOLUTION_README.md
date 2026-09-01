# Hierarchical Bayes pipeline

The official public path is `scripts/run_hb_solution.py` on `main`.

SKU shrinkage is on the **log-mean** scale (`Var(log residual) / n_weeks`). An earlier mix of that with NB count variance collapsed every SKU onto its department mean. The hurdle uses `μ / (1 − p0)` as the positive component.

A production-ready Hierarchical Bayes pipeline with:
- ✅ **Historical state reconstruction** for accurate CV
- ✅ **Corrected CV cutoffs** (Feb 05-26, all have ≥6 weeks future data)
- ✅ **Friend's recommendations** (p0, ABC/XYZ, canonical NB2, guardrails)
- ✅ **Graph integration** with weekly moments (mu_w1-w3, sigma_w1-w3)
- ✅ **Targeted spike cap** for graph runs (preserves 96% volume, trims outliers)

---

## 📁 Modules Created

### Core (`vn2inventory/`):
1. **`hb_core.py`** - GLM fitting, EB shrinkage, tau² sweep, dispersion
2. **`friend_recommendations.py`** - NB2, p0, ABC/XYZ, guardrails, QA
3. **`graph_integration.py`** - Graph loading (prefers weekly features)
4. **`cv_optimizer.py`** - Optuna CV with cutoff validation
5. **`state_reconstruction.py`** - Historical inventory state rollback ✨ NEW
6. **`hb_pipeline.py`** - Main orchestrator with cap option

### CLI (`scripts/`):
7. **`run_hb_solution.py`** - Single command interface

---

## Run

```bash
# Graph-enhanced with weekly features + targeted cap
mkdir -p submissions/final
python scripts/run_hb_solution.py \
  --model graph-enhanced \
  --features-599 path/to/orders_features_599.csv \
  --output-dir submissions/final \
  --apply-cap

# Output: submissions/final/orders_hierarchical_graph_enhanced_cv.csv
```

Typical output shape:
- CV cost: **~3204** (best of all runs)
- Max order: **270** (capped, was 380)
- Total units: **~2752** (balanced)
- All QA checks: ✅

---

## 📊 Results Summary

| Model | CV Cost | Total Units | Max Order | Zeros | File |
|-------|---------|-------------|-----------|-------|------|
| **Baseline** | 3274.3 | 2,866 | 126 | 198 | baseline_cv/orders_*.csv |
| **Graph (gated2)** | 3211.5 | 2,868 | 392 | 309 | graph_cv/orders_*.csv |
| **Graph (asof weekly)** | 3204.5 | 2,862 | 380 | 311 | graph_cv_asof_v2/orders_*.csv |
| **Graph (asof + cap)** ✨ | **3204.5** | **2,752** | **270** | **311** | **final/orders_*.csv** |

In-sample CV preferred graph + cap. Those scores did **not** hold on the live rounds: a unit bug in EB shrinkage collapsed every SKU onto its department mean. That is fixed on `main`.

---

## Design notes

### 1. **Corrected CV** (Historical State + Early Cutoffs)
- **Old**: March cutoffs with Week 0 state approximation
- **New**: February cutoffs with reconstructed historical states
- **Impact**: Accurate cost measurement → better knobs

### 2. **Weekly Graph Features** (No i.i.d. Assumption)
- **Old**: `mu_H / 3`, `sigma_H / sqrt(3)` (assumes independence)
- **New**: `avg(mu_w1, mu_w2, mu_w3)`, `avg(sigma_w1-w3)` (uses actual weekly)
- **Impact**: Better GLM fit (RMSE 8.46 → 8.42)

### 3. **Targeted Spike Cap** (Preserves Volume)
- **Issue**: Graph max 380 (risky outlier)
- **Solution**: Cap only if `graph > 4×HB AND graph > 220`
  - Cap = `max(HB×3.5 + 65, 270)`
- **Result**: Max 380 → 270, volume 2862 → 2752 (96% retained)

### 4. **All Friend's Recommendations**
- Availability-aware p0 ✅
- Canonical NB2 ✅
- ABC/XYZ segmented targets ✅
- Multi-layer guardrails ✅
- Export QA ✅

---

## 🎯 Quick Reference Commands

### Graph-enhanced:
```bash
mkdir -p submissions/final
python scripts/run_hb_solution.py --model graph-enhanced --features-599 path/to/orders_features_599.csv --output-dir submissions/final --apply-cap
```

### Baseline Only (If Graph Fails):
```bash
python scripts/run_hb_solution.py --model baseline --output-dir submissions/baseline_final
```

### Quick Test (No CV):
```bash
python scripts/run_hb_solution.py --model graph-enhanced --skip-cv --apply-cap
```

---

## 📋 CLI Options

| Flag | Options | Description |
|------|---------|-------------|
| `--model` | `baseline`, `graph-enhanced` | Model variant |
| `--apply-cap` | flag | Apply targeted cap vs HB (graph only) |
| `--skip-cv` | flag | Use default knobs (faster) |
| `--output-dir` | path | Where to save results |
| `--data-dir` | path | Data directory (default: ../data) |

---

## Outputs

- Orders CSV under `--output-dir`
- Optional execution log if you tee stdout

See [scripts/README_HB_CLI.md](scripts/README_HB_CLI.md) for flags.
