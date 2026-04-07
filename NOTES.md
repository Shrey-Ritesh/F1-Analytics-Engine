# F1 Project — Notes & Known Issues

## Data Quality

### 1. Tire Degradation Rate Formula (Current)
The existing formula used across the dataset:
```python
tire_degradation_rate = (lap_time - min_lap_time_in_stint) / tire_age
```
**Problems:**
- `min_lap_time_in_stint` is noisy (outliers, future data leaking into past laps)
- Does not remove fuel load effect — conflates fuel burn with tyre deg
- Warm-up laps (1–2) make early degradation look artificially steep
- Safety car laps corrupt the min entirely
- Measures cumulative average, not the actual rate of degradation

**Fix planned for Phase 2:**
1. Fuel-correct all lap times via `fuel_model.py` first
2. Recompute using **linear regression slope** of `fuel_corrected_lap_time` vs `tire_age_laps` per stint (skip first 2 warm-up laps)
3. Use clean slope-based rates in `circuit_dna.py` for `soft_deg_rate`, `medium_deg_rate`, `hard_deg_rate`

> Current proxy is kept as-is in the existing lap time model v7 (no retraining needed — XGBoost handles noisy features). All Phase 2 modules will use the better calculation.

---

### 2. Junior Red Bull Team — Triple Encoding
The junior Red Bull sister team appears as **3 separate team encodings** in `category_mappings.json` due to rebranding:

| Code | Name | Season |
|------|------|--------|
| `1` | AlphaTauri | 2023 |
| `9` | RB | Early 2024 |
| `10` | Racing Bulls | Late 2024 / 2025 |

These are the **same team** (same car, chassis, infrastructure). The model treats them as 3 distinct teams.

**Impact areas in Phase 2:** `circuit_dna.py`, `driver_form.py` — anywhere team grouping matters.

**Possible fix:** Consolidate all three under a single label (e.g. `"Racing Bulls"`) before feature engineering in Phase 2.

---

### 3. São Paulo Grand Prix — Unicode Encoding
`category_mappings.json` was storing `ã` as `\u00e3` due to Python's default `ensure_ascii=True`.

**Fixed:** Added `ensure_ascii=False` to `json.dump` in `expand_features.py`. Existing file also corrected directly.

---

## Dataset

### 4. f1_features_dataset vs f1_training_dataset
| | `f1_features_dataset.csv` | `f1_training_dataset.csv` |
|---|---|---|
| Purpose | Human-readable | ML-ready |
| `driver` column | ✅ String name included | ❌ Dropped |
| `circuit` column | ✅ String name included | ❌ Dropped (encoded as int) |
| Categoricals | String labels | Integer encoded via `category_mappings.json` |
| `compound_interaction` | ❌ Not present | ✅ Present (`stint_progress_pct × compound_base_deg_rate`) |

---

## Column Name Remapping (Dataset vs Phase 2 Brief)
The dataset uses different column names from what the Phase 2 brief assumes:

| Brief assumes | Actual column name |
|---|---|
| `lap_number` | `lap` |
| `tire_age` | `tire_age_laps` |
| `lap_time` | `lap_time_seconds` |
| `lap_time_delta` | `relative_pace_delta` |
| `pit_stop_flag` | `pit_stop` |
| `track_temp` | `track_temperature` |
| `expected_lap_time_baseline` | `expected_lap_time` |
| `gap_to_leader` | `gap_to_leader_seconds` |
| `gap_to_car_ahead` | `gap_to_car_ahead_seconds` |

---

## Phase 2 Build Notes

### 5. No Safety Car Flag in Dataset
There is no `safety_car_lap` column in the dataset. SC laps must be **inferred** via lap time spike detection:
- Compute median lap time per lap per race
- Rolling 5-lap median
- Flag laps where ratio > 1.28 (field-wide 28%+ slowdown)

### 6. Fuel Burn Rate Discrepancy
- Existing pipeline uses **1.5 kg/lap**
- Phase 2 `fuel_model.py` will use **1.85 kg/lap** (more accurate F1 value)
- The existing `fuel_load_estimate` column in the dataset uses 1.5 kg/lap — Phase 2 will add a new `fuel_load_kg` column alongside it

---

## Pit Stop Optimizer — Known Issues & Phase 2 Fixes

### 7. Physics Model Has a Systematic Multi-Stop Bias
**Finding:** The optimizer's combined score (50% physics time + 50% historical prior) consistently over-recommends 3-stop strategies, even at circuits where 1-stop is historically dominant 80–94% of the time (Azerbaijan, Saudi Arabia, Singapore).

**Root cause:** The XGBoost lap time model correctly predicts SOFT tires are faster lap-to-lap. With no constraint on track position value, the physics component always prefers the highest feasible stop count. The `time_score` normalization creates extreme spread (e.g. `0.0015` for 1-stop vs `0.9958` for 3-stop at Azerbaijan), which the 50% prior weight cannot overcome regardless of how confident the prior is.

**Validation results across 8 circuits (driver_id=1, team=3, grid=P3, temp=35°C):**
| Circuit | Hist. dominant | Model top pick | Match |
|---|---|---|---|
| Azerbaijan GP | 1-stop (94%) | 3-stop | ✗ |
| Saudi Arabian GP | 1-stop (91%) | 3-stop | ✗ |
| Singapore GP | 1-stop (80%) | 2-stop | ✗ |
| Bahrain GP | 2-stop (82%) | 3-stop | ✗ |
| Hungarian GP | 2-stop (76%) | 2-stop | ✓ |
| Spanish GP | 2-stop (70%) | 3-stop | ✗ |
| Qatar GP | 3-stop (43%) | 3-stop | ✓ |
| Austrian GP | 3-stop (36%) | 3-stop | ✓ |

Overall: **3/8 correct (38%)**. The model only gets it right when physics and history agree (high-deg circuits that genuinely need more stops).

---

### 8. `fuel_model.py` and `circuit_dna.py` Will NOT Fix the Multi-Stop Bias
- **`fuel_model.py`** corrects fuel load by ~0.055s/lap. Total swing between 1-stop and 3-stop over 50 laps ≈ 1–3s — nowhere near the 87–156s gaps observed. Negligible impact on strategy ranking.
- **`circuit_dna.py`** builds a circuit fingerprint/archetype clustering. It's a representation layer — it doesn't touch the optimizer's scoring function. No direct effect unless explicitly wired in.

---

### 9. Three Things That Will Actually Fix the Bias

**Fix 1 — Circuit-specific prior weight (quick, within optimizer)**
Scale the prior weight by `overtaking_difficulty` from `circuit_strategy_profiles.json`. At low-overtaking circuits (OT > 0.7), raise prior weight to ~65–70% instead of 50%. This is the minimum viable fix and can be done before Phase 2.

```python
# In optimize_strategy(), replace hardcoded 0.50 / 0.50:
ot = circuit_profiles.get(circuit, {}).get('overtaking_difficulty', 0.5)
prior_weight = 0.50 + 0.20 * ot        # 0.50 at OT=0 → 0.70 at OT=1.0
physics_weight = 1.0 - prior_weight
s['combined_score'] = physics_weight * s['time_score'] + prior_weight * s['prior_score']
```

**Fix 2 — `race_outcome.py` (Phase 2, high impact)**
Replacing lap-time-sum with actual position-based race outcome modeling will implicitly capture track position value. Losing 10 positions in a pit stop at Monaco that you can never recover is not reflected in any lap time model.

**Fix 3 — Tire degradation fix (already planned in NOTES #1)**
Better slope-based degradation rates will make Hard tires look more competitive on long stints, narrowing the performance gap that currently makes multi-stop look overwhelmingly better to the physics component.

---

### 10. Prior Score Calibration Is Sound — The Weighting Is the Problem
The `score_strategy_prior()` function in `historical_strategy_extractor.py` works correctly. At Azerbaijan, the 1-stop prior score is **0.887** — it correctly identifies the dominant strategy. The issue is purely in how time_score and prior_score are combined. Do not change the prior scoring function; change the weight in `optimize_strategy()`.
