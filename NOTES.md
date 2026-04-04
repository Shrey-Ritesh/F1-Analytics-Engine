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
