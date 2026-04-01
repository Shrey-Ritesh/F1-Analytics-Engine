# F1 AI Strategy System — Claude Context

## Project overview
End-to-end ML system for Formula 1 race strategy prediction and lap time modelling.
Built in Python using XGBoost, FastF1 API, and Streamlit.

**GitHub:** `https://github.com/Shrey-Ritesh/F1-Analytics-Engine`
**Dataset:** 76,163 laps across 2023, 2024, 2025 seasons (24 circuits, ~32 drivers)

---

## How to run

```bash
# All commands from: f1_ai_strategy_system/
cd "f1_ai_strategy_system"

# Dashboard (main UI)
PYTHONPATH=. venv/bin/streamlit run dashboard/Home.py

# Strategy optimizer (CLI validation — runs Bahrain/Monaco/Qatar scenarios)
PYTHONPATH=. venv/bin/python -m model.strategy_optimizer.pit_stop_optimizer

# Retrain lap time model
PYTHONPATH=. venv/bin/python model/lap_time_model/v7_temporal_train.py

# Rebuild driver performance metrics
PYTHONPATH=. venv/bin/python feature_engineering/driver_performance_analyzer_v2.py

# Full data pipeline (slow — fetches from FastF1 API)
PYTHONPATH=. venv/bin/python main.py
```

**PYTHONPATH=. is required** for all module imports. Streamlit uses conda's system `streamlit` binary but must have the venv modules on the path.

---

## Architecture

```
Ingestion → Preprocessing → Feature Engineering → Lap Time Model → Strategy Optimizer → Dashboard
```

### Module status

| Module | Status | Key file |
|--------|--------|----------|
| Data pipeline | ✅ Complete | `ingestion/fetch_data.py`, `main.py` |
| Feature engineering | ✅ Complete | `feature_engineering/expand_features.py` |
| Lap time model v7 | ✅ Complete | `model/lap_time_model/v7_temporal_train.py` |
| Pit stop optimizer | ✅ Complete | `model/strategy_optimizer/pit_stop_optimizer.py` |
| Driver performance | ✅ Complete | `feature_engineering/driver_performance_analyzer_v2.py` |
| Dashboard | ✅ Complete | `dashboard/Home.py` + 4 pages |
| Safety car predictor | 🔲 Not started | Module 4 |
| Race outcome predictor | 🔲 Not started | Module 5 |

---

## Project structure

```
f1_ai_strategy_system/
├── data/
│   ├── processed/
│   │   ├── driver_performance_metrics_v2.csv   ← 27 drivers, regenerated after consistency fix
│   │   └── historical_strategies.csv           ← 1,360 reconstructed race strategies
│   ├── training_data/
│   │   ├── f1_training_dataset.csv             ← 76,163 laps, 38 features (encoded)
│   │   └── category_mappings.json              ← int encodings for team/circuit/tire_compound
│   ├── f1_features_dataset.csv                 ← same but with string columns (unencoded)
│   └── master_dataset.csv                      ← 9-column base before expand_features
├── feature_engineering/
│   ├── build_features.py                       ← 9 base features from cleaned data
│   ├── expand_features.py                      ← 9 → 39 features (no leakage)
│   └── driver_performance_analyzer_v2.py       ← driver scores (rebuilt April 2026)
├── ingestion/fetch_data.py
├── preprocessing/clean_data.py
├── model/
│   ├── lap_time_model/
│   │   ├── lap_time_model_v7.pkl               ← production XGBoost model
│   │   ├── circuit_baselines.json              ← 24 circuits, median clean lap time
│   │   ├── pit_loss_estimates.json             ← per-circuit pit penalties (11.8–38.0s)
│   │   └── model_metadata.json                 ← low_confidence_circuits list
│   └── strategy_optimizer/
│       ├── pit_stop_optimizer.py               ← main optimizer + realism checks
│       ├── race_simulator.py                   ← vectorized batch lap prediction
│       ├── historical_strategy_extractor.py    ← circuit profile builder + prior scoring
│       ├── regulations.py                      ← FIA rule validator
│       ├── regulations.json                    ← 2025 circuit-specific rules
│       └── circuit_strategy_profiles.json      ← 24 circuits, stop dist + pit windows
├── dashboard/
│   ├── Home.py                                 ← entry point
│   ├── pages/
│   │   ├── 1_Strategy_Optimizer.py
│   │   ├── 2_Circuit_Intelligence.py
│   │   ├── 3_Driver_Performance.py
│   │   └── 4_Historical_Analysis.py
│   └── utils/
│       ├── constants.py   ← CIRCUIT_LAPS, DRIVER_NAMES, COMPOUND_COLORS
│       ├── data_loader.py ← all @st.cache_data loaders + load_optimizer()
│       └── charts.py      ← make_stint_bar(), make_driver_radar(), dark_layout()
├── main.py
└── requirements.txt
```

---

## Lap time model v7

- **Algorithm:** XGBoost Regressor
- **Target:** `lap_time_seconds` (absolute seconds)
- **Train:** 2023 + 2024 (42,294 clean laps) → **Test:** 2025 (21,737 clean laps)
- **RMSE:** 1.88s | **MAE:** 1.46s | **R²:** 0.97
- **Bias correction:** +1.05s added at inference (mean residual offset)
- **Low-confidence circuits:** Canadian GP, British GP, Australian GP (wet-data depletion) — simulator widens bounds ±4s for these

### Clean lap filter (applied to both splits, threshold from training only)
```python
pit_stop == 0
lap > 1
tire_compound not in [INTERMEDIATE, WET]  # encoded as 1, 4
lap_time_seconds < circuit_baseline_pace * 1.07  # removes SC/VSC laps
```

### Top features (by XGBoost gain)
1. `circuit_baseline_pace` (28,891 — non-leaky median from training set only)
2. `round_number` (4,198)
3. `compound_base_deg_rate` (870)
4. `track_temperature` (729)
5. `stint_number` (549)

### Critical leakage — never use these as features
- `expected_lap_time` — derived from actual lap times
- `tire_degradation_rate` — uses `lap_time_seconds` in numerator
- `final_race_position` — post-race result

---

## Strategy optimizer

### How it works
1. **Enumerate** valid strategies (pit laps × compound combinations) — pruned by STINT_BOUNDS and FIA regulations
2. **Simulate** all strategies in one vectorized `model.predict()` call (2M rows in ~0.5s)
3. **Score** each strategy: `combined = 0.70 × time_score + 0.30 × prior_score`
4. **Prior score** = `0.35 × stop_score + 0.40 × pit_window_score + 0.25 × compound_score`

### STINT_BOUNDS (empirical, 25th–95th percentile from 2023–2025 data)
```python
SOFT:   min=9,  max=27 laps
MEDIUM: min=13, max=35 laps
HARD:   min=19, max=47 laps
```

### Compound encodings (must match training data)
```python
COMPOUND_ENCODING = {'HARD': 0, 'MEDIUM': 2, 'SOFT': 3}
```

### Known issue (not yet fixed)
The optimizer over-recommends 3-stop strategies because:
1. **Tire degradation is linear in the model** — a 40-lap HARD stint looks catastrophically slow compared to a 20-lap stint, but real tires plateau. Fix: cap `tire_age_laps` at compound-specific plateau in feature matrix (SOFT: 15, MEDIUM: 22, HARD: 30).
2. **No track position cost** — each pit stop loses ~8-12s of track position (circuit-dependent). Fix: add `traffic_cost = overtaking_difficulty × 10s` per stop to the simulated race time.
3. **Prior weight is 30%** — consider raising to 40% since historical data strongly favours 1-2 stops at most circuits.

These fixes go in `model/strategy_optimizer/race_simulator.py` and `pit_stop_optimizer.py`.

---

## Driver performance metrics

**File:** `data/processed/driver_performance_metrics_v2.csv`

### Score formula
```
driver_performance_score = 0.40 × pace_score + 0.25 × consistency_score
                         + 0.20 × podium_rate + 0.15 × win_rate
```

### Consistency score (fixed April 2026)
- Old (wrong): std of `lap_time_delta` across ALL laps — penalised fast drivers for strategic pace variation
- New (correct): mean within-stint std of `lap_time_delta`, computed on clean laps only (no pit laps, `tire_age > 2`, dry compounds)
- Clean lap filter uses **integer compound codes**: HARD=0, MEDIUM=2, SOFT=3 (not strings — training data is pre-encoded)

### Current top 5
| Driver | Score | Notes |
|--------|-------|-------|
| VER | 0.838 | Fastest avg delta (-0.11s), wins 100% |
| NOR | 0.608 | Strong pace, high podium rate |
| LEC | 0.581 | Consistent pace, limited wins 2023-25 |
| HAM | 0.533 | Solid pace, zero wins in this period |
| RUS | 0.484 | Good pace, some wins |

---

## Dashboard pages

| Page | Route | Key inputs |
|------|-------|-----------|
| Home | `/` | None — static overview |
| Strategy Optimizer | `1_Strategy_Optimizer` | Circuit, driver, team, grid pos, compound, laps, track temp |
| Circuit Intelligence | `2_Circuit_Intelligence` | Circuit selectbox |
| Driver Performance | `3_Driver_Performance` | Driver multiselect (radar), single driver (history) |
| Historical Analysis | `4_Historical_Analysis` | Year filter, quality filter, optional circuit |

### Key patterns in dashboard code
- `load_optimizer()` uses `@st.cache_resource` — wraps the import of `optimize_strategy` so the pkl is loaded only once; stdout is suppressed during import and call
- Strategy runs are cached with `@st.cache_data` keyed on all inputs — `n_stops_range` must be a `tuple` (not list) to be hashable
- All file reads use `@st.cache_data` in `dashboard/utils/data_loader.py`
- `dark_layout()` in `charts.py` applies consistent dark theme to all Plotly figures
- `make_stint_bar()` builds horizontal stacked compound bars from `pit_laps + compounds + total_laps`

---

## Encodings reference

### Team → integer
```
0: Alfa Romeo    1: AlphaTauri   2: Alpine        3: Aston Martin
4: Ferrari       5: Haas F1 Team 6: Kick Sauber   7: McLaren
8: Mercedes      9: RB          10: Racing Bulls  11: Red Bull Racing
12: Williams
```

### Tire compound → integer
```
0: HARD   1: INTERMEDIATE   2: MEDIUM   3: SOFT   4: WET
```

### Circuit → integer (0–23, alphabetical order in category_mappings.json)

---

## Regulations (2025)

```json
Default:  min_stops=1, min_compounds=2
Monaco:   min_stops=2, min_compounds=3
Qatar:    min_stops=2, max_laps_per_set=25
```

---

## What to work on next

### Priority 1 — Fix 3-stop bias in optimizer (race_simulator.py)
- Cap `tire_age_laps` at plateau per compound in `build_feature_matrix()`
- Add `traffic_cost` per pit stop based on `overtaking_difficulty`
- Raise prior weight from 0.30 to 0.40 in `optimize_strategy()`

### Priority 2 — Module 4: Safety Car Predictor
- Classification model (XGBoost) per lap
- Features: circuit characteristics, historical SC rate, gap variance, race phase
- Output: P(safety car on this lap)

### Priority 3 — Module 5: Race Outcome Predictor
- Combines driver scores + predicted pace + pit strategies + SC probability
- Output: P(finishing position) per driver
