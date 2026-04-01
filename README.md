# F1 Analytics Engine 🏎️

End-to-end ML system for Formula 1 race strategy prediction, lap time modelling, and driver performance analysis. Built with Python, XGBoost, FastF1, and Streamlit.

**Dataset:** 76,163 laps · 2023–2025 seasons · 24 circuits · ~32 drivers

---

## Quickstart

> All commands must be run from inside the `f1_ai_strategy_system/` directory.

```bash
cd "f1_ai_strategy_system"
```

---

## 1. Launch the Dashboard

The main UI — strategy optimizer, circuit intelligence, driver performance, and historical analysis.

```bash
PYTHONPATH=. streamlit run dashboard/Home.py
```

Then open **http://localhost:8501** in your browser.

> **Note:** If `streamlit` isn't on your PATH, use the venv binary:
> ```bash
> PYTHONPATH=. venv/bin/streamlit run dashboard/Home.py
> ```

---

## 2. Run the Strategy Optimizer (CLI)

Runs Bahrain, Monaco, and Qatar validation scenarios and prints ranked strategy tables with realism checks.

```bash
PYTHONPATH=. python -m model.strategy_optimizer.pit_stop_optimizer
```

---

## 3. Retrain the Lap Time Model

Trains XGBoost on 2023–2024 data, evaluates on 2025, saves `lap_time_model_v7.pkl`.

```bash
PYTHONPATH=. python model/lap_time_model/v7_temporal_train.py
```

---

## 4. Rebuild Driver Performance Metrics

Recomputes driver scores (pace, consistency, win rate, podium rate) from the training dataset.

```bash
PYTHONPATH=. python feature_engineering/driver_performance_analyzer_v2.py
```

---

## 5. Run the Full Data Pipeline

Fetches raw data from the FastF1 API, cleans, engineers features, and writes all CSVs. **Slow — makes live API calls.**

```bash
PYTHONPATH=. python main.py
```

---

## Architecture

```
FastF1 API
    ↓
ingestion/fetch_data.py          — raw lap/weather/compound data
    ↓
preprocessing/clean_data.py      — imputation, type casting
    ↓
feature_engineering/
  build_features.py              — 9 base features
  expand_features.py             — 9 → 39 features (no leakage)
  driver_performance_analyzer_v2 — driver ranking scores
    ↓
model/lap_time_model/
  v7_temporal_train.py           — XGBoost regressor (RMSE 1.88s)
  lap_time_model_v7.pkl          — production model
    ↓
model/strategy_optimizer/
  pit_stop_optimizer.py          — enumerate + rank all valid strategies
  race_simulator.py              — vectorized batch lap prediction
    ↓
dashboard/                       — Streamlit multi-page app
```

---

## Module Status

| Module | Status |
|--------|--------|
| Data pipeline | ✅ Complete |
| Feature engineering | ✅ Complete |
| Lap time model v7 | ✅ Complete |
| Pit stop optimizer | ✅ Complete |
| Driver performance | ✅ Complete |
| Dashboard (5 pages) | ✅ Complete |
| Safety car predictor | 🔲 Not started |
| Race outcome predictor | 🔲 Not started |

---

## Model Performance

| Metric | Value |
|--------|-------|
| Algorithm | XGBoost Regressor |
| Train set | 2023 + 2024 (42,294 laps) |
| Test set | 2025 (21,737 laps) |
| RMSE | 1.88s |
| MAE | 1.46s |
| R² | 0.97 |

---

## Project Structure

```
f1_ai_strategy_system/
├── data/
│   ├── processed/
│   │   ├── driver_performance_metrics_v2.csv
│   │   └── historical_strategies.csv
│   ├── training_data/
│   │   ├── f1_training_dataset.csv
│   │   └── category_mappings.json
│   └── f1_features_dataset.csv
├── feature_engineering/
│   ├── build_features.py
│   ├── expand_features.py
│   └── driver_performance_analyzer_v2.py
├── ingestion/fetch_data.py
├── preprocessing/clean_data.py
├── model/
│   ├── lap_time_model/
│   │   ├── v7_temporal_train.py
│   │   ├── lap_time_model_v7.pkl
│   │   ├── circuit_baselines.json
│   │   ├── pit_loss_estimates.json
│   │   └── model_metadata.json
│   └── strategy_optimizer/
│       ├── pit_stop_optimizer.py
│       ├── race_simulator.py
│       ├── regulations.py
│       ├── regulations.json
│       └── circuit_strategy_profiles.json
├── dashboard/
│   ├── Home.py
│   ├── pages/
│   │   ├── 1_Strategy_Optimizer.py
│   │   ├── 2_Circuit_Intelligence.py
│   │   ├── 3_Driver_Performance.py
│   │   └── 4_Historical_Analysis.py
│   └── utils/
│       ├── constants.py
│       ├── data_loader.py
│       └── charts.py
├── main.py
└── requirements.txt
```

---

## Setup from Scratch

```bash
# Clone the repo
git clone https://github.com/Shrey-Ritesh/F1-Analytics-Engine.git
cd F1-Analytics-Engine/f1_ai_strategy_system

# Create virtual environment
python3.9 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# macOS only — XGBoost requires OpenMP
brew install libomp
```

> `PYTHONPATH=.` is required for all commands because the codebase uses absolute imports like `from model.strategy_optimizer import ...`. Always set it as shown in the commands above.
