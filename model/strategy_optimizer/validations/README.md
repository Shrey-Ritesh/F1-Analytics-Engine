# Strategy Optimizer Validations

Command-line utilities for inspecting, debugging, and validating the pit stop strategy optimizer.

## Quick Reference

```bash
cd f1_ai_strategy_system
PYTHONPATH=. python model/strategy_optimizer/validations/<script_name>.py
```

---

## Scripts

### `check_pit_data.py`
Inspect pit stop events from the feature dataset.

**Output:** Counts of pit stops by circuit and race year.

```bash
PYTHONPATH=. python model/strategy_optimizer/validations/check_pit_data.py
```

---

### `check_priors.py`
View historical prior scores for key circuits.

**Displays:**
- Stop distribution (% 1-stop, 2-stop, 3-stop, etc.)
- Pit window timing (mean ± std per stop number)
- Historical sample size per circuit

**Circuits:** Bahrain, Monaco, Qatar

```bash
PYTHONPATH=. python model/strategy_optimizer/validations/check_priors.py
```

---

### `check_stint_ranges.py`
Analyze stint length distributions from historical race strategies.

**Displays:**
- Overall stint stats by compound
- Percentiles (5th, 25th, 50th, 75th, 95th) per compound
- Opening stint lengths (first stint in race)
- Final stint lengths (last stint before finish)

**Used to derive STINT_BOUNDS:**
```python
SOFT:   min=9,  max=27 laps   (5th-95th percentile)
MEDIUM: min=13, max=35 laps
HARD:   min=19, max=47 laps
```

```bash
PYTHONPATH=. python model/strategy_optimizer/validations/check_stint_ranges.py
```

---

### `sanity_check.py`
Validate circuit strategy profile data.

**Output:** Sample sizes and winning strategies for selected circuits.

**Circuits:** Dutch GP, Canadian GP

```bash
PYTHONPATH=. python model/strategy_optimizer/validations/sanity_check.py
```

---

### `extra_validations.py`
Extended optimizer test scenarios across multiple circuits.

**Tests on:**
- Miami Grand Prix (57 laps)
- Emilia Romagna Grand Prix (63 laps)
- British Grand Prix (52 laps)
- Italian Grand Prix (53 laps)

**Default parameters:**
- Driver ID: 1
- Team: Aston Martin (encoded=3)
- Grid position: 5
- Start compound: MEDIUM
- Stop range: 1–2 stops

```bash
PYTHONPATH=. python model/strategy_optimizer/validations/extra_validations.py
```

Modify the `scenarios` list in the script to test different circuits or parameters.

---

## Why these are in the strategy optimizer folder

These validation utilities are tightly coupled to the strategy optimizer's:
- Circuit profiles and prior scores (`circuit_strategy_profiles.json`)
- Stint bound definitions (from `check_stint_ranges.py` output)
- Historical strategy reconstruction (for prior scoring)

Keeping them here makes it clear they're part of the optimizer's validation/debugging toolkit.

---

## Integration with main optimizer

All validations read from shared data files:
- `circuit_strategy_profiles.json` — pit windows, stop distributions
- `pit_stop_optimizer.py` — optimize_strategy(), print_strategy_report()
- `race_simulator.py` — lap time prediction

These scripts are **independent tools** — they don't affect the main optimizer, they only inspect its inputs and outputs.

---

## Adding new validation scripts

When adding a script to validate the optimizer:

1. Create it in this folder: `model/strategy_optimizer/validations/<name>.py`
2. Use `Path(__file__).parent` for relative path navigation (see examples above)
3. Avoid relative imports; instead use absolute imports or sys.path manipulation (see `extra_validations.py`)
4. Add an entry to this README under the "Scripts" section

Example:
```python
from pathlib import Path
data_path = Path(__file__).parent.parent.parent.parent / 'data' / 'my_file.csv'
```
