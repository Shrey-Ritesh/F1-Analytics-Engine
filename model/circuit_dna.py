"""
circuit_dna.py — Phase 2 circuit fingerprinting and archetype clustering
========================================================================
Builds an 18-feature fingerprint for each of the 24 F1 circuits and
assigns one of 4 archetypes via KMeans (k=4) clustering.

Archetypes
----------
  street_circuit   — 1-stop dominant, track position is precious
                     (Azerbaijan, Italian, Miami, Saudi Arabian, Singapore)
  high_degradation — multi-stop forced by tire wear, 3-stop tendency
                     (Austrian, Bahrain, Qatar, Spanish)
  high_overtaking  — 2-stop dominant, overtaking feasible, active strategy
                     (Abu Dhabi, Australian, Belgian, British, Canadian,
                      Dutch, Hungarian, Las Vegas)
  balanced         — mixed strategy profile, no single dominant approach
                     (Chinese, Emilia Romagna, Japanese, Mexico City,
                      Monaco, São Paulo, United States)

Design note
-----------
Clustering uses CLUSTERING_FEATURES (9 strategy-profile features), a
curated subset of FEATURE_NAMES.  Raw lap-physics features (soft_deg_rate,
lap_time_std) are excluded from clustering because their extreme outliers
(Las Vegas surface / Monaco safety-car variance) collapse KMeans into
singletons even after log-transform.  All 18 FEATURE_NAMES are still
written to the output CSV for use by downstream modules.

Outputs
-------
  data/processed/circuit_dna.csv          — per-circuit fingerprint + label
  data/processed/circuit_archetypes.json  — archetype → circuits mapping
"""

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths (all resolved relative to this file's parent's parent = project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_CSV = PROJECT_ROOT / "data" / "f1_features_dataset.csv"
PROFILES_JSON = PROJECT_ROOT / "model" / "strategy_optimizer" / "circuit_strategy_profiles.json"
BASELINES_JSON = PROJECT_ROOT / "model" / "lap_time_model" / "circuit_baselines.json"
PIT_LOSS_JSON = PROJECT_ROOT / "model" / "lap_time_model" / "pit_loss_estimates.json"

OUT_CSV = PROJECT_ROOT / "data" / "processed" / "circuit_dna.csv"
OUT_JSON = PROJECT_ROOT / "data" / "processed" / "circuit_archetypes.json"

FEATURE_NAMES = [
    "baseline_lap_time",
    "lap_time_std",
    "avg_stint_length",
    "track_temperature_mean",
    "pit_loss_time",
    "soft_deg_rate",
    "medium_deg_rate",
    "hard_deg_rate",
    "overtaking_difficulty",
    "one_stop_pct",
    "two_stop_pct",
    "three_stop_pct",
    "dominant_stop_count",
    "first_pit_mean",
    "pit_window_spread",
    "strategy_entropy",
    "deg_spread",
    "top_compound_freq",
]

ARCHETYPE_LABELS = ["street_circuit", "high_degradation", "high_overtaking", "balanced"]

# Features used for KMeans clustering — strategy-focused subset of FEATURE_NAMES.
# Raw lap-physics features (soft_deg_rate, lap_time_std) are excluded here because
# they contain extreme outliers (Las Vegas deg spike, Monaco SC variance) that
# collapse clusters into singletons even after log-transform.
# All 18 FEATURE_NAMES still appear in the output CSV for downstream modules.
CLUSTERING_FEATURES = [
    "overtaking_difficulty",
    "one_stop_pct",
    "two_stop_pct",
    "three_stop_pct",
    "dominant_stop_count",
    "strategy_entropy",
    "first_pit_mean",
    "avg_stint_length",
    "top_compound_freq",
]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _shannon_entropy(dist: dict) -> float:
    """Shannon entropy of a probability distribution given as a dict."""
    total = sum(dist.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for v in dist.values():
        p = v / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def _dominant_stop_key(stop_distribution: dict) -> str:
    """Return the string key ('1', '2', or '3') with the highest probability."""
    return max(stop_distribution, key=lambda k: stop_distribution[k])


def _first_pit_mean(pit_windows: dict, dominant_key: str) -> float:
    """Mean lap of stop_1 for the dominant stop count's pit window."""
    window = pit_windows.get(dominant_key, {})
    stop_1 = window.get("stop_1")
    if stop_1 is None:
        return 0.0
    # stop_1 is [mean, std]
    return float(stop_1[0])


def _pit_window_spread(pit_windows: dict, dominant_key: str) -> float:
    """Mean std across all stops in the dominant stop count's pit windows."""
    window = pit_windows.get(dominant_key, {})
    if not window:
        return 0.0
    stds = []
    for stop_data in window.values():
        # stop_data is [mean, std]
        if isinstance(stop_data, (list, tuple)) and len(stop_data) >= 2:
            stds.append(float(stop_data[1]))
    return float(np.mean(stds)) if stds else 0.0


# ---------------------------------------------------------------------------
# Core function: build_fingerprints
# ---------------------------------------------------------------------------

def build_fingerprints(
    df: pd.DataFrame,
    profiles: dict,
    baselines: dict,
    pit_losses: dict,
) -> pd.DataFrame:
    """
    Build an 18-feature fingerprint DataFrame, one row per circuit.

    Parameters
    ----------
    df        : raw lap data DataFrame (f1_features_dataset.csv)
    profiles  : circuit_strategy_profiles.json as dict
    baselines : circuit_baselines.json as dict
    pit_losses: pit_loss_estimates.json as dict

    Returns
    -------
    DataFrame indexed by circuit name with columns == FEATURE_NAMES
    """
    circuits = sorted(profiles.keys())

    # Pre-compute dataset-wide medians for deg rate fallbacks
    global_soft_med = df.loc[df["tire_compound"] == "SOFT", "tire_degradation_rate"].median()
    global_medium_med = df.loc[df["tire_compound"] == "MEDIUM", "tire_degradation_rate"].median()
    global_hard_med = df.loc[df["tire_compound"] == "HARD", "tire_degradation_rate"].median()

    # Global pit loss fallback
    global_pit_loss = pit_losses.get("__global_fallback__", float(np.mean(
        [v for k, v in pit_losses.items() if k != "__global_fallback__"]
    )))

    rows = []
    for circuit in circuits:
        profile = profiles[circuit]
        circ_df = df[df["circuit"] == circuit]

        # --- Lap-data features ---
        baseline_lap_time = float(baselines.get(circuit, np.nan))
        lap_time_std = float(circ_df["lap_time_seconds"].std()) if len(circ_df) > 1 else 0.0
        avg_stint_length = float(circ_df["stint_length"].mean())
        track_temperature_mean = float(circ_df["track_temperature"].mean())
        pit_loss_time = float(pit_losses.get(circuit, global_pit_loss))

        # Degradation rates per compound (min 5 rows, else global fallback)
        for compound, col_name, global_med in [
            ("SOFT", "soft_deg_rate", global_soft_med),
            ("MEDIUM", "medium_deg_rate", global_medium_med),
            ("HARD", "hard_deg_rate", global_hard_med),
        ]:
            subset = circ_df.loc[circ_df["tire_compound"] == compound, "tire_degradation_rate"]
            if len(subset) >= 5:
                val = float(subset.median())
            else:
                val = float(global_med)
            if compound == "SOFT":
                soft_deg_rate = val
            elif compound == "MEDIUM":
                medium_deg_rate = val
            else:
                hard_deg_rate = val

        # --- Profile features ---
        stop_dist = profile.get("stop_distribution", {})
        pit_windows = profile.get("pit_windows", {})
        top_compounds = profile.get("top_compounds", [])

        overtaking_difficulty = float(profile.get("overtaking_difficulty", 0.0))
        one_stop_pct = float(stop_dist.get("1", 0.0))
        two_stop_pct = float(stop_dist.get("2", 0.0))
        three_stop_pct = float(stop_dist.get("3", 0.0))

        dom_key = _dominant_stop_key(stop_dist) if stop_dist else "2"
        dominant_stop_count = int(dom_key)

        first_pit_mean = _first_pit_mean(pit_windows, dom_key)
        pit_window_spread = _pit_window_spread(pit_windows, dom_key)

        # --- Derived features ---
        strategy_entropy = _shannon_entropy(stop_dist)
        deg_spread = soft_deg_rate - hard_deg_rate
        # Frequency of the most common compound sequence (0–1).
        # Replaces the old compound_diversity which was constant=3 for all circuits.
        top_compound_freq = float(top_compounds[0][1]) if top_compounds else 0.0

        rows.append({
            "circuit": circuit,
            "baseline_lap_time": baseline_lap_time,
            "lap_time_std": lap_time_std,
            "avg_stint_length": avg_stint_length,
            "track_temperature_mean": track_temperature_mean,
            "pit_loss_time": pit_loss_time,
            "soft_deg_rate": soft_deg_rate,
            "medium_deg_rate": medium_deg_rate,
            "hard_deg_rate": hard_deg_rate,
            "overtaking_difficulty": overtaking_difficulty,
            "one_stop_pct": one_stop_pct,
            "two_stop_pct": two_stop_pct,
            "three_stop_pct": three_stop_pct,
            "dominant_stop_count": dominant_stop_count,
            "first_pit_mean": first_pit_mean,
            "pit_window_spread": pit_window_spread,
            "strategy_entropy": strategy_entropy,
            "deg_spread": deg_spread,
            "top_compound_freq": top_compound_freq,
        })

    result = pd.DataFrame(rows).set_index("circuit")
    return result


# ---------------------------------------------------------------------------
# Core function: cluster_circuits
# ---------------------------------------------------------------------------

def cluster_circuits(fingerprint_df: pd.DataFrame, k: int = 4) -> pd.DataFrame:
    """
    Run KMeans clustering on the fingerprint DataFrame and assign archetype labels.

    Two features — soft_deg_rate and lap_time_std — are log1p-transformed
    internally before StandardScaler + KMeans to prevent extreme outliers
    (Las Vegas deg spike, Monaco safety-car variance) from collapsing clusters
    into singletons.  Raw values are preserved in the output DataFrame.

    Parameters
    ----------
    fingerprint_df : DataFrame with FEATURE_NAMES columns (index = circuit name)
    k              : number of clusters (default 4)

    Returns
    -------
    fingerprint_df with two additional columns: archetype_id, archetype_label
    """
    df = fingerprint_df.copy()

    # Cluster on CLUSTERING_FEATURES only (strategy-focused subset).
    # All 18 FEATURE_NAMES are preserved in df / output CSV for downstream use.
    X_raw = df[CLUSTERING_FEATURES].values.astype(float)

    # Normalize and cluster
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    df["archetype_id"] = labels

    # Centroids in original (unscaled) feature space — used for label assignment.
    centroids = scaler.inverse_transform(km.cluster_centers_)
    centroid_df = pd.DataFrame(centroids, columns=CLUSTERING_FEATURES)

    assigned: dict[int, str] = {}
    remaining = list(range(k))

    # --- Greedy priority label assignment ---

    # 1. street_circuit: highest one_stop_pct × overtaking_difficulty
    #    Circuits where track position is precious → drivers run long stints
    #    rather than sacrificing position for a faster tire.
    street_score = centroid_df["one_stop_pct"] * centroid_df["overtaking_difficulty"]
    street_cluster = int(street_score.idxmax())
    assigned[street_cluster] = "street_circuit"
    remaining = [c for c in remaining if c != street_cluster]

    # 2. high_degradation: highest (three_stop_pct - one_stop_pct)
    #    Circuits that chew through tires force multi-stop strategies.
    #    Subtracting one_stop_pct avoids picking up circuits that happen to
    #    have high strategy_entropy but are 1-stop dominated.
    deg_score = centroid_df["three_stop_pct"] - centroid_df["one_stop_pct"]
    hd_cluster = int(deg_score.iloc[remaining].idxmax())
    assigned[hd_cluster] = "high_degradation"
    remaining = [c for c in remaining if c != hd_cluster]

    # 3. high_overtaking: highest two_stop_pct × strategy_entropy
    #    Open circuits where overtaking is feasible → aggressive 2-stop strategies
    #    dominate and the field adopts a varied mix of approaches.
    ot_score = centroid_df["two_stop_pct"] * centroid_df["strategy_entropy"]
    ot_cluster = int(ot_score.iloc[remaining].idxmax())
    assigned[ot_cluster] = "high_overtaking"
    remaining = [c for c in remaining if c != ot_cluster]

    # 4. balanced: whatever cluster remains
    if remaining:
        assigned[remaining[0]] = "balanced"

    # Safety fallback (shouldn't trigger with k=4)
    for cid in range(k):
        if cid not in assigned:
            assigned[cid] = ARCHETYPE_LABELS[cid % len(ARCHETYPE_LABELS)]

    df["archetype_label"] = df["archetype_id"].map(assigned)
    return df


# ---------------------------------------------------------------------------
# Orchestrator: run
# ---------------------------------------------------------------------------

def run(save: bool = True):
    """
    Full pipeline: load data → build fingerprints → cluster → optionally save.

    Returns
    -------
    (fingerprint_df, archetypes_dict)
    """
    # Load data
    print("Loading data...")
    raw_df = pd.read_csv(DATA_CSV)

    with open(PROFILES_JSON) as f:
        profiles = json.load(f)
    with open(BASELINES_JSON) as f:
        baselines = json.load(f)
    with open(PIT_LOSS_JSON) as f:
        pit_losses = json.load(f)

    # Build fingerprints
    print("Building 18-feature fingerprints for 24 circuits...")
    fp_df = build_fingerprints(raw_df, profiles, baselines, pit_losses)

    # Cluster
    print("Running KMeans (k=4) clustering...")
    fp_df = cluster_circuits(fp_df, k=4)

    # Sort by circuit name (index)
    fp_df = fp_df.sort_index()

    # Build archetypes dict
    archetypes_map: dict[str, list[str]] = {label: [] for label in ARCHETYPE_LABELS}
    for circuit, row in fp_df.iterrows():
        archetypes_map[row["archetype_label"]].append(circuit)
    for label in archetypes_map:
        archetypes_map[label] = sorted(archetypes_map[label])

    archetypes_dict = {
        "archetypes": archetypes_map,
        "feature_names": FEATURE_NAMES,
        "n_circuits": len(fp_df),
        "kmeans_k": 4,
    }

    # Save outputs
    if save:
        out_dir = OUT_CSV.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        # CSV: reset index so 'circuit' becomes a column
        out_df = fp_df.reset_index()
        # Reorder: circuit, then features, then archetype cols
        col_order = ["circuit"] + FEATURE_NAMES + ["archetype_id", "archetype_label"]
        out_df = out_df[col_order]
        out_df.to_csv(OUT_CSV, index=False)
        print(f"Saved: {OUT_CSV}")

        with open(OUT_JSON, "w") as f:
            json.dump(archetypes_dict, f, indent=2)
        print(f"Saved: {OUT_JSON}")

    return fp_df, archetypes_dict


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------

def _verify(fp_df: pd.DataFrame, archetypes_dict: dict) -> bool:
    """Run assertions and print results. Returns True if all pass."""
    all_ok = True

    # 1. Exactly 24 rows
    n_rows = len(fp_df)
    ok = n_rows == 24
    all_ok &= ok
    print(f"  [{'OK' if ok else 'FAIL'}] Row count: {n_rows} (expected 24)")

    # 2. All 18 feature columns present
    missing_features = [f for f in FEATURE_NAMES if f not in fp_df.columns]
    ok = len(missing_features) == 0
    all_ok &= ok
    print(f"  [{'OK' if ok else 'FAIL'}] All 18 features present"
          + (f" — missing: {missing_features}" if missing_features else ""))

    # 3. No NaN values
    nan_counts = fp_df[FEATURE_NAMES].isna().sum()
    has_nan = nan_counts.sum() > 0
    ok = not has_nan
    all_ok &= ok
    if has_nan:
        print(f"  [FAIL] NaN values found:\n{nan_counts[nan_counts > 0]}")
    else:
        print("  [OK]   No NaN values in feature columns")

    # 4. archetype_label is one of 4 defined labels
    bad_labels = fp_df[~fp_df["archetype_label"].isin(ARCHETYPE_LABELS)]
    ok = len(bad_labels) == 0
    all_ok &= ok
    print(f"  [{'OK' if ok else 'FAIL'}] All archetype labels valid"
          + (f" — bad: {bad_labels['archetype_label'].unique()}" if len(bad_labels) else ""))

    # 5. JSON valid and all 24 circuits appear in exactly one archetype
    all_in_json = []
    for label, circuits in archetypes_dict["archetypes"].items():
        all_in_json.extend(circuits)
    ok_count = len(all_in_json) == 24
    ok_unique = len(set(all_in_json)) == 24
    ok = ok_count and ok_unique
    all_ok &= ok
    print(f"  [{'OK' if ok else 'FAIL'}] circuit_archetypes.json: "
          f"{len(all_in_json)} total entries, {len(set(all_in_json))} unique circuits")

    return all_ok


def _print_archetype_table(fp_df: pd.DataFrame) -> None:
    """Print a formatted table of archetype assignments."""
    print("\nArchetype Assignments:")
    print(f"{'Circuit':<35} {'Archetype':<20} {'ClusterID'}")
    print("-" * 65)
    for circuit in sorted(fp_df.index):
        row = fp_df.loc[circuit]
        print(f"{circuit:<35} {row['archetype_label']:<20} {int(row['archetype_id'])}")

    print("\nSummary by archetype:")
    for label in ARCHETYPE_LABELS:
        members = sorted(fp_df[fp_df["archetype_label"] == label].index.tolist())
        print(f"  {label} ({len(members)}): {', '.join(members)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    fp_df, archetypes_dict = run(save=True)

    print("\n--- Verification ---")
    all_ok = _verify(fp_df, archetypes_dict)
    print(f"\nAll checks {'PASSED' if all_ok else 'FAILED'}")

    _print_archetype_table(fp_df)
