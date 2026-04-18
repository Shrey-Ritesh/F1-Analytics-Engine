"""
circuit_dna.py — Phase 2 circuit fingerprinting and archetype clustering
========================================================================
Builds an 18-feature fingerprint for each of the 24 F1 circuits and
assigns one of 4 archetypes via KMeans clustering.

Archetypes: street_circuit | high_degradation | power_circuit | balanced

Outputs:
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
    "compound_diversity",
]

ARCHETYPE_LABELS = ["street_circuit", "high_degradation", "power_circuit", "balanced"]


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
        compound_diversity = min(len(top_compounds), 3)

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
            "compound_diversity": compound_diversity,
        })

    result = pd.DataFrame(rows).set_index("circuit")
    return result


# ---------------------------------------------------------------------------
# Core function: cluster_circuits
# ---------------------------------------------------------------------------

def cluster_circuits(fingerprint_df: pd.DataFrame, k: int = 4) -> pd.DataFrame:
    """
    Run KMeans clustering on the fingerprint DataFrame and assign archetype labels.

    Parameters
    ----------
    fingerprint_df : DataFrame with FEATURE_NAMES columns (index = circuit name)
    k              : number of clusters (default 4)

    Returns
    -------
    fingerprint_df with two additional columns: archetype_id, archetype_label
    """
    df = fingerprint_df.copy()

    # Normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURE_NAMES].values)

    # KMeans
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    df["archetype_id"] = labels

    # Map cluster id → archetype label using centroids (in original feature space)
    centroids = scaler.inverse_transform(km.cluster_centers_)
    centroid_df = pd.DataFrame(centroids, columns=FEATURE_NAMES)

    assigned: dict[int, str] = {}
    remaining_clusters = list(range(k))

    # Rule 1: street_circuit — highest overtaking_difficulty AND lowest one_stop_pct
    #   Score = overtaking_difficulty rank (desc) + (1 - one_stop_pct rank asc)
    od_rank = centroid_df["overtaking_difficulty"].rank(ascending=False)
    os_rank = centroid_df["one_stop_pct"].rank(ascending=True)   # lower one_stop → higher rank
    street_score = od_rank + os_rank
    street_cluster = int(street_score.idxmin())   # lowest combined rank = best street match
    assigned[street_cluster] = "street_circuit"
    remaining_clusters = [c for c in remaining_clusters if c != street_cluster]

    # Rule 2: high_degradation — highest soft_deg_rate + medium_deg_rate
    deg_sum = centroid_df["soft_deg_rate"] + centroid_df["medium_deg_rate"]
    # Only consider remaining clusters
    best_deg = deg_sum.iloc[remaining_clusters].idxmax()
    hd_cluster = int(best_deg)
    assigned[hd_cluster] = "high_degradation"
    remaining_clusters = [c for c in remaining_clusters if c != hd_cluster]

    # Rule 3: power_circuit — lowest baseline_lap_time + high lap_time_std
    #   Score = baseline_lap_time rank (asc, lower is better) + lap_time_std rank (desc)
    blt_rank = centroid_df["baseline_lap_time"].rank(ascending=True)
    lts_rank = centroid_df["lap_time_std"].rank(ascending=False)
    power_score = blt_rank + lts_rank
    best_power = power_score.iloc[remaining_clusters].idxmin()
    pc_cluster = int(best_power)
    assigned[pc_cluster] = "power_circuit"
    remaining_clusters = [c for c in remaining_clusters if c != pc_cluster]

    # Rule 4: balanced — whatever is left
    if remaining_clusters:
        assigned[remaining_clusters[0]] = "balanced"

    # Safety fallback: if any cluster unassigned (shouldn't happen with k=4)
    for cid in range(k):
        if cid not in assigned:
            assigned[cid] = ARCHETYPE_LABELS[cid]

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
