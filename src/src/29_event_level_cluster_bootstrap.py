from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "reviewer_response_v3"
OUT.mkdir(parents=True, exist_ok=True)
REPEATS = 1000
BOOTSTRAP_SEED = 20260916

with open(ROOT / "data/windows_v1/model_feature_columns.txt", encoding="utf-8") as handle:
    FEATURES = [line.strip() for line in handle if line.strip()]


def load(path):
    frame = pd.read_csv(path, low_memory=False)
    for column in [
        "Window_End_Time", "Scenario_Start_Time", "Scenario_End_Time",
        "Scenario_Attack_Start_Time",
    ]:
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def matrix_sets():
    guam = load(ROOT / "data/windows_v1/test_window_samples.csv")
    la = load(ROOT / "external_validation/la_2025/windows_v2/test_window_samples.csv")
    result = {}
    for domain, frame, mapping in [
        ("Guam", guam, {"A": "A", "B": "C"}),
        ("LA", la, {"A": "B", "B": "D"}),
    ]:
        normal = frame[frame["Attack_Type"].eq("normal")]
        for generator, experiment in mapping.items():
            result[experiment] = pd.concat(
                [normal, frame[frame["Generator"].eq(generator)]],
                ignore_index=True,
            )
    return result


def score(model, frame):
    values = frame[FEATURES].replace([np.inf, -np.inf], np.nan)
    return model.predict_proba(values)[:, 1]


def alerts(subset):
    ordered = subset.sort_values("Window_Start_Index").reset_index(drop=True)
    result = []
    current = None
    for ordinal, row in ordered.iterrows():
        if int(row["Prediction"]) != 1:
            if current is not None:
                result.append(current)
                current = None
            continue
        record = {
            "start": int(row["Window_Start_Index"]),
            "end": int(row["Window_End_Index"]),
            "decision_time": row["Window_End_Time"],
        }
        if current is None or ordinal != current["last_ordinal"] + 1:
            if current is not None:
                result.append(current)
            current = {
                "start_time": record["decision_time"],
                "decision_index": record["end"],
                "last_ordinal": ordinal,
                "windows": [record],
            }
        else:
            current["last_ordinal"] = ordinal
            current["windows"].append(record)
    if current is not None:
        result.append(current)
    return result


def components_by_mmsi(frame):
    rows = []
    for mmsi, vessel in frame.groupby("MMSI", sort=False):
        comp = {
            "MMSI": mmsi,
            "Alert_Count": 0,
            "Matched_Alert_Count": 0,
            "False_Alert_Count": 0,
            "Attack_Count": 0,
            "Detected_Attack_Count": 0,
            "Exposure_Hours": 0.0,
            "Delays": [],
        }
        for _, scenario in vessel.groupby("Scenario_ID", sort=False):
            scenario_alerts = alerts(scenario)
            comp["Alert_Count"] += len(scenario_alerts)
            start = scenario["Scenario_Start_Time"].iloc[0]
            end = scenario["Scenario_End_Time"].iloc[0]
            if pd.notna(start) and pd.notna(end):
                comp["Exposure_Hours"] += max(
                    0.0, (end - start).total_seconds() / 3600.0
                )
            if str(scenario["Attack_Type"].iloc[0]) == "normal":
                comp["False_Alert_Count"] += len(scenario_alerts)
                continue

            attack_start = int(scenario["Scenario_Attack_Start_Index"].iloc[0])
            attack_end = int(scenario["Scenario_Attack_End_Index"].iloc[0])
            attack_start_time = scenario["Scenario_Attack_Start_Time"].iloc[0]
            comp["Attack_Count"] += 1
            matched = []
            for alert in scenario_alerts:
                causal = (
                    pd.notna(alert["start_time"])
                    and pd.notna(attack_start_time)
                    and alert["start_time"] >= attack_start_time
                )
                overlap = any(
                    window["end"] >= attack_start
                    and window["start"] <= attack_end
                    for window in alert["windows"]
                )
                if causal and overlap:
                    matched.append(alert)
            comp["Matched_Alert_Count"] += len(matched)
            comp["False_Alert_Count"] += len(scenario_alerts) - len(matched)
            if matched:
                comp["Detected_Attack_Count"] += 1
                first = min(matched, key=lambda item: item["start_time"])
                comp["Delays"].append(
                    (first["start_time"] - attack_start_time).total_seconds()
                    / 60.0
                )
        rows.append(comp)
    return rows


def aggregate(components, indices=None):
    selected = components if indices is None else [components[i] for i in indices]
    alert_count = sum(row["Alert_Count"] for row in selected)
    matched_alerts = sum(row["Matched_Alert_Count"] for row in selected)
    false_alerts = sum(row["False_Alert_Count"] for row in selected)
    attack_count = sum(row["Attack_Count"] for row in selected)
    detected = sum(row["Detected_Attack_Count"] for row in selected)
    exposure = sum(row["Exposure_Hours"] for row in selected)
    delays = [delay for row in selected for delay in row["Delays"]]
    return {
        "Event_Level_Precision": matched_alerts / alert_count if alert_count else 0.0,
        "Event_Level_Recall": detected / attack_count if attack_count else np.nan,
        "Median_Detection_Delay_Minutes": float(np.median(delays)) if delays else np.nan,
        "P95_Detection_Delay_Minutes": float(np.percentile(delays, 95)) if delays else np.nan,
        "False_Alerts_Per_100_Synthetic_Scenario_Hours": (
            false_alerts / exposure * 100 if exposure else np.nan
        ),
    }


model = joblib.load(ROOT / "models_v1/HistGradientBoosting_pipeline.joblib")
thresholds = pd.read_csv(ROOT / "results/models_v1/guam_validation_fixed_far_thresholds.csv")
threshold = float(thresholds[
    thresholds["Model"].eq("HistGradientBoosting")
    & np.isclose(thresholds["Target_FAR"], 0.03)
]["Frozen_Threshold"].iloc[0])

rng = np.random.default_rng(BOOTSTRAP_SEED)
summary = []
distribution = []
for experiment, frame in matrix_sets().items():
    frame = frame.copy()
    frame["Prediction"] = (score(model, frame) >= threshold).astype(int)
    components = components_by_mmsi(frame)
    point = aggregate(components)
    draws = {metric: [] for metric in point}
    for repeat in range(REPEATS):
        indices = rng.integers(0, len(components), size=len(components))
        metrics = aggregate(components, indices)
        for metric, value in metrics.items():
            draws[metric].append(value)
            distribution.append({
                "Experiment": experiment,
                "Bootstrap_Repeat": repeat,
                "Metric": metric,
                "Value": value,
            })
    for metric, values in draws.items():
        summary.append({
            "Experiment": experiment,
            "Metric": metric,
            "Point_Estimate": point[metric],
            "Bootstrap_Mean": float(np.nanmean(values)),
            "CI95_Lower": float(np.nanpercentile(values, 2.5)),
            "CI95_Upper": float(np.nanpercentile(values, 97.5)),
            "MMSI_Count": len(components),
            "Bootstrap_Repeats": REPEATS,
            "Bootstrap_Seed": BOOTSTRAP_SEED,
            "Cluster_Unit": "MMSI",
        })

pd.DataFrame(summary).to_csv(
    OUT / "table_event_level_cluster_bootstrap_ci.csv",
    index=False,
    encoding="utf-8-sig",
)
pd.DataFrame(distribution).to_csv(
    OUT / "event_level_cluster_bootstrap_distribution.csv",
    index=False,
    encoding="utf-8-sig",
)
print(pd.DataFrame(summary).to_string(index=False))
