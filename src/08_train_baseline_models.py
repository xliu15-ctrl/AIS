from pathlib import Path
import json
import platform

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Compatibility for pandas 2.x with the older bundled Matplotlib.
from matplotlib import cbook
cbook._check_1d = lambda values: np.atleast_1d(np.asarray(values))
import matplotlib.axes._base as _axes_base
_axes_base._check_1d = cbook._check_1d

from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import joblib
import sklearn


# ============================================================
# 1. Paths and experiment protocol
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = PROJECT_ROOT / "data" / "windows_v1"
OUTPUT_DIR = PROJECT_ROOT / "results" / "models_v1"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures" / "models_v1"
MODEL_DIR = PROJECT_ROOT / "models_v1"
CONFIG_DIR = PROJECT_ROOT / "config"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = INPUT_DIR / "train_window_samples.csv"
VALIDATION_FILE = INPUT_DIR / "validation_window_samples.csv"
TEST_FILE = INPUT_DIR / "test_window_samples.csv"
FEATURE_FILE = INPUT_DIR / "model_feature_columns.txt"
PROTOCOL_FILE = CONFIG_DIR / "experiment_protocol.json"

LA_CANDIDATE_FILES = [
    PROJECT_ROOT
    / "external_validation"
    / "la_2025"
    / "windows_v2"
    / "test_window_samples.csv",
    INPUT_DIR / "la_window_samples.csv",
    PROJECT_ROOT / "data" / "windows_la" / "test_window_samples.csv",
    PROJECT_ROOT / "data" / "windows_la" / "la_window_samples.csv",
    PROJECT_ROOT
    / "external_validation"
    / "la_2025"
    / "windows_v1"
    / "la_2025_external_window_samples.csv"
]

with open(PROTOCOL_FILE, "r", encoding="utf-8") as file:
    PROTOCOL = json.load(file)

RANDOM_SEED = PROTOCOL["modeling"]["main_random_seed"]
TARGET_FARS = PROTOCOL["threshold_protocol"]["target_far_values"]
BOOTSTRAP_REPEATS = 1000
BOOTSTRAP_RANDOM_SEED = 20260916


# ============================================================
# 2. Load data
# ============================================================
required_files = [
    TRAIN_FILE,
    VALIDATION_FILE,
    TEST_FILE,
    FEATURE_FILE
]

for file_path in required_files:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Missing file: {file_path}\n"
            "Run 07_build_sliding_window_dataset.py first."
        )

train_data = pd.read_csv(TRAIN_FILE, low_memory=False)
validation_data = pd.read_csv(VALIDATION_FILE, low_memory=False)
test_data = pd.read_csv(TEST_FILE, low_memory=False)

with open(FEATURE_FILE, "r", encoding="utf-8") as file:
    feature_columns = [
        line.strip()
        for line in file.readlines()
        if line.strip()
    ]

la_data = None
la_file_used = None

for file_path in LA_CANDIDATE_FILES:
    if file_path.exists():
        la_data = pd.read_csv(file_path, low_memory=False)
        la_file_used = file_path
        break

print("=" * 90)
print("Stage 08: train models and evaluate frozen Guam-validation FAR thresholds")
print(f"Train windows: {len(train_data)}")
print("Model training uses only normal + Generator A windows.")
print(f"Guam validation windows: {len(validation_data)}")
print(f"Guam test windows: {len(test_data)}")
print(f"Feature count: {len(feature_columns)}")
if la_data is None:
    print("LA window file not found; LA evaluation will be skipped.")
else:
    print(f"LA windows: {len(la_data)} from {la_file_used}")
print("=" * 90)


# ============================================================
# 3. Shared utilities
# ============================================================
def prepare_xy(data, feature_columns):
    X = data[feature_columns].copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    y = pd.to_numeric(
        data["Window_Label"],
        errors="coerce"
    ).fillna(0).astype(int)

    return X, y


def calibrate_threshold_at_far(normal_scores, target_far):
    normal_scores = pd.Series(normal_scores).replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna().to_numpy()

    if len(normal_scores) == 0:
        raise ValueError("No Guam validation label-zero scores (nominal plus Generator-A/B negative context) for FAR calibration.")

    threshold = float(
        np.quantile(
            normal_scores,
            1.0 - float(target_far),
            method="higher"
        )
    )

    achieved_far = float((normal_scores >= threshold).mean())

    return threshold, achieved_far


def calculate_metrics(y_true, y_pred, scores):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    scores = np.asarray(scores)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    far = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    miss_rate = fn / (fn + tp) if (fn + tp) > 0 else np.nan

    try:
        auprc = average_precision_score(y_true, scores)
    except ValueError:
        auprc = np.nan

    return {
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "AUPRC": auprc,
        "FAR": far,
        "Miss_Rate": miss_rate,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    }


def add_default_metadata_columns(data):
    data = data.copy()

    if "Generator" not in data.columns:
        data["Generator"] = np.where(
            data["Attack_Type"].eq("normal"),
            "none",
            "A"
        )

    if "Attack_Overlap_Ratio" not in data.columns:
        data["Attack_Overlap_Ratio"] = data["Window_Label"].astype(float)

    if "Attack_Point_Count" not in data.columns:
        data["Attack_Point_Count"] = data["Window_Label"].astype(int)

    if "Scenario_Attack_Start_Index" not in data.columns:
        data["Scenario_Attack_Start_Index"] = np.nan

    if "Scenario_Attack_End_Index" not in data.columns:
        data["Scenario_Attack_End_Index"] = np.nan

    return data


train_data = add_default_metadata_columns(train_data)
validation_data = add_default_metadata_columns(validation_data)
test_data = add_default_metadata_columns(test_data)

if la_data is not None:
    la_data = add_default_metadata_columns(la_data)


train_model_data = train_data[
    (train_data["Attack_Type"] == "normal")
    | (train_data["Generator"] == "A")
].copy()

X_train, y_train = prepare_xy(train_model_data, feature_columns)
X_validation, y_validation = prepare_xy(validation_data, feature_columns)
X_test, y_test = prepare_xy(test_data, feature_columns)

def build_evaluation_sets(domain_name, data):
    sets = {}

    X_all, y_all = prepare_xy(data, feature_columns)
    sets[f"{domain_name} test"] = (data, X_all, y_all)

    normal_rows = data[data["Attack_Type"] == "normal"].copy()

    experiment_ids = (
        {"A": "A", "B": "C"}
        if domain_name == "Guam"
        else {"A": "B", "B": "D"}
    )

    for generator_name in ["A", "B"]:
        experiment_id = experiment_ids[generator_name]
        subset = pd.concat(
            [
                normal_rows,
                data[data["Generator"] == generator_name]
            ],
            ignore_index=True
        )

        if len(subset) == len(normal_rows):
            continue

        X_subset, y_subset = prepare_xy(subset, feature_columns)
        sets[
            f"{domain_name} experiment {experiment_id} Generator {generator_name}"
        ] = (subset, X_subset, y_subset)

    return sets


evaluation_sets = build_evaluation_sets("Guam", test_data)

if la_data is not None:
    evaluation_sets.update(build_evaluation_sets("LA", la_data))


def build_models(seed):
    return {
        "LogisticRegression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=seed
                    )
                )
            ]
        ),
        "LinearSVM": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LinearSVC(
                        class_weight="balanced",
                        random_state=seed,
                        max_iter=20000
                    )
                )
            ]
        ),
        "RandomForest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=-1
                    )
                )
            ]
        ),
        "ExtraTrees": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=400,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=-1
                    )
                )
            ]
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_leaf_nodes=31,
            l2_regularization=0.01,
            class_weight="balanced",
            random_state=seed
        )
    }


def get_abnormal_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    if hasattr(model, "decision_function"):
        return model.decision_function(X)

    raise TypeError(f"Model {type(model)} cannot produce anomaly scores.")


def summarize_per_attack_type(model_name, evaluation_name, data, y_pred):
    rows = []
    result = data.copy()
    result["Prediction"] = np.asarray(y_pred).astype(int)

    for (generator_name, attack_type), subset in result.groupby(
        ["Generator", "Attack_Type"]
    ):
        y_true_attack = subset["Window_Label"].astype(int)
        y_pred_attack = subset["Prediction"].astype(int)

        metrics = calculate_metrics(
            y_true_attack,
            y_pred_attack,
            np.zeros(len(subset))
        )

        rows.append(
            {
                "Model": model_name,
                "Evaluation_Set": evaluation_name,
                "Generator": generator_name,
                "Attack_Type": attack_type,
                "Sample_Count": len(subset),
                "Recall": metrics["Recall"],
                "F1": metrics["F1"],
                "FAR": metrics["FAR"],
                "Miss_Rate": metrics["Miss_Rate"],
                "Predicted_Abnormal_Count": int(y_pred_attack.sum())
            }
        )

    return rows


def build_alert_events(subset):
    ordered = subset.sort_values("Window_Start_Index").reset_index(drop=True)
    events = []
    current = None

    # Causal protocol: an alert is emitted only after the complete window has
    # been observed. Therefore its decision time is Window_End_Time. Alert
    # events contain strictly consecutive positive windows; one intervening
    # negative window closes the current event (merge gap = 0 windows).
    for ordinal, row in ordered.iterrows():
        if int(row["Prediction"]) != 1:
            if current is not None:
                events.append(current)
                current = None
            continue

        start = int(row["Window_Start_Index"])
        end = int(row["Window_End_Index"])
        decision_time = pd.to_datetime(
            row.get("Window_End_Time", pd.NaT), errors="coerce"
        )
        window_record = {
            "start": start,
            "end": end,
            "decision_time": decision_time
        }

        if current is None:
            current = {
                "start": start,
                "end": end,
                "start_time": decision_time,
                "end_time": decision_time,
                "decision_index": end,
                "last_ordinal": ordinal,
                "count": 1,
                "windows": [window_record]
            }
            continue

        if ordinal == current["last_ordinal"] + 1:
            current["end"] = max(current["end"], end)
            if pd.notna(decision_time):
                if pd.isna(current["end_time"]):
                    current["end_time"] = decision_time
                else:
                    current["end_time"] = max(
                        current["end_time"], decision_time
                    )
            current["last_ordinal"] = ordinal
            current["count"] += 1
            current["windows"].append(window_record)
        else:
            events.append(current)
            current = {
                "start": start,
                "end": end,
                "start_time": decision_time,
                "end_time": decision_time,
                "decision_index": end,
                "last_ordinal": ordinal,
                "count": 1,
                "windows": [window_record]
            }

    if current is not None:
        events.append(current)

    return events


def calculate_event_metrics(data, predictions):
    result = data.copy()
    result["Prediction"] = np.asarray(predictions).astype(int)

    attack_event_count = 0
    matched_attack_count = 0
    alert_count = 0
    matched_alert_event_count = 0
    false_alert_count = 0
    detection_delays = []
    alert_durations = []
    alert_durations_minutes = []
    repeated_alerts = []
    trajectory_count = result["Scenario_ID"].nunique()

    for _, subset in result.groupby("Scenario_ID"):
        subset = subset.sort_values("Window_Start_Index")
        attack_type = str(subset["Attack_Type"].iloc[0])

        alerts = build_alert_events(subset)
        alert_count += len(alerts)

        for alert in alerts:
            alert_durations.append(alert["end"] - alert["start"] + 1)
            if pd.notna(alert["start_time"]) and pd.notna(alert["end_time"]):
                alert_durations_minutes.append(
                    max(
                        0.0,
                        (
                            alert["end_time"] - alert["start_time"]
                        ).total_seconds() / 60.0
                    )
                )

        if attack_type == "normal":
            false_alert_count += len(alerts)
            continue

        attack_start = pd.to_numeric(
            subset["Scenario_Attack_Start_Index"],
            errors="coerce"
        ).dropna()
        attack_end = pd.to_numeric(
            subset["Scenario_Attack_End_Index"],
            errors="coerce"
        ).dropna()

        if len(attack_start) == 0 or len(attack_end) == 0:
            continue

        attack_start = int(attack_start.iloc[0])
        attack_end = int(attack_end.iloc[0])

        attack_event_count += 1

        attack_start_time = pd.to_datetime(
            subset.get("Scenario_Attack_Start_Time", pd.Series([pd.NaT])).iloc[0],
            errors="coerce"
        )

        matched_alerts = []
        for alert in alerts:
            # A pre-existing alert is a false alert, not an attack detection.
            causal_start = (
                pd.notna(alert["start_time"])
                and pd.notna(attack_start_time)
                and alert["start_time"] >= attack_start_time
            ) or (
                pd.isna(attack_start_time)
                and alert["decision_index"] >= attack_start
            )
            evidence_overlap = any(
                window["end"] >= attack_start
                and window["start"] <= attack_end
                for window in alert["windows"]
            )
            if causal_start and evidence_overlap:
                matched_alerts.append(alert)

        if matched_alerts:
            matched_attack_count += 1
            matched_alert_event_count += len(matched_alerts)
            first_alert = min(
                matched_alerts,
                key=lambda item: item["start_time"]
                if pd.notna(item["start_time"])
                else pd.Timestamp.max
            )
            if (
                "Scenario_Attack_Start_Time" in subset.columns
                and "Window_End_Time" in subset.columns
            ):
                first_alert_time = first_alert["start_time"]
                delay_minutes = (
                    (first_alert_time - attack_start_time).total_seconds()
                    / 60.0
                    if pd.notna(attack_start_time) and pd.notna(first_alert_time)
                    else np.nan
                )
            else:
                delay_minutes = float(
                    first_alert["decision_index"] - attack_start
                )
            detection_delays.append(delay_minutes)
            repeated_alerts.append(max(0, len(matched_alerts) - 1))

        false_alert_count += len(alerts) - len(matched_alerts)

    if {"Scenario_Start_Time", "Scenario_End_Time"}.issubset(result.columns):
        exposure = result.groupby("Scenario_ID", as_index=False).agg(
            Scenario_Start_Time=("Scenario_Start_Time", "first"),
            Scenario_End_Time=("Scenario_End_Time", "first")
        )
        exposure["Scenario_Start_Time"] = pd.to_datetime(
            exposure["Scenario_Start_Time"], errors="coerce"
        )
        exposure["Scenario_End_Time"] = pd.to_datetime(
            exposure["Scenario_End_Time"], errors="coerce"
        )
        total_window_hours = (
            (
                exposure["Scenario_End_Time"]
                - exposure["Scenario_Start_Time"]
            ).dt.total_seconds().clip(lower=0).sum()
            / 3600.0
        )
        exposure_method = "sum of synthetic scenario time spans (scenario-hours)"
    else:
        total_window_hours = (
            result.groupby("Scenario_ID")["Window_Duration_Minutes"].max().sum()
            / 60.0
        )
        exposure_method = "legacy maximum window duration per scenario"

    false_alerts_per_100_vessel_hours = (
        false_alert_count / total_window_hours * 100
        if total_window_hours > 0
        else np.nan
    )

    return {
        "Event_Level_Precision": (
            matched_alert_event_count / alert_count
            if alert_count > 0
            else 0.0
        ),
        "Event_Level_Recall": (
            matched_attack_count / attack_event_count
            if attack_event_count > 0
            else np.nan
        ),
        "Median_Detection_Delay_Minutes": (
            float(np.median(detection_delays))
            if detection_delays
            else np.nan
        ),
        "P95_Detection_Delay_Minutes": (
            float(np.percentile(detection_delays, 95))
            if detection_delays
            else np.nan
        ),
        "False_Alerts_Per_100_Synthetic_Scenario_Hours": (
            false_alerts_per_100_vessel_hours
        ),
        "Alerts_Per_Trajectory": (
            alert_count / trajectory_count
            if trajectory_count > 0
            else np.nan
        ),
        "Mean_Alert_Duration_Points": (
            float(np.mean(alert_durations))
            if alert_durations
            else 0.0
        ),
        "Mean_Alert_Duration_Minutes": (
            float(np.mean(alert_durations_minutes))
            if alert_durations_minutes
            else np.nan
        ),
        "Repeated_Alerts_Per_Attack_Event": (
            float(np.mean(repeated_alerts))
            if repeated_alerts
            else 0.0
        ),
        "Alert_Event_Count": int(alert_count),
        "Matched_Alert_Event_Count": int(matched_alert_event_count),
        "False_Alert_Event_Count": int(false_alert_count),
        "Attack_Event_Count": int(attack_event_count),
        "Detected_Attack_Event_Count": int(matched_attack_count),
        "Missed_Attack_Event_Count": int(
            attack_event_count - matched_attack_count
        ),
        "Repeated_Matched_Alert_Count": int(sum(repeated_alerts)),
        "Exposure_Hours": float(total_window_hours),
        "Exposure_Method": exposure_method,
        "Alert_Decision_Time": "Window_End_Time",
        "Alert_Merge_Gap_Windows": 0,
        "Attack_Match_Rule": (
            "alert starts at/after attack start and contains a positive "
            "window whose evidence interval overlaps [s,e]"
        ),
        "Precision_Unit": "matched alert events / all alert events",
        "Recall_Unit": "unique detected attack intervals / all attack intervals",
        "Pre_Attack_Alert_Handling": "unmatched false alert",
        "Negative_Delay_Clipping": False
    }


# ============================================================
# 4. Train models and apply frozen thresholds
# ============================================================
models = build_models(RANDOM_SEED)

threshold_rows = []
metrics_rows = []
per_attack_rows = []
event_rows = []
model_scores_for_curves = {}
trained_models = {}

for model_name, model in models.items():
    print(f"Training {model_name}")
    model.fit(X_train, y_train)
    trained_models[model_name] = model

    validation_scores = get_abnormal_scores(model, X_validation)
    validation_normal_scores = validation_scores[y_validation.to_numpy() == 0]

    for target_far in TARGET_FARS:
        threshold, validation_actual_far = calibrate_threshold_at_far(
            validation_normal_scores,
            target_far
        )

        threshold_rows.append(
            {
                "Model": model_name,
                "Threshold_Source": (
                    f"Guam validation normal FAR={target_far:.0%}"
                ),
                "Target_FAR": target_far,
                "Frozen_Threshold": threshold,
                "Validation_Normal_Actual_FAR": validation_actual_far
            }
        )

        for evaluation_name, (eval_data, X_eval, y_eval) in evaluation_sets.items():
            eval_scores = get_abnormal_scores(model, X_eval)
            eval_predictions = (eval_scores >= threshold).astype(int)

            metrics = calculate_metrics(y_eval, eval_predictions, eval_scores)

            metrics_rows.append(
                {
                    "Model": model_name,
                    "Threshold_Source": (
                        f"Guam validation FAR={target_far:.0%}"
                    ),
                    "Evaluation_Set": evaluation_name,
                    "Target_FAR": target_far,
                    "Actual_FAR": metrics["FAR"],
                    "Recall_under_threshold_calibrated_to_target_FAR_on_Guam_validation": metrics["Recall"],
                    **metrics
                }
            )

            if target_far == 0.03:
                per_attack_rows.extend(
                    summarize_per_attack_type(
                        model_name,
                        evaluation_name,
                        eval_data,
                        eval_predictions
                    )
                )

                event_metrics = calculate_event_metrics(
                    eval_data,
                    eval_predictions
                )
                event_metrics.update(
                    {
                        "Model": model_name,
                        "Evaluation_Set": evaluation_name,
                        "Target_FAR": target_far,
                        "Experiment_Name": (
                            "Low-prevalence and operating-point stress analysis"
                        )
                    }
                )
                event_rows.append(event_metrics)

                if evaluation_name == "Guam test":
                    model_scores_for_curves[model_name] = eval_scores

    joblib.dump(model, MODEL_DIR / f"{model_name}_pipeline.joblib")


threshold_table = pd.DataFrame(threshold_rows)
metrics_table = pd.DataFrame(metrics_rows)
per_attack_table = pd.DataFrame(per_attack_rows)
event_table = pd.DataFrame(event_rows)

threshold_file = OUTPUT_DIR / "guam_validation_fixed_far_thresholds.csv"
metrics_file = OUTPUT_DIR / "fixed_far_threshold_transfer_metrics.csv"
legacy_metrics_file = OUTPUT_DIR / "baseline_model_test_metrics.csv"
per_attack_file = OUTPUT_DIR / "per_attack_type_test_metrics.csv"
event_file = OUTPUT_DIR / "event_level_operating_point_metrics.csv"

threshold_table.to_csv(threshold_file, index=False, encoding="utf-8-sig")
metrics_table.to_csv(metrics_file, index=False, encoding="utf-8-sig")
metrics_table.to_csv(legacy_metrics_file, index=False, encoding="utf-8-sig")
per_attack_table.to_csv(per_attack_file, index=False, encoding="utf-8-sig")
event_table.to_csv(event_file, index=False, encoding="utf-8-sig")

print("Frozen-FAR threshold transfer metrics:")
print(
    metrics_table[
        [
            "Model",
            "Threshold_Source",
            "Evaluation_Set",
            "Actual_FAR",
            "Recall_under_threshold_calibrated_to_target_FAR_on_Guam_validation",
            "F1",
            "AUPRC",
            "Miss_Rate"
        ]
    ].to_string(index=False)
)
print(f"Threshold table saved to: {threshold_file}")
print(f"Main metrics saved to: {metrics_file}")
print(f"Per-attack metrics saved to: {per_attack_file}")
print(f"Event-level metrics saved to: {event_file}")


# ============================================================
# 5. MMSI-level cluster bootstrap for HGB vs RF
# ============================================================
def bootstrap_metric_arrays(
    y_true,
    scores,
    threshold,
    cluster_codes,
    bootstrap_cluster_counts,
    batch_size=25
):
    """Compute exact weighted bootstrap metrics after sorting scores once."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    cluster_codes = np.asarray(cluster_codes).astype(int)
    predictions = scores >= threshold
    cluster_count = bootstrap_cluster_counts.shape[1]

    def cluster_sum(mask):
        return np.bincount(
            cluster_codes,
            weights=np.asarray(mask, dtype=float),
            minlength=cluster_count
        )

    true_positive_by_cluster = cluster_sum((y_true == 1) & predictions)
    false_positive_by_cluster = cluster_sum((y_true == 0) & predictions)
    false_negative_by_cluster = cluster_sum((y_true == 1) & ~predictions)
    true_negative_by_cluster = cluster_sum((y_true == 0) & ~predictions)

    tp = bootstrap_cluster_counts @ true_positive_by_cluster
    fp = bootstrap_cluster_counts @ false_positive_by_cluster
    fn = bootstrap_cluster_counts @ false_negative_by_cluster
    tn = bootstrap_cluster_counts @ true_negative_by_cluster
    f1 = np.divide(
        2 * tp,
        2 * tp + fp + fn,
        out=np.zeros_like(tp, dtype=float),
        where=(2 * tp + fp + fn) > 0
    )
    far = np.divide(
        fp,
        fp + tn,
        out=np.full_like(fp, np.nan, dtype=float),
        where=(fp + tn) > 0
    )

    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = y_true[order]
    sorted_cluster_codes = cluster_codes[order]
    group_ends = np.r_[
        np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]),
        len(sorted_scores) - 1
    ]
    auprc = np.full(len(bootstrap_cluster_counts), np.nan, dtype=float)
    for batch_start in range(0, len(bootstrap_cluster_counts), batch_size):
        batch_end = min(batch_start + batch_size, len(bootstrap_cluster_counts))
        counts = bootstrap_cluster_counts[batch_start:batch_end]
        sample_weights = counts[:, sorted_cluster_codes]
        cumulative_tp = np.cumsum(
            sample_weights * sorted_labels[None, :],
            axis=1
        )[:, group_ends]
        cumulative_total = np.cumsum(sample_weights, axis=1)[:, group_ends]
        increments_tp = np.diff(
            np.pad(cumulative_tp, ((0, 0), (1, 0))),
            axis=1
        )
        precision = np.divide(
            cumulative_tp,
            cumulative_total,
            out=np.zeros_like(cumulative_tp, dtype=float),
            where=cumulative_total > 0
        )
        total_positive = cumulative_tp[:, -1]
        auprc[batch_start:batch_end] = np.divide(
            np.sum(increments_tp * precision, axis=1),
            total_positive,
            out=np.full(batch_end - batch_start, np.nan, dtype=float),
            where=total_positive > 0
        )
    return {"F1": f1, "AUPRC": auprc, "FAR": far}


def mmsi_cluster_bootstrap(evaluation_name, eval_data, X_eval, y_eval):
    if (
        "HistGradientBoosting" not in trained_models
        or "RandomForest" not in trained_models
    ):
        return []

    hgb = trained_models["HistGradientBoosting"]
    rf = trained_models["RandomForest"]

    hgb_validation_scores = get_abnormal_scores(hgb, X_validation)
    rf_validation_scores = get_abnormal_scores(rf, X_validation)

    hgb_threshold, _ = calibrate_threshold_at_far(
        hgb_validation_scores[y_validation.to_numpy() == 0],
        0.03
    )
    rf_threshold, _ = calibrate_threshold_at_far(
        rf_validation_scores[y_validation.to_numpy() == 0],
        0.03
    )

    hgb_scores = get_abnormal_scores(hgb, X_eval)
    rf_scores = get_abnormal_scores(rf, X_eval)

    scored = eval_data.copy()
    scored["HGB_Score"] = hgb_scores
    scored["RF_Score"] = rf_scores
    y_true = scored["Window_Label"].astype(int).to_numpy()

    rng = np.random.default_rng(BOOTSTRAP_RANDOM_SEED)
    mmsi_values = np.sort(scored["MMSI"].dropna().astype(int).unique())

    rows = []

    if len(mmsi_values) == 0:
        return rows

    mmsi_to_code = {mmsi: index for index, mmsi in enumerate(mmsi_values)}
    cluster_codes = scored["MMSI"].astype(int).map(mmsi_to_code).to_numpy()
    bootstrap_cluster_counts = rng.multinomial(
        len(mmsi_values),
        np.full(len(mmsi_values), 1.0 / len(mmsi_values)),
        size=BOOTSTRAP_REPEATS
    )
    hgb_metrics = bootstrap_metric_arrays(
        y_true,
        scored["HGB_Score"].to_numpy(),
        hgb_threshold,
        cluster_codes,
        bootstrap_cluster_counts
    )
    rf_metrics = bootstrap_metric_arrays(
        y_true,
        scored["RF_Score"].to_numpy(),
        rf_threshold,
        cluster_codes,
        bootstrap_cluster_counts
    )

    for bootstrap_index in range(BOOTSTRAP_REPEATS):

        rows.append(
            {
                "Evaluation_Set": evaluation_name,
                "Bootstrap_Index": bootstrap_index,
                "HGB_F1": hgb_metrics["F1"][bootstrap_index],
                "HGB_AUPRC": hgb_metrics["AUPRC"][bootstrap_index],
                "HGB_FAR": hgb_metrics["FAR"][bootstrap_index],
                "RF_F1": rf_metrics["F1"][bootstrap_index],
                "RF_AUPRC": rf_metrics["AUPRC"][bootstrap_index],
                "RF_FAR": rf_metrics["FAR"][bootstrap_index],
                "Delta_F1_HGB_minus_RF": (
                    hgb_metrics["F1"][bootstrap_index]
                    - rf_metrics["F1"][bootstrap_index]
                ),
                "Delta_AUPRC_HGB_minus_RF": (
                    hgb_metrics["AUPRC"][bootstrap_index]
                    - rf_metrics["AUPRC"][bootstrap_index]
                )
            }
        )

    return rows


bootstrap_rows = []

for evaluation_name, (eval_data, X_eval, y_eval) in evaluation_sets.items():
    bootstrap_rows.extend(
        mmsi_cluster_bootstrap(evaluation_name, eval_data, X_eval, y_eval)
    )

bootstrap_distribution = pd.DataFrame(bootstrap_rows)
bootstrap_distribution_file = (
    OUTPUT_DIR / "mmsi_cluster_bootstrap_distribution.csv"
)
bootstrap_summary_file = OUTPUT_DIR / "mmsi_cluster_bootstrap_summary.csv"

if len(bootstrap_distribution) > 0:
    bootstrap_distribution.to_csv(
        bootstrap_distribution_file,
        index=False,
        encoding="utf-8-sig"
    )

    summary_rows = []

    for evaluation_name, subset in bootstrap_distribution.groupby(
        "Evaluation_Set"
    ):
        for metric in [
            "HGB_F1",
            "HGB_AUPRC",
            "HGB_FAR",
            "Delta_F1_HGB_minus_RF",
            "Delta_AUPRC_HGB_minus_RF"
        ]:
            values = subset[metric].replace(
                [np.inf, -np.inf],
                np.nan
            ).dropna()

            summary_rows.append(
                {
                    "Evaluation_Set": evaluation_name,
                    "Metric": metric,
                    "Mean": values.mean(),
                    "CI95_Lower": values.quantile(0.025),
                    "CI95_Upper": values.quantile(0.975),
                    "Bootstrap_Repeats": len(values),
                    "Cluster_Unit": "MMSI"
                }
            )

    bootstrap_summary = pd.DataFrame(summary_rows)
    bootstrap_summary.to_csv(
        bootstrap_summary_file,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"Bootstrap distribution saved to: {bootstrap_distribution_file}")
    print(f"Bootstrap summary saved to: {bootstrap_summary_file}")


# ============================================================
# 6. Figures and reproducibility table
# ============================================================
if model_scores_for_curves:
    plt.figure(figsize=(7, 6))

    for model_name, scores in model_scores_for_curves.items():
        precision, recall, _ = precision_recall_curve(y_test, scores)
        auprc = average_precision_score(y_test, scores)

        plt.plot(
            recall,
            precision,
            label=f"{model_name} AUPRC={auprc:.3f}"
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves on Guam Test Set")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    pr_curve_file = FIGURE_DIR / "precision_recall_curves_guam_test.png"
    plt.savefig(pr_curve_file, dpi=300)
    plt.close()
    print(f"PR curve saved to: {pr_curve_file}")

model_configuration_rows = []

for model_name in models:
    model_configuration_rows.append(
        {
            "Model": model_name,
            "Preprocessing": (
                PROTOCOL["modeling"]["missing_value_policy"].get(
                    model_name,
                    "median imputation fitted on training data only"
                )
            ),
            "Hyperparameters": str(models[model_name].get_params()),
            "Class_Weighting": "balanced where supported",
            "Threshold_Protocol": (
                "Thresholds calibrated only on Guam validation label-zero scores (nominal plus Generator-A/B negative context) at target FAR values 1%, 3%, and 5%; frozen "
                "before Guam test and LA evaluation."
            ),
            "Random_Seed": RANDOM_SEED,
            "Software_Version": (
                f"python={platform.python_version()}; "
                f"scikit-learn={sklearn.__version__}; "
                f"pandas={pd.__version__}; numpy={np.__version__}"
            )
        }
    )

model_configuration_rows.append(
    {
        "Model": "GRU",
        "Preprocessing": (
            "per-point median imputation fitted on Guam training points only; "
            "training-only standardization; 10-point sequences"
        ),
        "Hyperparameters": (
            "GRU(input_size=7, hidden_size=32, batch_first=True); "
            "head=Linear(32,16)+ReLU+Linear(16,1); epochs=5; "
            "batch_size=1024; optimizer=Adam(lr=0.001); BCEWithLogitsLoss"
        ),
        "Class_Weighting": "none",
        "Threshold_Protocol": (
            "Thresholds calibrated only on Guam validation label-zero scores (nominal plus Generator-A/B negative context) at "
            "target FAR values 1%, 3%, and 5%; frozen before Guam test and LA evaluation."
        ),
        "Random_Seed": RANDOM_SEED,
        "Software_Version": (
            "python=3.11.15; pytorch=2.12.0+cpu; "
            "pandas=3.0.3; numpy=2.4.6"
        )
    }
)

model_configuration_table = pd.DataFrame(model_configuration_rows)
model_configuration_file = OUTPUT_DIR / "supplementary_table_S3_model_config.csv"
model_configuration_table.to_csv(
    model_configuration_file,
    index=False,
    encoding="utf-8-sig"
)

print(f"Supplementary model config table saved to: {model_configuration_file}")
print("=" * 90)
print("Stage 08 complete.")
