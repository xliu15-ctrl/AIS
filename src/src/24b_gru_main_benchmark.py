from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "windows_v1"
LA_ROOT = ROOT / "external_validation" / "la_2025"
OUT = ROOT / "results" / "gru_main_benchmark_v2"
OUT.mkdir(parents=True, exist_ok=True)

with open(ROOT / "config" / "experiment_protocol.json", encoding="utf-8") as handle:
    PROTOCOL = json.load(handle)

SEED = int(PROTOCOL["modeling"]["main_random_seed"])
TARGET_FARS = PROTOCOL["threshold_protocol"]["target_far_values"]
POINT_FEATURES = [
    "Delta_Minutes",
    "Distance_NM",
    "Calculated_Speed_Knots",
    "Speed_Residual_Knots",
    "COG_Change_Rate_Deg_Per_Min",
    "Acceleration_Knots_Per_Min",
    "Course_Position_Mismatch_Deg",
]
SEQ_LEN = 10
BATCH = 1024
EPOCHS = 5
HIDDEN = 32
DEVICE = torch.device("cpu")


def load_windows(path):
    frame = pd.read_csv(path, low_memory=False)
    if "Generator" not in frame:
        frame["Generator"] = np.where(
            frame["Attack_Type"].eq("normal"), "none", "A"
        )
    return frame


def build_sequences(windows, point_file, label):
    needed = ["Scenario_ID", "Point_Index", *POINT_FEATURES]
    points = pd.read_csv(point_file, usecols=needed, low_memory=False)
    points["Point_Index"] = pd.to_numeric(
        points["Point_Index"], errors="coerce"
    ).astype(int)
    point_groups = {
        scenario_id: group.sort_values("Point_Index")[POINT_FEATURES].to_numpy(
            np.float32
        )
        for scenario_id, group in points.groupby("Scenario_ID", sort=False)
    }
    chunks = []
    labels = []
    metadata = []
    for scenario_id, group in windows.groupby("Scenario_ID", sort=False):
        values = point_groups.get(scenario_id)
        if values is None:
            continue
        starts = group["Window_Start_Index"].astype(int).to_numpy()
        indices = starts[:, None] + np.arange(SEQ_LEN)
        valid = indices[:, -1] < len(values)
        if not valid.any():
            continue
        chunks.append(values[indices[valid]])
        labels.append(
            group.loc[group.index[valid], "Window_Label"].astype(int).to_numpy()
        )
        metadata.append(group.loc[group.index[valid]].copy())
    del points, point_groups
    X = np.concatenate(chunks).astype(np.float32)
    y = np.concatenate(labels).astype(np.int64)
    meta = pd.concat(metadata, ignore_index=True)
    print(label, X.shape, flush=True)
    return X, y, meta


class GRUDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(len(POINT_FEATURES), HIDDEN, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(HIDDEN, 16), nn.ReLU(), nn.Linear(16, 1)
        )

    def forward(self, values):
        sequence, _ = self.gru(values)
        return self.head(sequence[:, -1]).squeeze(1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def predict(model, X):
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X)), batch_size=BATCH, shuffle=False
    )
    values = []
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            values.append(torch.sigmoid(model(batch)).numpy())
    return np.concatenate(values)


def threshold_at_far(normal_scores, target_far):
    return float(np.quantile(normal_scores, 1 - target_far, method="higher"))


def classification_metrics(y_true, scores, threshold):
    predictions = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()
    return {
        "F1": f1_score(y_true, predictions, zero_division=0),
        "AUPRC": average_precision_score(y_true, scores),
        "FAR": fp / (fp + tn) if fp + tn else np.nan,
        "Recall": recall_score(y_true, predictions, zero_division=0),
        "Precision": precision_score(y_true, predictions, zero_division=0),
        "Miss_Rate": fn / (fn + tp) if fn + tp else np.nan,
    }


def build_alert_events(subset):
    ordered = subset.sort_values("Window_Start_Index").reset_index(drop=True)
    events = []
    current = None
    for ordinal, row in ordered.iterrows():
        if int(row["Prediction"]) != 1:
            if current is not None:
                events.append(current)
                current = None
            continue
        decision_time = pd.to_datetime(
            row["Window_End_Time"], errors="coerce"
        )
        item = {
            "start": int(row["Window_Start_Index"]),
            "end": int(row["Window_End_Index"]),
            "start_time": decision_time,
            "end_time": decision_time,
            "decision_index": int(row["Window_End_Index"]),
            "last_ordinal": ordinal,
            "windows": [{
                "start": int(row["Window_Start_Index"]),
                "end": int(row["Window_End_Index"]),
            }],
        }
        if current is None:
            current = item
        elif ordinal == current["last_ordinal"] + 1:
            current["end"] = max(current["end"], item["end"])
            if pd.notna(item["end_time"]):
                current["end_time"] = (
                    item["end_time"]
                    if pd.isna(current["end_time"])
                    else max(current["end_time"], item["end_time"])
                )
            current["last_ordinal"] = ordinal
            current["windows"].extend(item["windows"])
        else:
            events.append(current)
            current = item
    if current is not None:
        events.append(current)
    return events


def event_metrics(metadata, predictions):
    scored = metadata.copy()
    scored["Prediction"] = np.asarray(predictions).astype(int)
    attack_events = matched_attacks = alert_count = matched_alerts_total = 0
    false_alerts = 0
    delays = []
    durations = []
    repeated = []
    for _, subset in scored.groupby("Scenario_ID"):
        subset = subset.sort_values("Window_Start_Index")
        alerts = build_alert_events(subset)
        alert_count += len(alerts)
        for alert in alerts:
            if pd.notna(alert["start_time"]) and pd.notna(alert["end_time"]):
                durations.append(
                    max(
                        0.0,
                        (alert["end_time"] - alert["start_time"]).total_seconds()
                        / 60.0,
                    )
                )
        if str(subset["Attack_Type"].iloc[0]) == "normal":
            false_alerts += len(alerts)
            continue
        attack_start = pd.to_numeric(
            subset["Scenario_Attack_Start_Index"], errors="coerce"
        ).dropna()
        attack_end = pd.to_numeric(
            subset["Scenario_Attack_End_Index"], errors="coerce"
        ).dropna()
        if attack_start.empty or attack_end.empty:
            continue
        attack_start = int(attack_start.iloc[0])
        attack_end = int(attack_end.iloc[0])
        attack_events += 1
        attack_time = pd.to_datetime(
            subset["Scenario_Attack_Start_Time"].iloc[0], errors="coerce"
        )
        matched = []
        for alert in alerts:
            causal_start = (
                pd.notna(alert["start_time"])
                and pd.notna(attack_time)
                and alert["start_time"] >= attack_time
            ) or (
                pd.isna(attack_time)
                and alert["decision_index"] >= attack_start
            )
            evidence_overlap = any(
                window["end"] >= attack_start
                and window["start"] <= attack_end
                for window in alert["windows"]
            )
            if causal_start and evidence_overlap:
                matched.append(alert)
        if matched:
            matched_attacks += 1
            matched_alerts_total += len(matched)
            first = min(
                matched,
                key=lambda item: item["start_time"]
                if pd.notna(item["start_time"])
                else pd.Timestamp.max,
            )
            if pd.notna(attack_time) and pd.notna(first["start_time"]):
                delays.append(
                    (first["start_time"] - attack_time).total_seconds() / 60.0
                )
            repeated.append(max(0, len(matched) - 1))
        false_alerts += len(alerts) - len(matched)

    exposure = scored.groupby("Scenario_ID", as_index=False).agg(
        start=("Scenario_Start_Time", "first"),
        end=("Scenario_End_Time", "first"),
    )
    exposure["start"] = pd.to_datetime(exposure["start"], errors="coerce")
    exposure["end"] = pd.to_datetime(exposure["end"], errors="coerce")
    exposure_hours = (
        (exposure["end"] - exposure["start"])
        .dt.total_seconds()
        .clip(lower=0)
        .sum()
        / 3600.0
    )
    return {
        "Event_Level_Precision": (
            matched_alerts_total / alert_count if alert_count else 0.0
        ),
        "Event_Level_Recall": (
            matched_attacks / attack_events if attack_events else np.nan
        ),
        "Median_Detection_Delay_Minutes": (
            float(np.median(delays)) if delays else np.nan
        ),
        "P95_Detection_Delay_Minutes": (
            float(np.percentile(delays, 95)) if delays else np.nan
        ),
        "False_Alerts_Per_100_Synthetic_Scenario_Hours": (
            false_alerts / exposure_hours * 100 if exposure_hours else np.nan
        ),
        "Alerts_Per_Trajectory": alert_count / scored["Scenario_ID"].nunique(),
        "Mean_Alert_Duration_Minutes": (
            float(np.mean(durations)) if durations else np.nan
        ),
        "Repeated_Alerts_Per_Attack_Event": (
            float(np.mean(repeated)) if repeated else 0.0
        ),
        "Alert_Event_Count": int(alert_count),
        "Matched_Alert_Event_Count": int(matched_alerts_total),
        "False_Alert_Event_Count": int(false_alerts),
        "Attack_Event_Count": int(attack_events),
        "Detected_Attack_Event_Count": int(matched_attacks),
        "Missed_Attack_Event_Count": int(attack_events - matched_attacks),
        "Repeated_Matched_Alert_Count": int(sum(repeated)),
        "Exposure_Hours": float(exposure_hours),
        "Exposure_Method": "sum of synthetic scenario time spans (scenario-hours)",
        "Alert_Decision_Time": "Window_End_Time",
        "Alert_Merge_Gap_Windows": 0,
        "Precision_Unit": "matched alert events / all alert events",
        "Recall_Unit": "unique detected attack intervals / all attack intervals",
        "Pre_Attack_Alert_Handling": "unmatched false alert",
        "Negative_Delay_Clipping": False,
    }


train_windows = load_windows(DATA / "train_window_samples.csv")
train_windows = train_windows[
    train_windows["Attack_Type"].eq("normal")
    | train_windows["Generator"].eq("A")
].copy()
validation_windows = load_windows(DATA / "validation_window_samples.csv")
guam_windows = load_windows(DATA / "test_window_samples.csv")
la_windows = load_windows(LA_ROOT / "windows_v2" / "test_window_samples.csv")

X_train, y_train, _ = build_sequences(
    train_windows,
    ROOT / "data" / "synthetic_v2" / "train_synthetic_scenarios_v2.csv",
    "train",
)
X_validation, y_validation, _ = build_sequences(
    validation_windows,
    ROOT / "data" / "synthetic_v2" / "validation_synthetic_scenarios_v2.csv",
    "validation",
)
X_guam, y_guam, metadata_guam = build_sequences(
    guam_windows,
    ROOT / "data" / "synthetic_v2" / "test_synthetic_scenarios_v2.csv",
    "Guam",
)
X_la, y_la, metadata_la = build_sequences(
    la_windows,
    LA_ROOT / "synthetic_v3" / "test_synthetic_scenarios_v2.csv",
    "LA",
)

flat_train = X_train.reshape(-1, len(POINT_FEATURES))
medians = np.nanmedian(flat_train, axis=0)
medians = np.where(np.isfinite(medians), medians, 0)
flat_train = np.where(np.isfinite(flat_train), flat_train, medians)
means = flat_train.mean(axis=0)
standard_deviations = flat_train.std(axis=0)
standard_deviations = np.where(standard_deviations > 1e-6, standard_deviations, 1)


def transform(X):
    imputed = np.where(np.isfinite(X), X, medians)
    return ((imputed - means) / standard_deviations).astype(np.float32)


X_train = transform(X_train)
X_validation = transform(X_validation)
X_guam = transform(X_guam)
X_la = transform(X_la)

evaluation_sets = {}
for domain, X, y, metadata, mapping in [
    ("Guam", X_guam, y_guam, metadata_guam, {"A": "A", "B": "C"}),
    ("LA", X_la, y_la, metadata_la, {"A": "B", "B": "D"}),
]:
    normal_mask = metadata["Attack_Type"].eq("normal").to_numpy()
    for generator, experiment in mapping.items():
        mask = normal_mask | metadata["Generator"].eq(generator).to_numpy()
        evaluation_sets[experiment] = (
            domain,
            generator,
            X[mask],
            y[mask],
            metadata.loc[mask].reset_index(drop=True),
        )

set_seed(SEED)
model = GRUDetector().to(DEVICE)
positive = max(1, int(y_train.sum()))
negative = len(y_train) - positive
loss_function = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor(negative / positive, dtype=torch.float32)
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
train_loader = DataLoader(
    TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train.astype(np.float32))
    ),
    batch_size=BATCH,
    shuffle=True,
)
for epoch in range(EPOCHS):
    model.train()
    losses = []
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        loss = loss_function(model(batch_X), batch_y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    print(f"epoch={epoch + 1} loss={np.mean(losses):.6f}", flush=True)

validation_scores = predict(model, X_validation)
validation_normal_scores = validation_scores[y_validation == 0]
thresholds = {
    float(target_far): threshold_at_far(validation_normal_scores, float(target_far))
    for target_far in TARGET_FARS
}

pd.DataFrame(
    [
        {
            "Model": "GRU",
            "Target_FAR": target_far,
            "Threshold_Source": f"Guam validation normal FAR={target_far:.0%}",
            "Frozen_Threshold": threshold,
            "Calibration_Normal_Window_Count": len(validation_normal_scores),
        }
        for target_far, threshold in thresholds.items()
    ]
).to_csv(OUT / "gru_guam_validation_fixed_far_thresholds.csv", index=False, encoding="utf-8-sig")

metric_rows = []
per_attack_rows = []
event_rows = []
prediction_rows = []
for experiment, (domain, generator, X, y, metadata) in evaluation_sets.items():
    scores = predict(model, X)
    for target_far, threshold in thresholds.items():
        calculated = classification_metrics(y, scores, threshold)
        metric_rows.append(
            {
                "Model": "GRU",
                "Seed": SEED,
                "Experiment": experiment,
                "Domain": domain,
                "Generator": generator,
                "Evaluation_Set": f"{domain} test",
                "Target_FAR": target_far,
                "Threshold_Source": f"Guam validation normal FAR={target_far:.0%}",
                "Frozen_Threshold": threshold,
                **calculated,
            }
        )
    threshold = thresholds[0.03]
    predictions = scores >= threshold
    for attack_type in sorted(metadata["Attack_Type"].unique()):
        mask = metadata["Attack_Type"].eq(attack_type).to_numpy()
        if attack_type == "normal" or not mask.any():
            continue
        per_attack_rows.append(
            {
                "Model": "GRU",
                "Experiment": experiment,
                "Domain": domain,
                "Generator": generator,
                "Attack_Type": attack_type,
                "Sample_Count": int(mask.sum()),
                "Recall": recall_score(y[mask], predictions[mask], zero_division=0),
                "Threshold_Source": "Guam validation normal FAR=3%",
                "Frozen_Threshold": threshold,
            }
        )
    event_rows.append(
        {
            "Model": "GRU",
            "Experiment": experiment,
            "Domain": domain,
            "Generator": generator,
            "Threshold_Source": "Guam validation normal FAR=3%",
            "Frozen_Threshold": threshold,
            **event_metrics(metadata, predictions),
        }
    )
    compact = metadata[
        ["Scenario_ID", "MMSI", "Generator", "Attack_Type", "Window_Start_Index"]
    ].copy()
    compact["Experiment"] = experiment
    compact["Window_Label"] = y
    compact["Score"] = scores
    compact["Prediction_FAR3"] = predictions.astype(int)
    prediction_rows.append(compact)

pd.DataFrame(metric_rows).to_csv(
    OUT / "gru_fixed_far_transfer_metrics.csv", index=False, encoding="utf-8-sig"
)
pd.DataFrame(per_attack_rows).to_csv(
    OUT / "gru_per_attack_recall_far3.csv", index=False, encoding="utf-8-sig"
)
pd.DataFrame(event_rows).to_csv(
    OUT / "gru_event_metrics_far3.csv", index=False, encoding="utf-8-sig"
)
pd.concat(prediction_rows, ignore_index=True).to_csv(
    OUT / "gru_scored_windows_far3.csv", index=False, encoding="utf-8-sig"
)
torch.save(model.state_dict(), OUT / "gru_seed42_state_dict.pt")
np.savez(
    OUT / "gru_training_preprocessing.npz",
    point_features=np.asarray(POINT_FEATURES),
    medians=medians,
    means=means,
    standard_deviations=standard_deviations,
)
print(pd.DataFrame(metric_rows).to_string(index=False), flush=True)
print(OUT, flush=True)
