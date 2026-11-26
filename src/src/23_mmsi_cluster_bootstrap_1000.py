from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "models_v1"
FEATURE_FILE = ROOT / "data" / "windows_v1" / "model_feature_columns.txt"
THRESHOLD_FILE = OUT / "guam_validation_fixed_far_thresholds.csv"
REPEATS = 1000
RANDOM_SEED = 20260916

with open(FEATURE_FILE, encoding="utf-8") as handle:
    FEATURES = [line.strip() for line in handle if line.strip()]


def prepare(frame):
    X = frame[FEATURES].replace([np.inf, -np.inf], np.nan)
    y = pd.to_numeric(frame["Window_Label"], errors="coerce").fillna(0).astype(int)
    return X, y


def scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    raw = model.decision_function(X)
    return 1 / (1 + np.exp(-np.clip(raw, -50, 50)))


def experiment_sets(domain, frame):
    normal = frame[frame["Attack_Type"].eq("normal")]
    mapping = {"A": "A", "B": "C"} if domain == "Guam" else {"A": "B", "B": "D"}
    result = {}
    for generator, experiment in mapping.items():
        subset = pd.concat(
            [normal, frame[frame["Generator"].eq(generator)]], ignore_index=True
        )
        result[f"{domain} experiment {experiment} Generator {generator}"] = subset
    return result


def bootstrap_metric_arrays(
    y_true, score_values, threshold, cluster_codes, cluster_draw_counts, batch_size=25
):
    y_true = np.asarray(y_true).astype(int)
    score_values = np.asarray(score_values, dtype=float)
    predictions = score_values >= threshold
    cluster_count = cluster_draw_counts.shape[1]

    def aggregate(mask):
        return np.bincount(
            cluster_codes,
            weights=np.asarray(mask, dtype=float),
            minlength=cluster_count,
        )

    tp = cluster_draw_counts @ aggregate((y_true == 1) & predictions)
    fp = cluster_draw_counts @ aggregate((y_true == 0) & predictions)
    fn = cluster_draw_counts @ aggregate((y_true == 1) & ~predictions)
    tn = cluster_draw_counts @ aggregate((y_true == 0) & ~predictions)
    f1 = np.divide(
        2 * tp,
        2 * tp + fp + fn,
        out=np.zeros_like(tp, dtype=float),
        where=(2 * tp + fp + fn) > 0,
    )
    far = np.divide(
        fp,
        fp + tn,
        out=np.full_like(fp, np.nan, dtype=float),
        where=(fp + tn) > 0,
    )

    order = np.argsort(-score_values, kind="mergesort")
    sorted_scores = score_values[order]
    sorted_labels = y_true[order]
    sorted_codes = cluster_codes[order]
    group_ends = np.r_[
        np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]),
        len(sorted_scores) - 1,
    ]
    auprc = np.full(len(cluster_draw_counts), np.nan)
    for start in range(0, len(cluster_draw_counts), batch_size):
        end = min(start + batch_size, len(cluster_draw_counts))
        weights = cluster_draw_counts[start:end][:, sorted_codes]
        cumulative_tp = np.cumsum(weights * sorted_labels[None, :], axis=1)[
            :, group_ends
        ]
        cumulative_total = np.cumsum(weights, axis=1)[:, group_ends]
        increments_tp = np.diff(
            np.pad(cumulative_tp, ((0, 0), (1, 0))), axis=1
        )
        precision = np.divide(
            cumulative_tp,
            cumulative_total,
            out=np.zeros_like(cumulative_tp, dtype=float),
            where=cumulative_total > 0,
        )
        total_positive = cumulative_tp[:, -1]
        auprc[start:end] = np.divide(
            np.sum(increments_tp * precision, axis=1),
            total_positive,
            out=np.full(end - start, np.nan),
            where=total_positive > 0,
        )
    return {"F1": f1, "AUPRC": auprc, "FAR": far}


thresholds = pd.read_csv(THRESHOLD_FILE)
thresholds = thresholds[np.isclose(thresholds["Target_FAR"], 0.03)]
hgb = joblib.load(ROOT / "models_v1" / "HistGradientBoosting_pipeline.joblib")
rf = joblib.load(ROOT / "models_v1" / "RandomForest_pipeline.joblib")
hgb_threshold = float(
    thresholds.loc[thresholds["Model"].eq("HistGradientBoosting"), "Frozen_Threshold"].iloc[0]
)
rf_threshold = float(
    thresholds.loc[thresholds["Model"].eq("RandomForest"), "Frozen_Threshold"].iloc[0]
)

guam = pd.read_csv(ROOT / "data" / "windows_v1" / "test_window_samples.csv", low_memory=False)
la = pd.read_csv(
    ROOT / "external_validation" / "la_2025" / "windows_v2" / "test_window_samples.csv",
    low_memory=False,
)
sets = {**experiment_sets("Guam", guam), **experiment_sets("LA", la)}
rows = []
for evaluation_name, frame in sets.items():
    print(evaluation_name, len(frame), flush=True)
    X, y = prepare(frame)
    hgb_scores = scores(hgb, X)
    rf_scores = scores(rf, X)
    mmsi_values = np.sort(frame["MMSI"].astype(int).unique())
    mapping = {mmsi: index for index, mmsi in enumerate(mmsi_values)}
    codes = frame["MMSI"].astype(int).map(mapping).to_numpy()
    rng = np.random.default_rng(RANDOM_SEED)
    counts = rng.multinomial(
        len(mmsi_values),
        np.full(len(mmsi_values), 1.0 / len(mmsi_values)),
        size=REPEATS,
    )
    hgb_metrics = bootstrap_metric_arrays(
        y, hgb_scores, hgb_threshold, codes, counts
    )
    rf_metrics = bootstrap_metric_arrays(y, rf_scores, rf_threshold, codes, counts)
    for index in range(REPEATS):
        rows.append(
            {
                "Evaluation_Set": evaluation_name,
                "Bootstrap_Index": index,
                "HGB_F1": hgb_metrics["F1"][index],
                "HGB_AUPRC": hgb_metrics["AUPRC"][index],
                "HGB_FAR": hgb_metrics["FAR"][index],
                "RF_F1": rf_metrics["F1"][index],
                "RF_AUPRC": rf_metrics["AUPRC"][index],
                "RF_FAR": rf_metrics["FAR"][index],
                "Delta_F1_HGB_minus_RF": (
                    hgb_metrics["F1"][index] - rf_metrics["F1"][index]
                ),
                "Delta_AUPRC_HGB_minus_RF": (
                    hgb_metrics["AUPRC"][index] - rf_metrics["AUPRC"][index]
                ),
            }
        )

distribution = pd.DataFrame(rows)
distribution.to_csv(
    OUT / "mmsi_cluster_bootstrap_distribution.csv", index=False, encoding="utf-8-sig"
)
summary_rows = []
for evaluation_name, subset in distribution.groupby("Evaluation_Set"):
    for metric in [
        "HGB_F1",
        "HGB_AUPRC",
        "HGB_FAR",
        "Delta_F1_HGB_minus_RF",
        "Delta_AUPRC_HGB_minus_RF",
    ]:
        values = subset[metric].replace([np.inf, -np.inf], np.nan).dropna()
        summary_rows.append(
            {
                "Evaluation_Set": evaluation_name,
                "Metric": metric,
                "Mean": values.mean(),
                "CI95_Lower": values.quantile(0.025),
                "CI95_Upper": values.quantile(0.975),
                "Bootstrap_Repeats": len(values),
                "Cluster_Unit": "MMSI",
                "Comparison": "paired HGB minus RF on identical bootstrap samples",
            }
        )
summary = pd.DataFrame(summary_rows)
summary.to_csv(
    OUT / "mmsi_cluster_bootstrap_summary.csv", index=False, encoding="utf-8-sig"
)
print(summary.to_string(index=False), flush=True)
