"""Constant-velocity Kalman/NIS benchmark using the frozen 3% FAR protocol.

An auditable CV implementation of the AIS residual/chi-square integrity-test
family described by Siegert et al. (IEEE/ION PLANS 2016, DOI
10.1109/PLANS.2016.7479784). It is an adapted baseline, not a reproduction of
their full EKF/IMM/GLR system.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "reviewer_followup_t4"
OUT.mkdir(parents=True, exist_ok=True)
WINDOWS = {
    "Guam_val": ROOT / "data/windows_v1/validation_window_samples.csv",
    "Guam_test": ROOT / "data/windows_v1/test_window_samples.csv",
    "LA_test": ROOT / "external_validation/la_2025/windows_v2/test_window_samples.csv",
}
POINTS = {
    "Guam_val": ROOT / "data/synthetic_v2/validation_synthetic_scenarios_v2.csv",
    "Guam_test": ROOT / "data/synthetic_v2/test_synthetic_scenarios_v2.csv",
    "LA_test": ROOT / "external_validation/la_2025/synthetic_v3/test_synthetic_scenarios_v2.csv",
}
R_METRES = 25.0
ACCELERATION_PSD = 0.25
CHUNK = 400_000


def point_nis(frame):
    frame = frame.sort_values("Point_Index")
    lon = pd.to_numeric(frame.Longitude, errors="coerce").to_numpy(float)
    lat = pd.to_numeric(frame.Latitude, errors="coerce").to_numpy(float)
    t = pd.to_datetime(frame.Time, errors="coerce", utc=True).astype("int64").to_numpy() / 1e9
    n = len(frame)
    out = np.full(n, np.nan)
    good = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(t)
    if n < 3 or not good.all():
        return out
    lat0 = np.deg2rad(lat[0])
    z = np.column_stack(((lon - lon[0]) * 111320.0 * np.cos(lat0),
                         (lat - lat[0]) * 111320.0))
    dt0 = t[1] - t[0]
    if dt0 <= 0:
        return out
    x = np.array([z[1, 0], z[1, 1], (z[1, 0]-z[0, 0])/dt0, (z[1, 1]-z[0, 1])/dt0])
    P = np.diag([R_METRES**2, R_METRES**2, 25.0**2, 25.0**2])
    H = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.]])
    R = np.eye(2) * R_METRES**2
    I = np.eye(4)
    for k in range(2, n):
        dt = t[k] - t[k-1]
        if dt <= 0 or dt > 1800:
            # The preprocessor splits >30-minute gaps; guard against malformed input.
            continue
        F = np.array([[1., 0., dt, 0.], [0., 1., 0., dt],
                      [0., 0., 1., 0.], [0., 0., 0., 1.]])
        Q = np.zeros((4, 4))
        qpp, qpv, qvv = dt**3/3, dt**2/2, dt
        for p, v in [(0, 2), (1, 3)]:
            Q[p, p] = ACCELERATION_PSD*qpp
            Q[p, v] = Q[v, p] = ACCELERATION_PSD*qpv
            Q[v, v] = ACCELERATION_PSD*qvv
        xp = F @ x
        Pp = F @ P @ F.T + Q
        innovation = z[k] - H @ xp
        S = H @ Pp @ H.T + R
        try:
            nis = float(innovation @ np.linalg.solve(S, innovation))
        except np.linalg.LinAlgError:
            continue
        out[k] = nis
        K = np.linalg.solve(S, H @ Pp).T
        x = xp + K @ innovation
        P = (I-K@H) @ Pp @ (I-K@H).T + K @ R @ K.T
    return out


def score_dataset(key):
    windows = pd.read_csv(WINDOWS[key], low_memory=False)
    windows["Window_NIS_Max"] = np.nan
    ids = set(windows.Scenario_ID.astype(str))
    chunks = []
    use = ["Scenario_ID", "Point_Index", "Time", "Longitude", "Latitude"]
    for part in pd.read_csv(POINTS[key], usecols=use, chunksize=CHUNK, low_memory=False):
        part = part[part.Scenario_ID.astype(str).isin(ids)]
        if len(part):
            chunks.append(part)
    points = pd.concat(chunks, ignore_index=True)
    scored = []
    point_groups = {str(sid): group for sid, group in points.groupby("Scenario_ID", sort=False)}
    for sid, group in windows.groupby("Scenario_ID", sort=False):
        p = point_groups.get(str(sid))
        if p is None:
            continue
        p = p.sort_values("Point_Index")
        nis = point_nis(p)
        idx = p.Point_Index.to_numpy(int)
        lookup = {int(i): float(v) for i, v in zip(idx, nis) if np.isfinite(v)}
        for rid, row in group.iterrows():
            vals = [lookup.get(i) for i in range(int(row.Window_Start_Index), int(row.Window_End_Index)+1)]
            vals = [v for v in vals if v is not None]
            if vals:
                scored.append((rid, max(vals)))
    score = pd.Series(dict(scored), name="CV_KF_Max_NIS")
    windows = windows.join(score, how="left")
    return windows


def metrics(frame, threshold, label):
    data = frame[frame.CV_KF_Max_NIS.notna()]
    y = data.Window_Label.to_numpy(int).astype(bool)
    pred = data.CV_KF_Max_NIS.to_numpy(float) >= threshold
    tp = int(np.sum(y & pred)); fp = int(np.sum(~y & pred)); fn = int(np.sum(y & ~pred)); tn = int(np.sum(~y & ~pred))
    precision = tp / max(1, tp+fp); recall = tp / max(1, tp+fn)
    return {"Evaluation_Set": label, "Threshold_Source": "Guam validation Window_Label=0; empirical 97th percentile (higher)",
            "Frozen_Threshold_Max_NIS": threshold, "Window_Count": len(data), "Positive_Windows": int(y.sum()),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn, "FAR": fp/max(1, fp+tn),
            "Precision": precision, "Recall": recall,
            "F1": 2*precision*recall/max(1e-15, precision+recall),
            "Score_Coverage": len(data)/len(frame)}


def main():
    val = score_dataset("Guam_val")
    normal = val.loc[val.Window_Label.eq(0), "CV_KF_Max_NIS"].dropna().to_numpy()
    threshold = float(np.quantile(normal, .97, method="higher"))
    rows = [metrics(val, threshold, "Guam validation (descriptive)")]
    scored = {"Guam_val": val}
    for key in ["Guam_test", "LA_test"]:
        scored[key] = score_dataset(key)
        frame = scored[key]
        if key == "Guam_test":
            for gen, label in [("A", "A"), ("B", "C")]:
                subset = frame[(frame.Generator.eq(gen)) | frame.Attack_Type.eq("normal")]
                rows.append(metrics(subset, threshold, label))
        else:
            for gen, label in [("A", "B"), ("B", "D")]:
                subset = frame[(frame.Generator.eq(gen)) | frame.Attack_Type.eq("normal")]
                rows.append(metrics(subset, threshold, label))
    pd.DataFrame(rows).to_csv(OUT / "cv_kalman_nis_metrics.csv", index=False)
    for key, frame in scored.items():
        frame[["Scenario_ID", "Window_Start_Index", "Window_End_Index", "Window_Label", "Generator", "Attack_Type", "CV_KF_Max_NIS"]].to_csv(
            OUT / f"{key.lower()}_window_scores.csv.gz", index=False, compression="gzip")
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print(f"Threshold={threshold}; saved to {OUT}", flush=True)


if __name__ == "__main__":
    main()
