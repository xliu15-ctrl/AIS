"""Audit AIS reporting intervals and transceiver-class composition by domain.

The public exports do not identify individual receiver stations. LA source-file
counts are retained as file provenance only and are not called receiver sources.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "reviewer_followup_t3"
OUT.mkdir(parents=True, exist_ok=True)
CHUNK = 500_000


def read_domain(domain, files, mmsi_col, time_col, class_col):
    parts = []
    file_counts = []
    for path in files:
        n = 0
        for chunk in pd.read_csv(path, usecols=[mmsi_col, time_col, class_col],
                                 chunksize=CHUNK, low_memory=False):
            n += len(chunk)
            chunk.columns = ["MMSI", "Time", "TransceiverClass"]
            chunk["MMSI"] = pd.to_numeric(chunk["MMSI"], errors="coerce")
            chunk["Time"] = pd.to_datetime(chunk["Time"], errors="coerce", utc=True)
            chunk["TransceiverClass"] = chunk["TransceiverClass"].astype("string").str.upper().str.strip()
            chunk = chunk.dropna(subset=["MMSI", "Time"])
            chunk["MMSI"] = chunk["MMSI"].astype("int64")
            parts.append(chunk[["MMSI", "Time", "TransceiverClass"]])
        file_counts.append({"Domain": domain, "Source_File": path.name, "Rows_Read": n})
        print(f"Read {domain} {path.name}: {n:,}", flush=True)
    data = pd.concat(parts, ignore_index=True)
    data = data.sort_values(["MMSI", "Time"], kind="mergesort")
    # Collapse repeated timestamps within a vessel before calculating positive
    # report gaps; duplicates are not inter-report intervals.
    data = data.drop_duplicates(["MMSI", "Time"], keep="first")
    gaps = data.groupby("MMSI", sort=False)["Time"].diff().dt.total_seconds().div(60)
    valid = gaps[gaps.gt(0) & gaps.notna()]
    # Class labels are reported from the raw transceiver field. Interval values
    # are assigned to the class on the later report in each pair.
    data["Gap_Minutes"] = gaps.to_numpy()
    data["Gap_Minutes"] = data["Gap_Minutes"].where(data["Gap_Minutes"].gt(0))
    return data, valid, file_counts


def interval_stats(values):
    values = pd.Series(values).dropna().to_numpy(dtype=float)
    if not len(values):
        return {"Interval_Count": 0, "Median_Minutes": np.nan,
                "IQR25_Minutes": np.nan, "IQR75_Minutes": np.nan,
                "P95_Minutes": np.nan}
    q = np.quantile(values, [.25, .50, .75, .95])
    return {"Interval_Count": len(values), "Median_Minutes": q[1],
            "IQR25_Minutes": q[0], "IQR75_Minutes": q[2], "P95_Minutes": q[3]}


def main():
    guam = read_domain("Guam", [ROOT / "data/raw/ais_sample.csv"],
                       "mmsi", "base_date_time", "transceiver")
    la_paths = sorted((ROOT / "external_validation/la_2025/raw").glob("la_2025_*.csv"))
    la = read_domain("LA", la_paths, "MMSI", "BaseDateTime", "TransceiverClass")
    summary, classes = [], []
    for domain, (data, _, _) in [("Guam", guam), ("LA", la)]:
        for cls, sub in data.groupby("TransceiverClass", dropna=False, sort=True):
            label = "missing/other" if pd.isna(cls) or cls not in {"A", "B"} else cls
            intervals = sub["Gap_Minutes"]
            row = {"Domain": domain, "Transceiver_Class": label,
                   "Report_Rows": len(sub), "Unique_MMSI": sub.MMSI.nunique(),
                   **interval_stats(intervals)}
            summary.append(row)
            classes.append({"Domain": domain, "Transceiver_Class": label,
                            "Report_Rows": len(sub), "Unique_MMSI": sub.MMSI.nunique(),
                            "Share_of_Rows": len(sub) / len(data),
                            "Share_of_MMSI_Class_Pairs": sub[["MMSI", "TransceiverClass"]].drop_duplicates().shape[0] /
                                data[["MMSI", "TransceiverClass"]].drop_duplicates().shape[0]})
        summary.append({"Domain": domain, "Transceiver_Class": "all classes",
                        "Report_Rows": len(data), "Unique_MMSI": data.MMSI.nunique(),
                        **interval_stats(data.Gap_Minutes)})
    pd.DataFrame(summary).to_csv(OUT / "report_interval_summary.csv", index=False)
    pd.DataFrame(classes).to_csv(OUT / "transceiver_class_composition.csv", index=False)
    pd.DataFrame(guam[2] + la[2]).to_csv(OUT / "source_file_row_counts.csv", index=False)
    meta = {
        "Interval_definition": "positive elapsed minutes between consecutive distinct timestamps per MMSI, across all raw rows",
        "Class_field": {"Guam": "transceiver", "LA": "TransceiverClass"},
        "Receiver_source_availability": "Neither source export includes an identifiable receiver/station field. LA file counts are provenance chunks, not receiver-source composition.",
        "Missingness": "Rows with missing MMSI/time excluded; duplicate MMSI-time rows collapsed before interval calculation.",
        "Files": [str(p.relative_to(ROOT)) for p in [ROOT / "data/raw/ais_sample.csv", *la_paths]],
    }
    (OUT / "method_and_scope.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(pd.DataFrame(summary).to_string(index=False), flush=True)
    print(f"Saved to {OUT}", flush=True)


if __name__ == "__main__":
    main()
