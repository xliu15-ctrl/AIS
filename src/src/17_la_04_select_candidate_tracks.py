from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cbook
import matplotlib.axes._base as _axes_base
cbook._check_1d = lambda values: np.atleast_1d(np.asarray(values))
_axes_base._check_1d = cbook._check_1d

warnings.filterwarnings("ignore")


# ============================================================
# 1. Path settings
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

EXT_ROOT = ROOT / "external_validation" / "la_2025"

PROCESSED_DIR = EXT_ROOT / "processed"
RESULT_DIR = EXT_ROOT / "results"
FIGURE_DIR = EXT_ROOT / "figures"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

INPUT_TRACKS = PROCESSED_DIR / "la_2025_continuous_tracks.csv"
INPUT_TRACK_STATS = RESULT_DIR / "la_step3_track_statistics.csv"

OUTPUT_CANDIDATE_TRACKS = PROCESSED_DIR / "la_2025_candidate_nominal_tracks.csv"
OUTPUT_CANDIDATE_STATS = RESULT_DIR / "la_step4_candidate_track_statistics.csv"
OUTPUT_SELECTION_SUMMARY = RESULT_DIR / "la_step4_candidate_selection_summary.csv"
OUTPUT_FIGURE = FIGURE_DIR / "la_step4_candidate_track_length_distribution.png"

# ============================================================
# 2. Candidate selection parameters
# ============================================================

with open(ROOT / "config" / "experiment_protocol.json", encoding="utf-8") as handle:
    PROTOCOL = json.load(handle)
RULES = PROTOCOL["preprocessing"]
MIN_POINTS_PER_CANDIDATE_TRACK = RULES["candidate_min_points"]
MIN_DURATION_MINUTES = RULES["candidate_min_duration_minutes"]
MIN_TOTAL_DISTANCE_NM = RULES["candidate_min_total_distance_nm"]
MIN_MOVING_RATIO = RULES["candidate_min_moving_point_ratio"]
MIN_UNIQUE_POSITIONS = RULES["candidate_min_unique_positions"]
MAX_P95_CALCULATED_SPEED_KNOTS = RULES["candidate_max_p95_calculated_speed_knots"]

# LA 数据规模较大，为避免后续合成文件过大，外部验证先固定抽取 1000 条候选基线轨迹。
# 如果你想用全部候选轨迹，把这个值改成 None。
MAX_EXTERNAL_BASELINE_TRACKS = 1000

# 为避免少数 MMSI 贡献过多轨迹，先限制每个 MMSI 最多贡献的轨迹数
MAX_TRACKS_PER_MMSI_BEFORE_SAMPLING = 3

RANDOM_STATE = 42
CHUNKSIZE = 300000

print("=" * 100)
print("LA 2025 CROSS-REGION VALIDATION: STEP 4")
print("Candidate nominal track selection")
print("=" * 100)
print("INPUT_TRACKS:", INPUT_TRACKS)
print("INPUT_TRACK_STATS:", INPUT_TRACK_STATS)
print("OUTPUT_CANDIDATE_TRACKS:", OUTPUT_CANDIDATE_TRACKS)

if not INPUT_TRACKS.exists():
    raise FileNotFoundError(f"找不到连续轨迹文件：{INPUT_TRACKS}")

if not INPUT_TRACK_STATS.exists():
    raise FileNotFoundError(f"找不到轨迹统计文件：{INPUT_TRACK_STATS}")


# ============================================================
# 3. Load track-level statistics
# ============================================================

print("\nReading LA track statistics...")
track_stats = pd.read_csv(INPUT_TRACK_STATS)

print(f"Total continuous tracks from step 3: {len(track_stats):,}")
print(f"Unique MMSI in track stats: {track_stats['MMSI'].nunique():,}")

required_cols = [
    "Track_ID",
    "MMSI",
    "Point_Count",
    "Duration_Minutes",
    "Total_Distance_NM",
    "Moving_Ratio",
    "Unique_Positions",
    "P95_Calculated_Speed_Knots",
    "Mean_Speed_Residual_Knots",
    "Mean_Course_Position_Mismatch_Deg",
]

missing_cols = [c for c in required_cols if c not in track_stats.columns]
if missing_cols:
    raise ValueError(f"轨迹统计文件缺少列：{missing_cols}")


# ============================================================
# 4. Apply candidate nominal track filters
# ============================================================

print("\nApplying candidate nominal track filters...")

initial_count = len(track_stats)

cond_point_count = track_stats["Point_Count"] >= MIN_POINTS_PER_CANDIDATE_TRACK
cond_duration = track_stats["Duration_Minutes"] >= MIN_DURATION_MINUTES
cond_distance = track_stats["Total_Distance_NM"] >= MIN_TOTAL_DISTANCE_NM
cond_moving = track_stats["Moving_Ratio"] >= MIN_MOVING_RATIO
cond_unique_pos = track_stats["Unique_Positions"] >= MIN_UNIQUE_POSITIONS
cond_speed = track_stats["P95_Calculated_Speed_Knots"] <= MAX_P95_CALCULATED_SPEED_KNOTS

candidate_stats = track_stats[
    cond_point_count
    & cond_duration
    & cond_distance
    & cond_moving
    & cond_unique_pos
    & cond_speed
].copy()

print(f"Initial track count: {initial_count:,}")
print(f"After candidate filters: {len(candidate_stats):,}")

audit_rows = []
cumulative = pd.Series(True, index=track_stats.index)
audit_rows.append({
    "Filter": "Initial continuous tracks",
    "Remaining_Tracks": int(len(track_stats)),
    "Remaining_MMSI": int(track_stats["MMSI"].nunique()),
    "Remaining_AIS_Points": int(track_stats["Point_Count"].sum())
})
for filter_name, condition in [
    (f"Point_Count >= {MIN_POINTS_PER_CANDIDATE_TRACK}", cond_point_count),
    (f"Duration_Minutes >= {MIN_DURATION_MINUTES}", cond_duration),
    (f"Total_Distance_NM >= {MIN_TOTAL_DISTANCE_NM}", cond_distance),
    (f"Moving_Ratio >= {MIN_MOVING_RATIO}", cond_moving),
    (f"Unique_Positions >= {MIN_UNIQUE_POSITIONS}", cond_unique_pos),
    (f"P95_Calculated_Speed_Knots <= {MAX_P95_CALCULATED_SPEED_KNOTS}", cond_speed)
]:
    cumulative &= condition
    remaining = track_stats[cumulative]
    audit_rows.append({
        "Filter": filter_name,
        "Remaining_Tracks": int(len(remaining)),
        "Remaining_MMSI": int(remaining["MMSI"].nunique()),
        "Remaining_AIS_Points": int(remaining["Point_Count"].sum())
    })
filter_summary = pd.DataFrame(audit_rows)
filter_summary.to_csv(
    RESULT_DIR / "la_preprocessing_filter_audit.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nFilter summary:")
print(filter_summary.to_string(index=False))


# ============================================================
# 5. Limit tracks per MMSI before final sampling
# ============================================================

print("\nLimiting maximum tracks per MMSI before final external sampling...")

candidate_stats = candidate_stats.sort_values(
    by=["MMSI", "Point_Count"],
    ascending=[True, False]
).copy()

candidate_limited = (
    candidate_stats
    .groupby("MMSI", group_keys=False)
    .head(MAX_TRACKS_PER_MMSI_BEFORE_SAMPLING)
    .copy()
)

print(f"Candidate tracks after per-MMSI limiting: {len(candidate_limited):,}")
print(f"Unique MMSI after per-MMSI limiting: {candidate_limited['MMSI'].nunique():,}")


# ============================================================
# 6. Final external baseline sampling
# ============================================================

if MAX_EXTERNAL_BASELINE_TRACKS is not None and len(candidate_limited) > MAX_EXTERNAL_BASELINE_TRACKS:
    print(f"\nSampling {MAX_EXTERNAL_BASELINE_TRACKS:,} external baseline tracks...")

    final_candidate_stats = candidate_limited.sample(
        n=MAX_EXTERNAL_BASELINE_TRACKS,
        random_state=RANDOM_STATE,
    ).copy()

else:
    print("\nUsing all candidate tracks after filtering.")
    final_candidate_stats = candidate_limited.copy()

final_candidate_stats = final_candidate_stats.sort_values(["MMSI", "Track_ID"]).reset_index(drop=True)

final_track_ids = set(final_candidate_stats["Track_ID"].astype(str))

print(f"Final external baseline Track_ID count: {len(final_track_ids):,}")
print(f"Final external baseline unique MMSI count: {final_candidate_stats['MMSI'].nunique():,}")

final_candidate_stats.to_csv(OUTPUT_CANDIDATE_STATS, index=False, encoding="utf-8-sig")

print("\nCandidate track statistics saved to:")
print(OUTPUT_CANDIDATE_STATS)


# ============================================================
# 7. Extract candidate track rows from continuous track file
# ============================================================

print("\nExtracting candidate track rows from continuous track file...")

if OUTPUT_CANDIDATE_TRACKS.exists():
    print("Removing old candidate output:")
    print(OUTPUT_CANDIDATE_TRACKS)
    OUTPUT_CANDIDATE_TRACKS.unlink()

header_written = False
total_rows_written = 0

for chunk_idx, chunk in enumerate(pd.read_csv(INPUT_TRACKS, chunksize=CHUNKSIZE), start=1):
    chunk["Track_ID"] = chunk["Track_ID"].astype(str)

    selected = chunk[chunk["Track_ID"].isin(final_track_ids)].copy()

    if len(selected) > 0:
        selected.to_csv(
            OUTPUT_CANDIDATE_TRACKS,
            mode="a",
            header=not header_written,
            index=False,
            encoding="utf-8-sig",
        )
        header_written = True
        total_rows_written += len(selected)

    print(
        f"  chunk {chunk_idx}: input={len(chunk):,}, "
        f"selected={len(selected):,}, total_written={total_rows_written:,}"
    )

print("\nCandidate nominal track rows saved to:")
print(OUTPUT_CANDIDATE_TRACKS)
print(f"Total candidate rows written: {total_rows_written:,}")


# ============================================================
# 8. Verify extracted candidate file
# ============================================================

print("\nVerifying candidate nominal track file...")

candidate_df_check = pd.read_csv(
    OUTPUT_CANDIDATE_TRACKS,
    usecols=["MMSI", "Track_ID", "Time", "Longitude", "Latitude"]
)

candidate_row_count = len(candidate_df_check)
candidate_track_count = candidate_df_check["Track_ID"].nunique()
candidate_mmsi_count = candidate_df_check["MMSI"].nunique()

print(f"Candidate row count: {candidate_row_count:,}")
print(f"Candidate unique Track_ID: {candidate_track_count:,}")
print(f"Candidate unique MMSI: {candidate_mmsi_count:,}")

missing_track_ids = final_track_ids - set(candidate_df_check["Track_ID"].astype(str).unique())
extra_track_ids = set(candidate_df_check["Track_ID"].astype(str).unique()) - final_track_ids

print(f"Missing selected Track_ID in extracted file: {len(missing_track_ids):,}")
print(f"Extra Track_ID in extracted file: {len(extra_track_ids):,}")

if len(missing_track_ids) > 0:
    print("Missing examples:", list(missing_track_ids)[:20])

if len(extra_track_ids) > 0:
    print("Extra examples:", list(extra_track_ids)[:20])


# ============================================================
# 9. Save final selection summary
# ============================================================

selection_summary = pd.DataFrame([{
    "Initial_Continuous_Track_Count": initial_count,
    "Tracks_After_All_Filters": len(candidate_stats),
    "Tracks_After_Per_MMSI_Limit": len(candidate_limited),
    "Final_External_Baseline_Track_Count": candidate_track_count,
    "Final_External_Baseline_MMSI_Count": candidate_mmsi_count,
    "Final_External_Baseline_Row_Count": candidate_row_count,
    "Max_External_Baseline_Tracks": MAX_EXTERNAL_BASELINE_TRACKS,
    "Max_Tracks_Per_MMSI_Before_Sampling": MAX_TRACKS_PER_MMSI_BEFORE_SAMPLING,
    "Min_Points_Per_Candidate_Track": MIN_POINTS_PER_CANDIDATE_TRACK,
    "Min_Duration_Minutes": MIN_DURATION_MINUTES,
    "Min_Total_Distance_NM": MIN_TOTAL_DISTANCE_NM,
    "Min_Moving_Ratio": MIN_MOVING_RATIO,
    "Min_Unique_Positions": MIN_UNIQUE_POSITIONS,
    "Max_P95_Calculated_Speed_Knots": MAX_P95_CALCULATED_SPEED_KNOTS,
    "Missing_Track_ID_Count": len(missing_track_ids),
    "Extra_Track_ID_Count": len(extra_track_ids),
}])

selection_summary.to_csv(OUTPUT_SELECTION_SUMMARY, index=False, encoding="utf-8-sig")

print("\n" + "=" * 100)
print("LA STEP 4 CANDIDATE SELECTION SUMMARY")
print("=" * 100)
print(selection_summary.to_string(index=False))

print("\nSelection summary saved to:")
print(OUTPUT_SELECTION_SUMMARY)


# ============================================================
# 10. Plot final candidate track length distribution
# ============================================================

plt.figure(figsize=(8, 6))
plt.hist(final_candidate_stats["Point_Count"], bins=50)
plt.xlabel("Track Length / Number of AIS Points")
plt.ylabel("Number of Candidate Tracks")
plt.title("LA External Validation Candidate Track Length Distribution")
plt.grid(True, alpha=0.35)
plt.tight_layout()
plt.savefig(OUTPUT_FIGURE, dpi=300)
plt.close()

print("\nCandidate track length figure saved to:")
print(OUTPUT_FIGURE)

print("\nDONE.")
