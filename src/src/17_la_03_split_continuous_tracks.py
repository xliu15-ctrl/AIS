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

INPUT_CLEANED = PROCESSED_DIR / "la_2025_cleaned_ais.csv"

OUTPUT_TRACKS = PROCESSED_DIR / "la_2025_continuous_tracks.csv"
OUTPUT_SUMMARY = RESULT_DIR / "la_step3_track_segmentation_summary.csv"
OUTPUT_TRACK_STATS = RESULT_DIR / "la_step3_track_statistics.csv"
OUTPUT_FIGURE = FIGURE_DIR / "la_step3_track_length_distribution.png"

with open(ROOT / "config" / "experiment_protocol.json", encoding="utf-8") as handle:
    PROTOCOL = json.load(handle)
RULES = PROTOCOL["preprocessing"]
COURSE_RULES = RULES["valid_course"]
MAX_TIME_GAP_MINUTES = RULES["split_track_when_time_gap_minutes_gt"]
MAX_POSITION_DERIVED_SPEED_KNOTS = RULES["split_track_when_calculated_speed_knots_gt"]
MIN_POINTS_PER_TRACK = RULES["drop_track_when_point_count_lt"]

print("=" * 100)
print("LA 2025 CROSS-REGION VALIDATION: STEP 3")
print("Continuous track segmentation")
print("=" * 100)
print("INPUT_CLEANED:", INPUT_CLEANED)
print("OUTPUT_TRACKS:", OUTPUT_TRACKS)

if not INPUT_CLEANED.exists():
    raise FileNotFoundError(f"找不到清洗后的 LA 数据文件：{INPUT_CLEANED}")


# ============================================================
# 2. Distance and bearing helpers
# ============================================================

def haversine_nm(lon1, lat1, lon2, lat2):
    """
    Great-circle distance in nautical miles.
    """
    lon1 = np.radians(lon1.astype(float))
    lat1 = np.radians(lat1.astype(float))
    lon2 = np.radians(lon2.astype(float))
    lat2 = np.radians(lat2.astype(float))

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return 3440.065 * c


def bearing_deg(lon1, lat1, lon2, lat2):
    """
    Bearing from previous point to current point.
    """
    lon1 = np.radians(lon1.astype(float))
    lat1 = np.radians(lat1.astype(float))
    lon2 = np.radians(lon2.astype(float))
    lat2 = np.radians(lat2.astype(float))

    dlon = lon2 - lon1

    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)

    b = np.degrees(np.arctan2(x, y))
    return (b + 360) % 360


def angle_diff_deg(a, b):
    """
    Circular absolute angle difference in degrees.
    """
    return np.abs((a - b + 180) % 360 - 180)


# ============================================================
# 3. Load cleaned LA AIS data
# ============================================================

usecols = [
    "MMSI",
    "Time",
    "SOG",
    "COG",
    "Heading",
    "VesselName",
    "VesselType",
    "Geometry",
    "Longitude",
    "Latitude",
]

print("\nReading cleaned LA AIS data...")
df = pd.read_csv(INPUT_CLEANED, usecols=usecols)

print(f"Raw cleaned rows loaded: {len(df):,}")
print(f"Unique MMSI before processing: {df['MMSI'].nunique():,}")

# Basic type conversion
df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
df["MMSI"] = pd.to_numeric(df["MMSI"], errors="coerce")
df["SOG"] = pd.to_numeric(df["SOG"], errors="coerce")
df["COG"] = pd.to_numeric(df["COG"], errors="coerce")
df["Heading"] = pd.to_numeric(df["Heading"], errors="coerce")
df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")

before_drop = len(df)
df = df.dropna(subset=["MMSI", "Time", "Longitude", "Latitude"])
after_drop = len(df)

print(f"Dropped rows with missing key fields: {before_drop - after_drop:,}")

df["MMSI"] = df["MMSI"].astype("int64")


# ============================================================
# 4. Remove exact duplicate reports
# ============================================================

print("\nDropping exact duplicate AIS reports...")

before_dup = len(df)

df = df.drop_duplicates(
    subset=["MMSI", "Time", "Longitude", "Latitude", "SOG", "COG"],
    keep="first"
)

after_dup = len(df)

print(f"Duplicate rows removed: {before_dup - after_dup:,}")
print(f"Rows after duplicate removal: {len(df):,}")


# ============================================================
# 5. Sort by MMSI and time
# ============================================================

print("\nSorting by MMSI and Time...")
df = df.sort_values(["MMSI", "Time"]).reset_index(drop=True)

print("Sorting finished.")


# ============================================================
# 6. Compute previous-point features
# ============================================================

print("\nComputing consecutive-point features...")

df["Previous_Time"] = df.groupby("MMSI")["Time"].shift(1)
df["Previous_Longitude"] = df.groupby("MMSI")["Longitude"].shift(1)
df["Previous_Latitude"] = df.groupby("MMSI")["Latitude"].shift(1)
df["Previous_SOG"] = df.groupby("MMSI")["SOG"].shift(1)
df["Previous_COG"] = df.groupby("MMSI")["COG"].shift(1)

df["Delta_Minutes"] = (df["Time"] - df["Previous_Time"]).dt.total_seconds() / 60.0

df["Distance_NM"] = haversine_nm(
    df["Previous_Longitude"],
    df["Previous_Latitude"],
    df["Longitude"],
    df["Latitude"],
)

df.loc[df["Delta_Minutes"] <= 0, "Distance_NM"] = np.nan

df["Calculated_Speed_Knots"] = df["Distance_NM"] / (df["Delta_Minutes"] / 60.0)

df["Speed_Residual_Knots"] = np.abs(df["Calculated_Speed_Knots"] - df["SOG"])

df["COG_Change_Deg"] = angle_diff_deg(df["COG"], df["Previous_COG"])

df["COG_Change_Rate_Deg_Per_Min"] = df["COG_Change_Deg"] / df["Delta_Minutes"]

df["Acceleration_Knots_Per_Min"] = np.abs(df["SOG"] - df["Previous_SOG"]) / df["Delta_Minutes"]

df["Calculated_Bearing_Deg"] = bearing_deg(
    df["Previous_Longitude"],
    df["Previous_Latitude"],
    df["Longitude"],
    df["Latitude"],
)

df["Course_Position_Mismatch_Deg"] = angle_diff_deg(
    df["Calculated_Bearing_Deg"],
    df["COG"],
)

valid_course = (
    (df["SOG"] >= COURSE_RULES["min_sog_knots"])
    & (df["Calculated_Speed_Knots"] >= COURSE_RULES["min_calculated_speed_knots"])
    & (df["Distance_NM"] > COURSE_RULES["min_distance_nm"])
)
df["Valid_Course_Feature"] = valid_course

df = df.replace([np.inf, -np.inf], np.nan)

print("Consecutive-point features computed.")


# ============================================================
# 7. Track segmentation
# ============================================================

print("\nSegmenting continuous tracks...")

# Start a new track if:
# 1. first point of each MMSI
# 2. time interval <= 0
# 3. time gap > 30 min
# 4. calculated speed > 60 knots
df["New_Track"] = (
    df["Previous_Time"].isna()
    | df["Delta_Minutes"].isna()
    | (df["Delta_Minutes"] <= 0)
    | (df["Delta_Minutes"] > MAX_TIME_GAP_MINUTES)
    | (df["Calculated_Speed_Knots"] > MAX_POSITION_DERIVED_SPEED_KNOTS)
).astype(int)

df["Local_Track_Number"] = df.groupby("MMSI")["New_Track"].cumsum()

df["Track_ID"] = (
    df["MMSI"].astype(str)
    + "_"
    + df["Local_Track_Number"].astype(int).astype(str).str.zfill(4)
)

# A new segment has no within-track predecessor, even if the cross-segment
# displacement happens to pass the low-speed course gate.
valid_course = valid_course & df["New_Track"].eq(0)
valid_course_change = valid_course & valid_course.groupby(df["Track_ID"]).shift(fill_value=False)
df.loc[~valid_course, "Course_Position_Mismatch_Deg"] = np.nan
df.loc[~valid_course_change, "COG_Change_Rate_Deg_Per_Min"] = np.nan
df["Valid_Course_Feature"] = valid_course

print(f"Initial segmented track count: {df['Track_ID'].nunique():,}")


# ============================================================
# 8. Remove short tracks
# ============================================================

print("\nRemoving short tracks...")

track_sizes = df.groupby("Track_ID").size()
valid_track_ids = track_sizes[track_sizes >= MIN_POINTS_PER_TRACK].index

before_filter_rows = len(df)
before_filter_tracks = df["Track_ID"].nunique()

df = df[df["Track_ID"].isin(valid_track_ids)].copy()

after_filter_rows = len(df)
after_filter_tracks = df["Track_ID"].nunique()

print(f"Rows before short-track filtering: {before_filter_rows:,}")
print(f"Rows after short-track filtering: {after_filter_rows:,}")
print(f"Removed rows: {before_filter_rows - after_filter_rows:,}")
print(f"Tracks before filtering: {before_filter_tracks:,}")
print(f"Tracks after filtering: {after_filter_tracks:,}")


# ============================================================
# 9. Add coordinate key and moving flag
# ============================================================

df["Coordinate_Key"] = (
    df["Longitude"].round(6).astype(str)
    + "_"
    + df["Latitude"].round(6).astype(str)
)

df["Is_Moving_Point"] = (df["SOG"].fillna(0) > 0.5).astype(int)


# ============================================================
# 10. Save segmented track data
# ============================================================

output_cols = [
    "MMSI",
    "Time",
    "SOG",
    "COG",
    "Heading",
    "VesselName",
    "VesselType",
    "Geometry",
    "Longitude",
    "Latitude",
    "Previous_Time",
    "Previous_Longitude",
    "Previous_Latitude",
    "Delta_Minutes",
    "Distance_NM",
    "Calculated_Speed_Knots",
    "New_Track",
    "Local_Track_Number",
    "Track_ID",
    "Previous_SOG",
    "Previous_COG",
    "Speed_Residual_Knots",
    "COG_Change_Deg",
    "COG_Change_Rate_Deg_Per_Min",
    "Acceleration_Knots_Per_Min",
    "Calculated_Bearing_Deg",
    "Course_Position_Mismatch_Deg",
    "Is_Moving_Point",
    "Coordinate_Key",
]

df = df[output_cols]

print("\nSaving segmented tracks...")
df.to_csv(OUTPUT_TRACKS, index=False, encoding="utf-8-sig")
print("Saved to:")
print(OUTPUT_TRACKS)


# ============================================================
# 11. Track statistics
# ============================================================

print("\nComputing track statistics...")

track_stats = (
    df.groupby("Track_ID")
    .agg(
        MMSI=("MMSI", "first"),
        Point_Count=("Track_ID", "size"),
        Start_Time=("Time", "min"),
        End_Time=("Time", "max"),
        Duration_Minutes=("Delta_Minutes", "sum"),
        Total_Distance_NM=("Distance_NM", "sum"),
        Mean_SOG=("SOG", "mean"),
        Moving_Ratio=("Is_Moving_Point", "mean"),
        Unique_Positions=("Coordinate_Key", "nunique"),
        P95_Calculated_Speed_Knots=("Calculated_Speed_Knots", lambda x: np.nanpercentile(x.dropna(), 95) if x.dropna().size > 0 else np.nan),
        Mean_Speed_Residual_Knots=("Speed_Residual_Knots", "mean"),
        Mean_Course_Position_Mismatch_Deg=("Course_Position_Mismatch_Deg", "mean"),
    )
    .reset_index()
)

track_stats.to_csv(OUTPUT_TRACK_STATS, index=False, encoding="utf-8-sig")

print("Track statistics saved to:")
print(OUTPUT_TRACK_STATS)


# ============================================================
# 12. Save summary
# ============================================================

summary = pd.DataFrame([{
    "Input_Cleaned_Row_Count": before_dup,
    "Duplicate_Removed_Count": before_dup - after_dup,
    "Rows_After_Duplicate_Removal": after_dup,
    "Rows_After_Short_Track_Filtering": len(df),
    "Initial_Track_Count": before_filter_tracks,
    "Final_Track_Count": after_filter_tracks,
    "Unique_MMSI_Count": df["MMSI"].nunique(),
    "Min_Points_Per_Track": MIN_POINTS_PER_TRACK,
    "Max_Time_Gap_Minutes": MAX_TIME_GAP_MINUTES,
    "Max_Position_Derived_Speed_Knots": MAX_POSITION_DERIVED_SPEED_KNOTS,
    "Start_Time": df["Time"].min(),
    "End_Time": df["Time"].max(),
}])

summary.to_csv(OUTPUT_SUMMARY, index=False, encoding="utf-8-sig")

print("\n" + "=" * 100)
print("LA STEP 3 TRACK SEGMENTATION SUMMARY")
print("=" * 100)
print(summary.to_string(index=False))

print("\nSummary saved to:")
print(OUTPUT_SUMMARY)


# ============================================================
# 13. Plot track length distribution
# ============================================================

plt.figure(figsize=(8, 6))
plt.hist(track_stats["Point_Count"], bins=60)
plt.xlabel("Track Length / Number of AIS Points")
plt.ylabel("Number of Tracks")
plt.title("LA External Validation Track Length Distribution")
plt.grid(True, alpha=0.35)
plt.tight_layout()
plt.savefig(OUTPUT_FIGURE, dpi=300)
plt.close()

print("\nTrack length distribution figure saved to:")
print(OUTPUT_FIGURE)

print("\nDONE.")
