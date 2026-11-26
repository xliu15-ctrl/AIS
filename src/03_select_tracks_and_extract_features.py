from pathlib import Path
import json

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


# ============================================================
# 1. 设置文件路径
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ais_continuous_tracks.csv"
)

OUTPUT_FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ais_features_all_tracks.csv"
)

OUTPUT_CANDIDATE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "candidate_nominal_tracks.csv"
)

OUTPUT_SUMMARY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "candidate_track_summary.csv"
)

PROTOCOL_FILE = PROJECT_ROOT / "config" / "experiment_protocol.json"
PREPROCESS_AUDIT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "preprocessing_audit_03_features.csv"
)

FIGURE_DIR = PROJECT_ROOT / "results" / "figures"

OUTPUT_TRACK_FIGURE = FIGURE_DIR / "candidate_tracks_example.png"
OUTPUT_QUALITY_FIGURE = FIGURE_DIR / "track_quality_scatter.png"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

with open(PROTOCOL_FILE, "r", encoding="utf-8") as file:
    PROTOCOL = json.load(file)

PREPROCESSING_RULES = PROTOCOL["preprocessing"]
VALID_COURSE_RULES = PREPROCESSING_RULES["valid_course"]


# ============================================================
# 2. 当前筛选条件
# ============================================================
MIN_POINTS_PER_TRACK = PREPROCESSING_RULES["candidate_min_points"]
MIN_DURATION_MINUTES = PREPROCESSING_RULES["candidate_min_duration_minutes"]
MIN_TOTAL_DISTANCE_NM = PREPROCESSING_RULES["candidate_min_total_distance_nm"]
MIN_MOVING_POINT_RATIO = PREPROCESSING_RULES["candidate_min_moving_point_ratio"]
MIN_UNIQUE_POSITIONS = PREPROCESSING_RULES["candidate_min_unique_positions"]
MAX_P95_CALCULATED_SPEED_KNOTS = PREPROCESSING_RULES[
    "candidate_max_p95_calculated_speed_knots"
]

MOVING_SPEED_THRESHOLD_KNOTS = PREPROCESSING_RULES[
    "moving_speed_threshold_knots"
]


# ============================================================
# 3. 检查输入文件
# ============================================================
if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"没有找到文件：{INPUT_FILE}\n"
        "请先运行 02_split_continuous_tracks.py。"
    )


# ============================================================
# 4. 读取连续轨迹数据
# ============================================================
data = pd.read_csv(INPUT_FILE, low_memory=False)

required_columns = [
    "Track_ID",
    "MMSI",
    "Time",
    "Longitude",
    "Latitude",
    "SOG",
    "COG",
    "VesselName"
]

missing_columns = [
    column for column in required_columns
    if column not in data.columns
]

if missing_columns:
    raise ValueError(
        f"文件缺少必要字段：{missing_columns}\n"
        f"实际字段为：{list(data.columns)}"
    )

data["Time"] = pd.to_datetime(data["Time"], errors="coerce")

numeric_columns = [
    "MMSI",
    "Longitude",
    "Latitude",
    "SOG",
    "COG"
]

for column in numeric_columns:
    data[column] = pd.to_numeric(data[column], errors="coerce")

data = data.dropna(
    subset=["Track_ID", "MMSI", "Time", "Longitude", "Latitude"]
)

data = data.sort_values(["Track_ID", "Time"]).reset_index(drop=True)

print("=" * 75)
print("第三阶段开始：筛选候选轨迹并提取运动学特征")
print(f"输入连续轨迹数据点数量：{len(data)}")
print(f"输入轨迹片段数量：{data['Track_ID'].nunique()}")
print("=" * 75)


# ============================================================
# 5. 定义距离、方向与角度差函数
# ============================================================
preprocessing_audit_rows = []


def append_audit_step(step_name, track_ids=None):
    if track_ids is None:
        frame = data
    else:
        frame = data[data["Track_ID"].isin(track_ids)]

    preprocessing_audit_rows.append(
        {
            "Stage": "03_select_tracks_and_extract_features",
            "Step": step_name,
            "Track_Count": int(frame["Track_ID"].nunique()),
            "MMSI_Count": int(frame["MMSI"].nunique()),
            "AIS_Point_Count": int(len(frame))
        }
    )


append_audit_step("input_continuous_tracks")


def haversine_distance_nm(lon1, lat1, lon2, lat2):
    """
    计算两个经纬度点之间的球面距离。
    返回单位：海里 nautical mile。
    """
    earth_radius_nm = 3440.065

    lon1_rad = np.radians(lon1)
    lat1_rad = np.radians(lat1)
    lon2_rad = np.radians(lon2)
    lat2_rad = np.radians(lat2)

    delta_lon = lon2_rad - lon1_rad
    delta_lat = lat2_rad - lat1_rad

    a = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1_rad)
        * np.cos(lat2_rad)
        * np.sin(delta_lon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    return earth_radius_nm * c


def calculate_bearing_deg(lon1, lat1, lon2, lat2):
    """
    根据相邻两个经纬度点计算移动方向。
    返回范围：0 到 360 度。
    """
    lon1_rad = np.radians(lon1)
    lat1_rad = np.radians(lat1)
    lon2_rad = np.radians(lon2)
    lat2_rad = np.radians(lat2)

    delta_lon = lon2_rad - lon1_rad

    x = np.sin(delta_lon) * np.cos(lat2_rad)
    y = (
        np.cos(lat1_rad) * np.sin(lat2_rad)
        - np.sin(lat1_rad)
        * np.cos(lat2_rad)
        * np.cos(delta_lon)
    )

    bearing = np.degrees(np.arctan2(x, y))

    return (bearing + 360) % 360


def angular_difference_deg(angle_a, angle_b):
    """
    计算两个角度之间的最小绝对差值。
    例如 359 度与 1 度的差值应为 2 度，而不是 358 度。
    """
    return np.abs((angle_a - angle_b + 180) % 360 - 180)


# ============================================================
# 6. 计算相邻点基础信息
# ============================================================
grouped = data.groupby("Track_ID")

data["Previous_Time"] = grouped["Time"].shift(1)
data["Previous_Longitude"] = grouped["Longitude"].shift(1)
data["Previous_Latitude"] = grouped["Latitude"].shift(1)
data["Previous_SOG"] = grouped["SOG"].shift(1)
data["Previous_COG"] = grouped["COG"].shift(1)

data["Delta_Minutes"] = (
    data["Time"] - data["Previous_Time"]
).dt.total_seconds() / 60

data["Distance_NM"] = haversine_distance_nm(
    data["Previous_Longitude"],
    data["Previous_Latitude"],
    data["Longitude"],
    data["Latitude"]
)

data["Calculated_Speed_Knots"] = (
    data["Distance_NM"]
    / (data["Delta_Minutes"] / 60)
)

invalid_time_interval = (
    data["Delta_Minutes"].isna()
    | (data["Delta_Minutes"] <= 0)
)

data.loc[
    invalid_time_interval,
    ["Distance_NM", "Calculated_Speed_Knots"]
] = np.nan


# ============================================================
# 7. 提取检测位置欺骗需要的特征
# ============================================================
data["Speed_Residual_Knots"] = np.abs(
    data["Calculated_Speed_Knots"] - data["SOG"]
)

data["COG_Change_Deg"] = angular_difference_deg(
    data["COG"],
    data["Previous_COG"]
)

data["COG_Change_Rate_Deg_Per_Min"] = (
    data["COG_Change_Deg"]
    / data["Delta_Minutes"]
)

data["Acceleration_Knots_Per_Min"] = np.abs(
    data["SOG"] - data["Previous_SOG"]
) / data["Delta_Minutes"]

data["Calculated_Bearing_Deg"] = calculate_bearing_deg(
    data["Previous_Longitude"],
    data["Previous_Latitude"],
    data["Longitude"],
    data["Latitude"]
)

data["Course_Position_Mismatch_Deg"] = angular_difference_deg(
    data["Calculated_Bearing_Deg"],
    data["COG"]
)

valid_course = (
    (data["SOG"] >= VALID_COURSE_RULES["min_sog_knots"])
    & (
        data["Calculated_Speed_Knots"]
        >= VALID_COURSE_RULES["min_calculated_speed_knots"]
    )
    & (data["Distance_NM"] > VALID_COURSE_RULES["min_distance_nm"])
)

valid_course_change = valid_course & valid_course.groupby(data["Track_ID"]).shift(fill_value=False)
data.loc[~valid_course, "Course_Position_Mismatch_Deg"] = np.nan
data.loc[~valid_course_change, "COG_Change_Rate_Deg_Per_Min"] = np.nan

data["Valid_Course_Feature"] = valid_course

data["Is_Moving_Point"] = (
    data["SOG"].fillna(0) > MOVING_SPEED_THRESHOLD_KNOTS
)

# 经纬度保留到小数点后六位后判断是否为不同位置
data["Coordinate_Key"] = (
    data["Longitude"].round(6).astype(str)
    + "_"
    + data["Latitude"].round(6).astype(str)
)


# ============================================================
# 8. 统计每一个轨迹片段的质量
# ============================================================
summary = (
    data.groupby("Track_ID")
    .agg(
        MMSI=("MMSI", "first"),
        VesselName=(
            "VesselName",
            lambda values: (
                str(values.dropna().iloc[0]).strip()
                if not values.dropna().empty
                else ""
            )
        ),
        Point_Count=("Track_ID", "size"),
        Start_Time=("Time", "min"),
        End_Time=("Time", "max"),
        Total_Distance_NM=("Distance_NM", "sum"),
        Mean_SOG=("SOG", "mean"),
        Max_SOG=("SOG", "max"),
        Moving_Point_Ratio=("Is_Moving_Point", "mean"),
        Unique_Positions=("Coordinate_Key", "nunique"),
        Median_Time_Gap_Minutes=("Delta_Minutes", "median"),
        P95_Calculated_Speed_Knots=(
            "Calculated_Speed_Knots",
            lambda values: values.quantile(0.95)
        ),
        Mean_Speed_Residual_Knots=("Speed_Residual_Knots", "mean"),
        Mean_Course_Mismatch_Deg=(
            "Course_Position_Mismatch_Deg",
            "mean"
        )
    )
    .reset_index()
)

summary["Duration_Minutes"] = (
    summary["End_Time"] - summary["Start_Time"]
).dt.total_seconds() / 60


# ============================================================
# 9. 根据质量规则选择候选基线轨迹
# ============================================================
summary["Pass_Point_Count"] = (
    summary["Point_Count"] >= MIN_POINTS_PER_TRACK
)

summary["Pass_Duration"] = (
    summary["Duration_Minutes"] >= MIN_DURATION_MINUTES
)

summary["Pass_Distance"] = (
    summary["Total_Distance_NM"] >= MIN_TOTAL_DISTANCE_NM
)

summary["Pass_Moving_Ratio"] = (
    summary["Moving_Point_Ratio"] >= MIN_MOVING_POINT_RATIO
)

summary["Pass_Unique_Positions"] = (
    summary["Unique_Positions"] >= MIN_UNIQUE_POSITIONS
)

summary["Pass_P95_Calculated_Speed"] = (
    summary["P95_Calculated_Speed_Knots"]
    <= MAX_P95_CALCULATED_SPEED_KNOTS
)

summary["Selected_As_Candidate"] = (
    summary["Pass_Point_Count"]
    & summary["Pass_Duration"]
    & summary["Pass_Distance"]
    & summary["Pass_Moving_Ratio"]
    & summary["Pass_Unique_Positions"]
    & summary["Pass_P95_Calculated_Speed"]
)

candidate_summary = summary[
    summary["Selected_As_Candidate"]
].copy()

candidate_summary = candidate_summary.sort_values(
    ["Total_Distance_NM", "Point_Count"],
    ascending=False
)

candidate_track_ids = candidate_summary["Track_ID"]

candidate_data = data[
    data["Track_ID"].isin(candidate_track_ids)
].copy()


# ============================================================
# 10. 打印筛选过程
# ============================================================
print("轨迹筛选结果：")
print(f"原始有效轨迹数量：{len(summary)}")
print(
    f"满足至少 {MIN_POINTS_PER_TRACK} 个点："
    f"{summary['Pass_Point_Count'].sum()}"
)
print(
    f"满足至少 {MIN_DURATION_MINUTES} 分钟："
    f"{summary['Pass_Duration'].sum()}"
)
print(
    f"满足至少 {MIN_TOTAL_DISTANCE_NM} 海里移动距离："
    f"{summary['Pass_Distance'].sum()}"
)
print(
    f"满足移动点比例至少 {MIN_MOVING_POINT_RATIO:.0%}："
    f"{summary['Pass_Moving_Ratio'].sum()}"
)
print(
    f"满足至少 {MIN_UNIQUE_POSITIONS} 个不同位置："
    f"{summary['Pass_Unique_Positions'].sum()}"
)
print(
    f"满足最终 P95 推算航速不超过 "
    f"{MAX_P95_CALCULATED_SPEED_KNOTS} 节："
    f"{summary['Pass_P95_Calculated_Speed'].sum()}"
)
print("-" * 75)
print(f"最终候选基线轨迹数量：{len(candidate_summary)}")
print(f"最终候选基线数据点数量：{len(candidate_data)}")
print("=" * 75)

filter_steps = [
    ("after_candidate_min_points", "Pass_Point_Count"),
    ("after_candidate_min_duration", "Pass_Duration"),
    ("after_candidate_min_total_distance", "Pass_Distance"),
    ("after_candidate_min_moving_point_ratio", "Pass_Moving_Ratio"),
    ("after_candidate_min_unique_positions", "Pass_Unique_Positions"),
    (
        "after_candidate_max_p95_calculated_speed",
        "Pass_P95_Calculated_Speed"
    )
]

current_mask = pd.Series(True, index=summary.index)

for step_name, column in filter_steps:
    current_mask = current_mask & summary[column]
    append_audit_step(
        step_name,
        set(summary.loc[current_mask, "Track_ID"])
    )

if len(candidate_summary) == 0:
    raise ValueError(
        "当前筛选规则下没有保留候选轨迹。\n"
        "请检查输入字段和各筛选步骤的保留数量。"
    )

print("航行距离最长的前 10 个候选轨迹：")
print(
    candidate_summary[
        [
            "Track_ID",
            "VesselName",
            "Point_Count",
            "Duration_Minutes",
            "Total_Distance_NM",
            "Moving_Point_Ratio",
            "Mean_SOG"
        ]
    ].head(10).to_string(index=False)
)
print("=" * 75)


# ============================================================
# 11. 保存数据文件
# ============================================================
data.to_csv(
    OUTPUT_FEATURE_FILE,
    index=False,
    encoding="utf-8-sig"
)

candidate_data.to_csv(
    OUTPUT_CANDIDATE_FILE,
    index=False,
    encoding="utf-8-sig"
)

candidate_summary.to_csv(
    OUTPUT_SUMMARY_FILE,
    index=False,
    encoding="utf-8-sig"
)

pd.DataFrame(preprocessing_audit_rows).to_csv(
    PREPROCESS_AUDIT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print(f"全部轨迹特征数据已保存到：{OUTPUT_FEATURE_FILE}")
print(f"候选基线轨迹数据已保存到：{OUTPUT_CANDIDATE_FILE}")
print(f"候选轨迹统计表已保存到：{OUTPUT_SUMMARY_FILE}")
print(f"预处理审计表已保存到：{PREPROCESS_AUDIT_FILE}")
print("=" * 75)


# ============================================================
# 12. 绘制候选轨迹示例图
# ============================================================
top_track_ids = candidate_summary.head(5)["Track_ID"]

plot_data = candidate_data[
    candidate_data["Track_ID"].isin(top_track_ids)
].copy()

plt.figure(figsize=(10, 8))

for track_id, track in plot_data.groupby("Track_ID"):
    track = track.sort_values("Time")

    vessel_name = track["VesselName"].dropna()

    if len(vessel_name) > 0:
        label = f"{str(vessel_name.iloc[0]).strip()} | {track_id}"
    else:
        label = track_id

    plt.plot(
        track["Longitude"],
        track["Latitude"],
        marker=".",
        markersize=2,
        linewidth=1,
        label=label
    )

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Selected Nominal AIS Track Segments for Further Experiments")
plt.legend(fontsize=7)
plt.grid(True)
plt.axis("equal")
plt.tight_layout()
plt.savefig(OUTPUT_TRACK_FIGURE, dpi=300)
plt.close()


# ============================================================
# 13. 绘制轨迹质量筛选散点图
# ============================================================
not_selected = summary[
    ~summary["Selected_As_Candidate"]
]

selected = summary[
    summary["Selected_As_Candidate"]
]

plt.figure(figsize=(10, 7))

plt.scatter(
    not_selected["Point_Count"],
    not_selected["Total_Distance_NM"],
    alpha=0.45,
    label="Not selected"
)

plt.scatter(
    selected["Point_Count"],
    selected["Total_Distance_NM"],
    alpha=0.65,
    label="Selected candidates"
)

plt.xlabel("Number of AIS Points in a Track Segment")
plt.ylabel("Total Travelled Distance (Nautical Miles)")
plt.title("Quality Screening of Nominal AIS Track Segments")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_QUALITY_FIGURE, dpi=300)
plt.close()

print(f"候选轨迹示例图已保存到：{OUTPUT_TRACK_FIGURE}")
print(f"轨迹质量筛选图已保存到：{OUTPUT_QUALITY_FIGURE}")
print("第三阶段完成：候选基线轨迹已经筛选，运动学特征已经提取。")
