from pathlib import Path
import json

import numpy as np
import pandas as pd
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

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "ais_clean_sample.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "ais_continuous_tracks.csv"
SUMMARY_FILE = PROJECT_ROOT / "data" / "processed" / "track_summary.csv"
PROTOCOL_FILE = PROJECT_ROOT / "config" / "experiment_protocol.json"
PREPROCESS_AUDIT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "preprocessing_audit_02_split.csv"
)

FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
OUTPUT_FIGURE = FIGURE_DIR / "continuous_tracks_example.png"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

with open(PROTOCOL_FILE, "r", encoding="utf-8") as file:
    PROTOCOL = json.load(file)

PREPROCESSING_RULES = PROTOCOL["preprocessing"]


# ============================================================
# 2. 检查上一步生成的数据是否存在
# ============================================================
if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"没有找到输入文件：{INPUT_FILE}\n"
        "请先成功运行 01_inspect_and_plot.py，"
        "生成 ais_clean_sample.csv。"
    )


# ============================================================
# 3. 读取清洗后的 AIS 数据
# ============================================================
data = pd.read_csv(INPUT_FILE, low_memory=False)

data["Time"] = pd.to_datetime(data["Time"], errors="coerce")

numeric_columns = ["MMSI", "SOG", "COG", "Longitude", "Latitude"]

for column in numeric_columns:
    if column in data.columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

data = data.dropna(
    subset=["MMSI", "Time", "Longitude", "Latitude"]
)

data = data.sort_values(["MMSI", "Time"]).reset_index(drop=True)

print("=" * 70)
print("连续航迹切分程序开始运行")
print(f"输入数据行数：{len(data)}")
print(f"船舶数量：{data['MMSI'].nunique()}")
print("=" * 70)


# ============================================================
# 4. 定义 Haversine 距离计算函数
# ============================================================
preprocessing_audit_rows = []


def append_audit_step(step_name, frame):
    preprocessing_audit_rows.append(
        {
            "Stage": "02_split_continuous_tracks",
            "Step": step_name,
            "Track_Count": (
                int(frame["Track_ID"].nunique())
                if "Track_ID" in frame.columns
                else np.nan
            ),
            "MMSI_Count": int(frame["MMSI"].nunique()),
            "AIS_Point_Count": int(len(frame))
        }
    )


append_audit_step("input_after_basic_cleaning", data)


def haversine_distance_nm(lon1, lat1, lon2, lat2):
    """
    根据经纬度计算两点之间的球面距离。
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


# ============================================================
# 5. 计算每艘船相邻 AIS 点之间的时间差和距离
# ============================================================
data["Previous_Time"] = data.groupby("MMSI")["Time"].shift(1)
data["Previous_Longitude"] = data.groupby("MMSI")["Longitude"].shift(1)
data["Previous_Latitude"] = data.groupby("MMSI")["Latitude"].shift(1)

data["Delta_Minutes"] = (
    data["Time"] - data["Previous_Time"]
).dt.total_seconds() / 60

data["Distance_NM"] = haversine_distance_nm(
    data["Previous_Longitude"],
    data["Previous_Latitude"],
    data["Longitude"],
    data["Latitude"]
)

# 根据相邻点距离和时间间隔，计算推算航速
# 1 节 = 1 海里 / 小时
data["Calculated_Speed_Knots"] = (
    data["Distance_NM"] / (data["Delta_Minutes"] / 60)
)

# 每艘船的第一条记录没有前一个点，因此速度为空
data.loc[data["Delta_Minutes"] <= 0, "Calculated_Speed_Knots"] = np.nan


# ============================================================
# 6. 根据时间断裂和极端速度切分航迹
# ============================================================
MAX_TIME_GAP_MINUTES = PREPROCESSING_RULES[
    "split_track_when_time_gap_minutes_gt"
]
MAX_CALCULATED_SPEED_KNOTS = PREPROCESSING_RULES[
    "split_track_when_calculated_speed_knots_gt"
]
MIN_POINTS_PER_TRACK = PREPROCESSING_RULES[
    "drop_track_when_point_count_lt"
]

# 新轨迹的判定条件：
# 1. 每艘船的第一条记录；
# 2. 相邻记录时间间隔超过 30 分钟；
# 3. 推算速度超过 60 节；
# 4. 时间顺序异常或相同时间重复。
data["New_Track"] = (
    data["Previous_Time"].isna()
    | (data["Delta_Minutes"] > MAX_TIME_GAP_MINUTES)
    | (data["Delta_Minutes"] <= 0)
    | (data["Calculated_Speed_Knots"] > MAX_CALCULATED_SPEED_KNOTS)
)

# 在每一艘船内部累计编号
data["Local_Track_Number"] = (
    data.groupby("MMSI")["New_Track"].cumsum().astype(int)
)

# 形成全局唯一的轨迹编号
data["Track_ID"] = (
    data["MMSI"].astype("Int64").astype(str)
    + "_"
    + data["Local_Track_Number"].astype(str).str.zfill(4)
)


# ============================================================
# 7. 统计每一个轨迹片段的质量
# ============================================================
track_summary = (
    data.groupby("Track_ID")
    .agg(
        MMSI=("MMSI", "first"),
        VesselName=("VesselName", "first"),
        Point_Count=("Track_ID", "size"),
        Start_Time=("Time", "min"),
        End_Time=("Time", "max"),
        Start_Longitude=("Longitude", "first"),
        Start_Latitude=("Latitude", "first"),
        End_Longitude=("Longitude", "last"),
        End_Latitude=("Latitude", "last"),
        Mean_SOG=("SOG", "mean"),
        Max_SOG=("SOG", "max"),
        Max_Calculated_Speed=("Calculated_Speed_Knots", "max"),
        Total_Distance_NM=("Distance_NM", "sum")
    )
    .reset_index()
)

track_summary["Duration_Minutes"] = (
    track_summary["End_Time"] - track_summary["Start_Time"]
).dt.total_seconds() / 60

print("切分前后统计结果：")
print(f"原始船舶数量：{data['MMSI'].nunique()}")
print(f"切分得到的全部轨迹片段数量：{len(track_summary)}")

valid_track_ids = track_summary[
    track_summary["Point_Count"] >= MIN_POINTS_PER_TRACK
]["Track_ID"]

valid_data = data[
    data["Track_ID"].isin(valid_track_ids)
].copy()

valid_summary = track_summary[
    track_summary["Track_ID"].isin(valid_track_ids)
].copy()

append_audit_step("after_track_splitting_before_min_points", data)
append_audit_step(
    f"after_drop_track_point_count_lt_{MIN_POINTS_PER_TRACK}",
    valid_data
)

print(f"至少包含 {MIN_POINTS_PER_TRACK} 个点的有效轨迹数量：{len(valid_summary)}")
print(f"有效轨迹数据点总数：{len(valid_data)}")
print("=" * 70)

preprocessing_audit = pd.DataFrame(preprocessing_audit_rows)
preprocessing_audit.to_csv(
    PREPROCESS_AUDIT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 8. 保存连续轨迹数据与轨迹统计表
# ============================================================
valid_data.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

valid_summary = valid_summary.sort_values(
    ["Point_Count", "Total_Distance_NM"],
    ascending=False
)

valid_summary.to_csv(
    SUMMARY_FILE,
    index=False,
    encoding="utf-8-sig"
)

print(f"连续轨迹数据已保存到：{OUTPUT_FILE}")
print(f"轨迹统计表已保存到：{SUMMARY_FILE}")
print(f"预处理审计表已保存到：{PREPROCESS_AUDIT_FILE}")
print("=" * 70)

print("数据点最多的前 10 个连续轨迹片段：")
print(
    valid_summary[
        [
            "Track_ID",
            "VesselName",
            "Point_Count",
            "Duration_Minutes",
            "Total_Distance_NM",
            "Mean_SOG"
        ]
    ].head(10).to_string(index=False)
)
print("=" * 70)


# ============================================================
# 9. 绘制前 5 个有效连续轨迹片段
# ============================================================
top_track_ids = valid_summary.head(5)["Track_ID"]

plot_data = valid_data[
    valid_data["Track_ID"].isin(top_track_ids)
].copy()

plt.figure(figsize=(10, 8))

for track_id, track in plot_data.groupby("Track_ID"):
    track = track.sort_values("Time")

    vessel_name_series = track["VesselName"].dropna()

    if len(vessel_name_series) > 0:
        vessel_name = str(vessel_name_series.iloc[0]).strip()
        label = f"{vessel_name} | {track_id}"
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
plt.title("Examples of Continuous AIS Track Segments in Guam")
plt.legend(fontsize=7)
plt.grid(True)
plt.tight_layout()

plt.savefig(OUTPUT_FIGURE, dpi=300)
plt.close()

print(f"连续轨迹示例图已保存到：{OUTPUT_FIGURE}")
print("第二阶段完成：连续航迹已经切分并保存。")
