from pathlib import Path
import re

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. 设置项目路径
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = PROJECT_ROOT / "data" / "raw" / "ais_sample.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"

PROCESSED_FILE = PROCESSED_DIR / "ais_clean_sample.csv"
OUTPUT_FIGURE = FIGURE_DIR / "first_tracks.png"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. 检查原始数据文件是否存在
# ============================================================
if not RAW_FILE.exists():
    raise FileNotFoundError(
        f"没有找到数据文件：{RAW_FILE}\n"
        "请确认 ais_sample.csv 已经放在 data/raw 文件夹中。"
    )


# ============================================================
# 3. 读取 AIS 数据
# ============================================================
# 读取所有AIS数据
df = pd.read_csv(RAW_FILE, low_memory=False)

print("=" * 70)
print("AIS 数据读取成功")
print(f"读取的数据行数：{len(df)}")
print(f"读取到的字段：{list(df.columns)}")
print("=" * 70)


# ============================================================
# 4. 检查输入数据的必需字段
# ============================================================
required_columns = ["mmsi", "base_date_time", "sog", "cog", "geometry"]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"当前文件缺少必要字段：{missing_columns}\n"
        f"实际字段为：{list(df.columns)}"
    )


# ============================================================
# 5. 提取我们当前研究需要的数据
# ============================================================
data = df[
    [
        "mmsi",
        "base_date_time",
        "sog",
        "cog",
        "heading",
        "vessel_name",
        "vessel_type",
        "geometry"
    ]
].copy()

# 修改成后续论文处理中更容易理解的统一名称
data = data.rename(
    columns={
        "mmsi": "MMSI",
        "base_date_time": "Time",
        "sog": "SOG",
        "cog": "COG",
        "heading": "Heading",
        "vessel_name": "VesselName",
        "vessel_type": "VesselType",
        "geometry": "Geometry"
    }
)


# ============================================================
# 6. 转换时间、航速、航向字段
# ============================================================
data["Time"] = pd.to_datetime(data["Time"], errors="coerce")

data["MMSI"] = pd.to_numeric(data["MMSI"], errors="coerce")
data["SOG"] = pd.to_numeric(data["SOG"], errors="coerce")
data["COG"] = pd.to_numeric(data["COG"], errors="coerce")


# ============================================================
# 7. 从 geometry 中提取经纬度
# ============================================================
# geometry 的格式通常类似：
# POINT (144.123456 13.456789)
# POINT 中第一个数字是经度 Longitude，第二个数字是纬度 Latitude

coordinate_pattern = (
    r"POINT(?:\s+Z)?\s*\(\s*"
    r"([-+]?\d+(?:\.\d+)?)\s+"
    r"([-+]?\d+(?:\.\d+)?)"
)

coordinates = data["Geometry"].astype(str).str.extract(
    coordinate_pattern,
    flags=re.IGNORECASE
)

data["Longitude"] = pd.to_numeric(coordinates[0], errors="coerce")
data["Latitude"] = pd.to_numeric(coordinates[1], errors="coerce")

print("geometry 中的前三个位置值：")
print(data[["Geometry", "Longitude", "Latitude"]].head(3))
print("=" * 70)


# ============================================================
# 8. 基础数据清洗
# ============================================================
before_cleaning = len(data)

# 删除缺少船舶编号、时间或位置的数据
data = data.dropna(
    subset=["MMSI", "Time", "Longitude", "Latitude"]
)

# 删除超出合理地理范围的数据
data = data[
    data["Longitude"].between(-180, 180)
    & data["Latitude"].between(-90, 90)
]

# 按照船舶编号和时间排序，形成连续轨迹
data = data.sort_values(["MMSI", "Time"])

after_cleaning = len(data)

print("基础清洗完成")
print(f"清洗前数据行数：{before_cleaning}")
print(f"清洗后数据行数：{after_cleaning}")
print(f"删除的数据行数：{before_cleaning - after_cleaning}")
print(f"不同船舶数量：{data['MMSI'].nunique()}")
print(f"数据时间范围：{data['Time'].min()} 到 {data['Time'].max()}")
print(
    f"经度范围：{data['Longitude'].min():.6f} "
    f"到 {data['Longitude'].max():.6f}"
)
print(
    f"纬度范围：{data['Latitude'].min():.6f} "
    f"到 {data['Latitude'].max():.6f}"
)
print("=" * 70)


# ============================================================
# 9. 保存清洗后的样本数据
# ============================================================
data.to_csv(PROCESSED_FILE, index=False, encoding="utf-8-sig")

print(f"清洗后的样本数据已保存到：{PROCESSED_FILE}")
print("=" * 70)


# ============================================================
# 10. 选择正在移动的船舶绘制轨迹
# ============================================================
# SOG 单位通常为节。这里只选择航速大于 0.5 节的数据点，
# 避免画出的主要是停泊船舶位置。
moving_data = data[data["SOG"] > 0.5].copy()

print(f"正在移动的数据点数量：{len(moving_data)}")

moving_vessel_counts = (
    moving_data.groupby("MMSI")
    .size()
    .sort_values(ascending=False)
)

print("移动状态下记录点数量最多的前 10 艘船：")
print(moving_vessel_counts.head(10))
print("=" * 70)

# 优先选择至少具有 20 个移动记录点的船舶
candidate_vessels = moving_vessel_counts[
    moving_vessel_counts >= 20
].head(5).index

# 如果满足 20 个点的船不足，则退回到记录最多的 5 艘移动船
if len(candidate_vessels) == 0:
    candidate_vessels = moving_vessel_counts.head(5).index

if len(candidate_vessels) == 0:
    raise ValueError(
        "没有找到可以绘图的移动船舶数据。\n"
        "请检查速度字段和移动轨迹的筛选计数。"
    )

plot_data = moving_data[
    moving_data["MMSI"].isin(candidate_vessels)
].copy()


# ============================================================
# 11. 绘制真实船舶轨迹
# ============================================================
plt.figure(figsize=(10, 8))

for mmsi, track in plot_data.groupby("MMSI"):
    track = track.sort_values("Time")

    vessel_name_series = track["VesselName"].dropna()
    if len(vessel_name_series) > 0:
        vessel_name = str(vessel_name_series.iloc[0]).strip()
        label = f"{vessel_name} ({int(mmsi)})"
    else:
        label = f"MMSI {int(mmsi)}"

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
plt.title("AIS Trajectories of Moving Vessels in Guam")
plt.legend(fontsize=8)
plt.grid(True)
plt.tight_layout()

plt.savefig(OUTPUT_FIGURE, dpi=300)
plt.close()

print(f"轨迹图已保存到：{OUTPUT_FIGURE}")
print("第一阶段完成：你已经成功读取、清洗并绘制了真实 AIS 船舶轨迹。")