from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 1. 设置文件路径
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_TRACKS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "preliminary_review_pass_nominal_tracks.csv"
)

INPUT_SUMMARY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "preliminary_review_pass_track_summary.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "splits"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"

TRAIN_FILE = OUTPUT_DIR / "train_nominal_tracks.csv"
VALIDATION_FILE = OUTPUT_DIR / "validation_nominal_tracks.csv"
TEST_FILE = OUTPUT_DIR / "test_nominal_tracks.csv"
MANIFEST_FILE = OUTPUT_DIR / "dataset_split_manifest.csv"

OUTPUT_FIGURE = FIGURE_DIR / "dataset_split_overview.png"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. 设置划分比例
# ============================================================
# 目标比例：
# 训练集 60%
# 验证集 20%
# 测试集 20%
#
# 注意：划分单位是船舶 MMSI，而不是单独的数据行。
RANDOM_SEED = 42
TEST_AND_VALIDATION_RATIO = 0.40
VALIDATION_RATIO_WITHIN_TEMP = 0.50


# ============================================================
# 3. 检查输入文件
# ============================================================
for file_path in [INPUT_TRACKS_FILE, INPUT_SUMMARY_FILE]:
    if not file_path.exists():
        raise FileNotFoundError(
            f"没有找到文件：{file_path}\n"
            "请先成功运行 04_audit_nominal_tracks.py。"
        )


# ============================================================
# 4. 读取复核通过的轨迹数据
# ============================================================
tracks = pd.read_csv(INPUT_TRACKS_FILE, low_memory=False)
summary = pd.read_csv(INPUT_SUMMARY_FILE, low_memory=False)

required_track_columns = ["Track_ID", "MMSI", "Time"]
required_summary_columns = ["Track_ID", "MMSI", "Point_Count"]

for column in required_track_columns:
    if column not in tracks.columns:
        raise ValueError(f"轨迹数据缺少必要字段：{column}")

for column in required_summary_columns:
    if column not in summary.columns:
        raise ValueError(f"轨迹统计表缺少必要字段：{column}")

tracks["MMSI"] = pd.to_numeric(tracks["MMSI"], errors="coerce")
summary["MMSI"] = pd.to_numeric(summary["MMSI"], errors="coerce")

tracks = tracks.dropna(subset=["Track_ID", "MMSI"])
summary = summary.dropna(subset=["Track_ID", "MMSI"])

tracks["MMSI"] = tracks["MMSI"].astype("Int64")
summary["MMSI"] = summary["MMSI"].astype("Int64")

print("=" * 78)
print("第五阶段开始：按船舶编号划分训练集、验证集和测试集")
print(f"输入轨迹片段数量：{summary['Track_ID'].nunique()}")
print(f"输入 AIS 数据点数量：{len(tracks)}")
print(f"输入独立船舶数量：{summary['MMSI'].nunique()}")
print("=" * 78)


# ============================================================
# 5. 第一次划分：训练集 与 临时集合
# ============================================================
# GroupShuffleSplit 使用 MMSI 作为分组依据：
# 同一艘船的轨迹不会同时进入训练集和临时集合。
first_splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=TEST_AND_VALIDATION_RATIO,
    random_state=RANDOM_SEED
)

train_indices, temp_indices = next(
    first_splitter.split(
        summary,
        groups=summary["MMSI"]
    )
)

train_summary = summary.iloc[train_indices].copy()
temp_summary = summary.iloc[temp_indices].copy()


# ============================================================
# 6. 第二次划分：临时集合拆成验证集和测试集
# ============================================================
second_splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=VALIDATION_RATIO_WITHIN_TEMP,
    random_state=RANDOM_SEED
)

validation_indices, test_indices = next(
    second_splitter.split(
        temp_summary,
        groups=temp_summary["MMSI"]
    )
)

validation_summary = temp_summary.iloc[validation_indices].copy()
test_summary = temp_summary.iloc[test_indices].copy()


# ============================================================
# 7. 为每条轨迹写入数据集标签
# ============================================================
train_summary["Dataset_Split"] = "train"
validation_summary["Dataset_Split"] = "validation"
test_summary["Dataset_Split"] = "test"

manifest = pd.concat(
    [train_summary, validation_summary, test_summary],
    ignore_index=True
)

train_track_ids = set(train_summary["Track_ID"])
validation_track_ids = set(validation_summary["Track_ID"])
test_track_ids = set(test_summary["Track_ID"])

train_tracks = tracks[
    tracks["Track_ID"].isin(train_track_ids)
].copy()

validation_tracks = tracks[
    tracks["Track_ID"].isin(validation_track_ids)
].copy()

test_tracks = tracks[
    tracks["Track_ID"].isin(test_track_ids)
].copy()


# ============================================================
# 8. 检查是否存在数据泄漏
# ============================================================
train_mmsi = set(train_tracks["MMSI"].dropna().astype(int))
validation_mmsi = set(validation_tracks["MMSI"].dropna().astype(int))
test_mmsi = set(test_tracks["MMSI"].dropna().astype(int))

train_validation_overlap = train_mmsi.intersection(validation_mmsi)
train_test_overlap = train_mmsi.intersection(test_mmsi)
validation_test_overlap = validation_mmsi.intersection(test_mmsi)

print("船舶 MMSI 重叠检查：")
print(f"训练集与验证集重叠船舶数量：{len(train_validation_overlap)}")
print(f"训练集与测试集重叠船舶数量：{len(train_test_overlap)}")
print(f"验证集与测试集重叠船舶数量：{len(validation_test_overlap)}")

if (
    len(train_validation_overlap) > 0
    or len(train_test_overlap) > 0
    or len(validation_test_overlap) > 0
):
    raise ValueError(
        "发现同一艘船同时出现在不同数据集合中，存在数据泄漏。"
    )

print("检查通过：三个数据集合之间不存在相同 MMSI。")
print("=" * 78)


# ============================================================
# 9. 打印各集合规模
# ============================================================
def build_split_statistics(split_name, split_tracks, split_summary):
    return {
        "Dataset_Split": split_name,
        "Vessel_Count": split_summary["MMSI"].nunique(),
        "Track_Count": split_summary["Track_ID"].nunique(),
        "Point_Count": len(split_tracks)
    }


statistics = pd.DataFrame(
    [
        build_split_statistics("train", train_tracks, train_summary),
        build_split_statistics(
            "validation",
            validation_tracks,
            validation_summary
        ),
        build_split_statistics("test", test_tracks, test_summary)
    ]
)

statistics["Point_Ratio"] = (
    statistics["Point_Count"]
    / statistics["Point_Count"].sum()
)

statistics["Track_Ratio"] = (
    statistics["Track_Count"]
    / statistics["Track_Count"].sum()
)

print("数据集划分结果：")
print(statistics.to_string(index=False))
print("=" * 78)


# ============================================================
# 10. 保存数据文件
# ============================================================
train_tracks.to_csv(
    TRAIN_FILE,
    index=False,
    encoding="utf-8-sig"
)

validation_tracks.to_csv(
    VALIDATION_FILE,
    index=False,
    encoding="utf-8-sig"
)

test_tracks.to_csv(
    TEST_FILE,
    index=False,
    encoding="utf-8-sig"
)

manifest.to_csv(
    MANIFEST_FILE,
    index=False,
    encoding="utf-8-sig"
)

statistics.to_csv(
    OUTPUT_DIR / "dataset_split_statistics.csv",
    index=False,
    encoding="utf-8-sig"
)

print(f"训练集已经保存到：{TRAIN_FILE}")
print(f"验证集已经保存到：{VALIDATION_FILE}")
print(f"测试集已经保存到：{TEST_FILE}")
print(f"轨迹划分清单已经保存到：{MANIFEST_FILE}")
print("=" * 78)


# ============================================================
# 11. 绘制数据集规模图
# ============================================================
plt.figure(figsize=(8, 6))

plt.bar(
    statistics["Dataset_Split"],
    statistics["Point_Count"]
)

plt.xlabel("Dataset Split")
plt.ylabel("Number of AIS Points")
plt.title("Point Counts in Vessel-Disjoint Dataset Splits")
plt.grid(axis="y")
plt.tight_layout()

plt.savefig(OUTPUT_FIGURE, dpi=300)
plt.close()

print(f"数据集规模图已经保存到：{OUTPUT_FIGURE}")
print("第五阶段完成：训练集、验证集和测试集已经按船舶编号划分。")