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
# 1. 项目路径与实验设置
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = PROJECT_ROOT / "external_validation" / "la_2025" / "synthetic_v3"
OUTPUT_DIR = PROJECT_ROOT / "external_validation" / "la_2025" / "windows_v2"
FIGURE_DIR = PROJECT_ROOT / "external_validation" / "la_2025" / "figures" / "windows_v2"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
PROTOCOL_FILE = PROJECT_ROOT / "config" / "experiment_protocol.json"

with open(PROTOCOL_FILE, "r", encoding="utf-8") as file:
    PROTOCOL = json.load(file)

INPUT_FILES = {
    "test": INPUT_DIR / "test_synthetic_scenarios_v2.csv"
}

PRIMARY_SCENARIOS = (
    ["normal"]
    + PROTOCOL["generators"]["A"]["attack_types"]
    + PROTOCOL["generators"]["B"]["attack_types"]
)

WINDOW_LABEL_OVERLAP_RATIO = PROTOCOL["window_labeling"][
    "attack_overlap_ratio_threshold"
]

WINDOW_LENGTH = 10
WINDOW_STRIDE = 5
ATTACK_CONTEXT_POINTS_EACH_SIDE = 2 * WINDOW_LENGTH

# 防止某些特别长的正常轨迹产生过多重复正常窗口，
# 让少数长轨迹主导训练集。
MAX_NORMAL_WINDOWS_PER_TRACK = 50

POINT_FEATURES = [
    "Delta_Minutes",
    "Distance_NM",
    "Calculated_Speed_Knots",
    "Speed_Residual_Knots",
    "COG_Change_Deg",
    "COG_Change_Rate_Deg_Per_Min",
    "Acceleration_Knots_Per_Min",
    "Course_Position_Mismatch_Deg"
]


# ============================================================
# 2. 检查输入文件
# ============================================================
for split_name, file_path in INPUT_FILES.items():
    if not file_path.exists():
        raise FileNotFoundError(
            f"没有找到 {split_name} 数据文件：{file_path}\n"
            "请先成功运行 06_generate_synthetic_spoofing_scenarios_v2.py。"
        )


# ============================================================
# 3. 基础地理距离计算函数
# ============================================================
def haversine_distance_nm(lon1, lat1, lon2, lat2):
    """
    根据经纬度计算两个位置之间的球面距离。
    返回单位：海里。
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

    a = np.clip(a, 0, 1)

    c = 2 * np.arcsin(np.sqrt(a))

    return earth_radius_nm * c


# ============================================================
# 4. 读取并检查场景数据
# ============================================================
def load_synthetic_data(file_path, split_name):
    data = pd.read_csv(file_path, low_memory=False)

    required_columns = [
        "Scenario_ID",
        "Track_ID",
        "MMSI",
        "Time",
        "Point_Index",
        "Attack_Type",
        "Scenario_Label",
        "Attack_Label",
        "Point_Label",
        "Detection_Label",
        "Longitude",
        "Latitude"
    ] + POINT_FEATURES

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{file_path.name} 缺少必要字段：{missing_columns}\n"
            f"实际字段为：{list(data.columns)}"
        )

    data["Time"] = pd.to_datetime(data["Time"], errors="coerce")

    numeric_columns = [
        "MMSI",
        "Point_Index",
        "Scenario_Label",
        "Attack_Label",
        "Point_Label",
        "Detection_Label",
        "Longitude",
        "Latitude"
    ] + POINT_FEATURES

    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(
        subset=[
            "Scenario_ID",
            "Track_ID",
            "MMSI",
            "Time",
            "Point_Index",
            "Longitude",
            "Latitude"
        ]
    )

    before_filter = len(data)

    data = data[
        data["Attack_Type"].isin(PRIMARY_SCENARIOS)
    ].copy()

    excluded_count = before_filter - len(data)

    data["Dataset_Split"] = split_name

    data = data.sort_values(
        ["Scenario_ID", "Point_Index"]
    ).reset_index(drop=True)

    print(
        f"{split_name.upper()}：按主实验攻击类型过滤后删除的数据点数量："
        f"{excluded_count}"
    )

    return data


# ============================================================
# 5. 窗口特征提取函数
# ============================================================
def aggregate_window_features(
    window,
    split_name,
    scenario_id,
    track_id,
    attack_type,
    generator_name,
    window_start,
    window_end,
    scenario_start_time,
    scenario_end_time,
    scenario_attack_start_time,
    scenario_attack_end_time
):
    """
    将一个连续 10 点窗口转换为一个机器学习样本。
    """

    attack_point_count = int(window["Attack_Label"].sum())
    attack_overlap_ratio = attack_point_count / len(window)

    attack_start = pd.to_numeric(
        window["Attack_Start_Index"],
        errors="coerce"
    ).dropna()

    attack_end = pd.to_numeric(
        window["Attack_End_Index"],
        errors="coerce"
    ).dropna()

    row = {
        "Dataset_Split": split_name,
        "Scenario_ID": scenario_id,
        "Track_ID": track_id,
        "MMSI": int(window["MMSI"].iloc[0]),
        "Generator": generator_name,
        "Attack_Type": attack_type,
        "Window_Start_Index": int(window_start),
        "Window_End_Index": int(window_end),
        "Window_Start_Time": window["Time"].iloc[0].isoformat(),
        "Window_End_Time": window["Time"].iloc[-1].isoformat(),
        "Scenario_Start_Time": scenario_start_time.isoformat(),
        "Scenario_End_Time": scenario_end_time.isoformat(),
        "Scenario_Attack_Start_Time": (
            scenario_attack_start_time.isoformat()
            if pd.notna(scenario_attack_start_time) else ""
        ),
        "Scenario_Attack_End_Time": (
            scenario_attack_end_time.isoformat()
            if pd.notna(scenario_attack_end_time) else ""
        ),
        "Window_Point_Count": int(len(window)),
        "Window_Label": int(
            attack_overlap_ratio >= WINDOW_LABEL_OVERLAP_RATIO
        ),
        "Attack_Overlap_Ratio": float(attack_overlap_ratio),
        "Attack_Point_Count": attack_point_count,
        "Direct_Modified_Point_Count": int(
            window["Point_Label"].sum()
        ),
        "Detection_Abnormal_Point_Count": int(
            window["Detection_Label"].sum()
        ),
        "Scenario_Attack_Start_Index": (
            int(attack_start.iloc[0]) if len(attack_start) else np.nan
        ),
        "Scenario_Attack_End_Index": (
            int(attack_end.iloc[0]) if len(attack_end) else np.nan
        )
    }

    # --------------------------------------------------------
    # 窗口整体时间与路径特征
    # --------------------------------------------------------
    window_duration_minutes = (
        window["Time"].iloc[-1] - window["Time"].iloc[0]
    ).total_seconds() / 60

    row["Window_Duration_Minutes"] = window_duration_minutes

    row["Window_Path_Length_NM"] = (
        window["Distance_NM"].sum(min_count=1)
    )

    row["Window_Start_End_Displacement_NM"] = haversine_distance_nm(
        window["Longitude"].iloc[0],
        window["Latitude"].iloc[0],
        window["Longitude"].iloc[-1],
        window["Latitude"].iloc[-1]
    )

    if (
        pd.notna(row["Window_Path_Length_NM"])
        and row["Window_Path_Length_NM"] > 0
    ):
        row["Window_Path_Efficiency"] = (
            row["Window_Start_End_Displacement_NM"]
            / row["Window_Path_Length_NM"]
        )
    else:
        row["Window_Path_Efficiency"] = np.nan

    # --------------------------------------------------------
    # 点级运动学特征的窗口统计量
    # --------------------------------------------------------
    for feature in POINT_FEATURES:
        values = pd.to_numeric(
            window[feature],
            errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)

        row[f"{feature}_Mean"] = values.mean()
        row[f"{feature}_Max"] = values.max()
        row[f"{feature}_Std"] = values.std(ddof=0)
        row[f"{feature}_Median"] = values.median()
        row[f"{feature}_P95"] = values.quantile(0.95)

    return row


# ============================================================
# 6. 为单个集合建立滑动窗口数据
# ============================================================
def build_windows_for_split(data, split_name):
    window_rows = []

    scenario_count = 0
    normal_window_count = 0
    abnormal_window_count = 0

    for scenario_id, scenario in data.groupby("Scenario_ID"):
        scenario = scenario.sort_values(
            "Point_Index"
        ).reset_index(drop=True)

        scenario_start_time = scenario["Time"].iloc[0]
        scenario_end_time = scenario["Time"].iloc[-1]
        attack_start_values = pd.to_numeric(
            scenario["Attack_Start_Index"], errors="coerce"
        ).dropna()
        attack_end_values = pd.to_numeric(
            scenario["Attack_End_Index"], errors="coerce"
        ).dropna()
        if len(attack_start_values) and len(attack_end_values):
            scenario_attack_start_time = scenario["Time"].iloc[
                int(attack_start_values.iloc[0])
            ]
            scenario_attack_end_time = scenario["Time"].iloc[
                int(attack_end_values.iloc[0])
            ]
        else:
            scenario_attack_start_time = pd.NaT
            scenario_attack_end_time = pd.NaT

        if len(scenario) < WINDOW_LENGTH:
            continue

        scenario_count += 1

        attack_type = str(scenario["Attack_Type"].iloc[0])
        generator_name = str(
            scenario["Generator"].iloc[0]
            if "Generator" in scenario.columns
            else "A"
        )
        track_id = str(scenario["Track_ID"].iloc[0])

        start_positions = list(
            range(
                0,
                len(scenario) - WINDOW_LENGTH + 1,
                WINDOW_STRIDE
            )
        )

        # ----------------------------------------------------
        # 正常轨迹：提取正常窗口，但限制特别长轨迹的窗口数量
        # ----------------------------------------------------
        if attack_type == "normal":
            if len(start_positions) > MAX_NORMAL_WINDOWS_PER_TRACK:
                selected_positions = np.linspace(
                    0,
                    len(start_positions) - 1,
                    MAX_NORMAL_WINDOWS_PER_TRACK,
                    dtype=int
                )

                start_positions = [
                    start_positions[position]
                    for position in np.unique(selected_positions)
                ]

            for start_index in start_positions:
                end_index = start_index + WINDOW_LENGTH - 1

                window = scenario.iloc[
                    start_index:start_index + WINDOW_LENGTH
                ].copy()

                row = aggregate_window_features(
                    window=window,
                    split_name=split_name,
                    scenario_id=scenario_id,
                    track_id=track_id,
                    attack_type=attack_type,
                    generator_name=generator_name,
                    window_start=start_index,
                    window_end=end_index,
                    scenario_start_time=scenario_start_time,
                    scenario_end_time=scenario_end_time,
                    scenario_attack_start_time=scenario_attack_start_time,
                    scenario_attack_end_time=scenario_attack_end_time
                )

                window_rows.append(row)
                normal_window_count += 1

        # ----------------------------------------------------
        # 攻击轨迹：保持攻击附近上下文窗口的时间顺序。
        # 主标签只由预先固定的 Attack_Label overlap 规则决定，
        # Detection_Label 只作为辅助分析字段保留。
        # ----------------------------------------------------
        else:
            attack_start = pd.to_numeric(
                scenario["Attack_Start_Index"],
                errors="coerce"
            ).dropna()

            attack_end = pd.to_numeric(
                scenario["Attack_End_Index"],
                errors="coerce"
            ).dropna()

            if len(attack_start) > 0 and len(attack_end) > 0:
                attack_start = int(attack_start.iloc[0])
                attack_end = int(attack_end.iloc[0])

                start_positions = [
                    start_index
                    for start_index in start_positions
                    if (
                        start_index + WINDOW_LENGTH - 1
                        >= attack_start - ATTACK_CONTEXT_POINTS_EACH_SIDE
                    )
                    and (
                        start_index
                        <= attack_end + ATTACK_CONTEXT_POINTS_EACH_SIDE
                    )
                ]

            for start_index in start_positions:
                end_index = start_index + WINDOW_LENGTH - 1

                window = scenario.iloc[
                    start_index:start_index + WINDOW_LENGTH
                ].copy()

                row = aggregate_window_features(
                    window=window,
                    split_name=split_name,
                    scenario_id=scenario_id,
                    track_id=track_id,
                    attack_type=attack_type,
                    generator_name=generator_name,
                    window_start=start_index,
                    window_end=end_index,
                    scenario_start_time=scenario_start_time,
                    scenario_end_time=scenario_end_time,
                    scenario_attack_start_time=scenario_attack_start_time,
                    scenario_attack_end_time=scenario_attack_end_time
                )

                window_rows.append(row)
                abnormal_window_count += 1

    if len(window_rows) == 0:
        raise ValueError(
            f"{split_name} 集合没有生成任何窗口样本。"
        )

    windows = pd.DataFrame(window_rows)

    print("=" * 86)
    print(f"{split_name.upper()} 窗口数据生成完成")
    print(f"参与生成窗口的场景数量：{scenario_count}")
    print(f"正常窗口数量：{normal_window_count}")
    print(f"异常窗口数量：{abnormal_window_count}")
    print(f"窗口总数：{len(windows)}")
    print(
        f"异常窗口比例："
        f"{windows['Window_Label'].mean():.4f}"
    )

    return windows


# ============================================================
# 7. 生成 train / validation / test 窗口数据
# ============================================================
all_window_data = {}
all_statistics = []

for split_name, input_file in INPUT_FILES.items():
    synthetic_data = load_synthetic_data(
        input_file,
        split_name
    )

    windows = build_windows_for_split(
        synthetic_data,
        split_name
    )

    output_file = (
        OUTPUT_DIR
        / f"{split_name}_window_samples.csv"
    )

    windows.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    all_window_data[split_name] = windows

    attack_statistics = (
        windows.groupby(
            ["Window_Label", "Attack_Type"],
            as_index=False
        )
        .size()
        .rename(columns={"size": "Window_Count"})
    )

    attack_statistics["Dataset_Split"] = split_name

    all_statistics.append(attack_statistics)

    print(f"窗口数据已保存到：{output_file}")


# ============================================================
# 8. 保存窗口统计表
# ============================================================
statistics = pd.concat(
    all_statistics,
    ignore_index=True
)

statistics = statistics[
    [
        "Dataset_Split",
        "Window_Label",
        "Attack_Type",
        "Window_Count"
    ]
]

statistics_file = (
    OUTPUT_DIR
    / "window_dataset_statistics.csv"
)

statistics.to_csv(
    statistics_file,
    index=False,
    encoding="utf-8-sig"
)

print("=" * 86)
print("三个集合的窗口样本统计：")
print(statistics.to_string(index=False))
print("=" * 86)
print(f"窗口统计表已保存到：{statistics_file}")


# ============================================================
# 9. 检查训练、验证、测试之间是否仍存在 MMSI 泄漏
# ============================================================
mmsi_sets = {
    split_name: set(
        windows["MMSI"].astype(int).unique()
    )
    for split_name, windows in all_window_data.items()
}

if {"train", "validation", "test"}.issubset(mmsi_sets):
    overlap_train_validation = mmsi_sets["train"].intersection(mmsi_sets["validation"])
    overlap_train_test = mmsi_sets["train"].intersection(mmsi_sets["test"])
    overlap_validation_test = mmsi_sets["validation"].intersection(mmsi_sets["test"])
    if overlap_train_validation or overlap_train_test or overlap_validation_test:
        raise ValueError("窗口数据中出现 MMSI 泄漏，请停止后续建模。")
    print("检查通过：窗口数据仍然不存在跨集合 MMSI 泄漏。")
else:
    print("外部地域仅含 test 集；跨集合 MMSI 检查不适用。")
print("=" * 86)


# ============================================================
# 10. 生成模型可使用的特征列清单
# ============================================================
non_feature_columns = {
    "Dataset_Split",
    "Scenario_ID",
    "Track_ID",
        "MMSI",
        "Generator",
        "Attack_Type",
        "Window_Start_Index",
        "Window_End_Index",
        "Window_Start_Time",
        "Window_End_Time",
        "Scenario_Start_Time",
        "Scenario_End_Time",
        "Scenario_Attack_Start_Time",
        "Scenario_Attack_End_Time",
        "Window_Point_Count",
        "Window_Label",
        "Attack_Overlap_Ratio",
        "Attack_Point_Count",
        "Direct_Modified_Point_Count",
        "Detection_Abnormal_Point_Count",
        "Scenario_Attack_Start_Index",
        "Scenario_Attack_End_Index"
}

feature_columns = [
    column
    for column in next(iter(all_window_data.values())).columns
    if column not in non_feature_columns
]

feature_list_file = (
    OUTPUT_DIR
    / "model_feature_columns.txt"
)

with open(feature_list_file, "w", encoding="utf-8") as file:
    for feature in feature_columns:
        file.write(feature + "\n")

print("后续模型允许使用的特征列：")
for feature in feature_columns:
    print(f"  {feature}")

print(f"特征列清单已保存到：{feature_list_file}")
print("=" * 86)


# ============================================================
# 11. 绘制不同集合的窗口标签数量图
# ============================================================
label_statistics = []

for split_name, windows in all_window_data.items():
    label_count = (
        windows.groupby("Window_Label")
        .size()
        .reindex([0, 1], fill_value=0)
    )

    label_statistics.append(
        {
            "Dataset_Split": split_name,
            "Normal_Windows": int(label_count.loc[0]),
            "Abnormal_Windows": int(label_count.loc[1])
        }
    )

label_statistics = pd.DataFrame(label_statistics)

x_positions = np.arange(len(label_statistics))
bar_width = 0.38

plt.figure(figsize=(9, 6))

plt.bar(
    x_positions - bar_width / 2,
    label_statistics["Normal_Windows"],
    width=bar_width,
    label="Normal windows"
)

plt.bar(
    x_positions + bar_width / 2,
    label_statistics["Abnormal_Windows"],
    width=bar_width,
    label="Abnormal windows"
)

plt.xticks(
    x_positions,
    label_statistics["Dataset_Split"]
)

plt.xlabel("Dataset Split")
plt.ylabel("Number of Sliding Windows")
plt.title("Sliding-Window Sample Distribution")
plt.legend()
plt.grid(axis="y")
plt.tight_layout()

label_figure_file = (
    FIGURE_DIR
    / "window_label_distribution.png"
)

plt.savefig(
    label_figure_file,
    dpi=300
)

plt.close()

print(f"窗口标签数量图已保存到：{label_figure_file}")


# ============================================================
# 12. 绘制三类攻击在测试集中的窗口数量图
# ============================================================
test_attack_statistics = (
    all_window_data["test"]
    .query("Window_Label == 1")
    .groupby(["Generator", "Attack_Type"])
    .size()
    .reset_index(name="Window_Count")
)

plt.figure(figsize=(9, 6))

plt.bar(
    (
        test_attack_statistics["Generator"]
        + ":"
        + test_attack_statistics["Attack_Type"]
    ),
    test_attack_statistics["Window_Count"]
)

plt.xlabel("Synthetic Position-Spoofing Scenario")
plt.ylabel("Number of Abnormal Test Windows")
plt.title("Abnormal Test Windows by Scenario Type")
plt.xticks(rotation=15)
plt.grid(axis="y")
plt.tight_layout()

attack_figure_file = (
    FIGURE_DIR
    / "test_attack_window_distribution.png"
)

plt.savefig(
    attack_figure_file,
    dpi=300
)

plt.close()

print(f"测试集攻击窗口数量图已保存到：{attack_figure_file}")
print("=" * 86)
print("LA 外部验证滑动窗口数据集已经生成。")
print(
    "主窗口标签使用 Attack_Label overlap 规则；"
    "Point_Label 和 Detection_Label 仅用于辅助分析。"
)

