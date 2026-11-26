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

INPUT_SUMMARY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "candidate_track_summary.csv"
)

INPUT_TRACK_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "candidate_nominal_tracks.csv"
)

OUTPUT_AUDIT_TABLE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "candidate_track_audit_table.csv"
)

OUTPUT_REVIEW_PASS_SUMMARY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "preliminary_review_pass_track_summary.csv"
)

OUTPUT_REVIEW_PASS_TRACKS = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "preliminary_review_pass_nominal_tracks.csv"
)

FIGURE_DIR = PROJECT_ROOT / "results" / "figures"

OUTPUT_SPEED_FIGURE = FIGURE_DIR / "audit_speed_quality_scatter.png"
OUTPUT_REPRESENTATIVE_FIGURE = FIGURE_DIR / "representative_nominal_tracks.png"
PROTOCOL_FILE = PROJECT_ROOT / "config" / "experiment_protocol.json"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

with open(PROTOCOL_FILE, "r", encoding="utf-8") as file:
    PROTOCOL = json.load(file)

PREPROCESSING_RULES = PROTOCOL["preprocessing"]


# ============================================================
# 2. 初步复核阈值
# ============================================================
# 这些阈值只用于发现需要进一步查看的轨迹，
# 不是船舶行业标准，也不是最终论文中的固定结论。
REVIEW_P95_SPEED_LIMIT_KNOTS = PREPROCESSING_RULES[
    "candidate_max_p95_calculated_speed_knots"
]
REVIEW_MEAN_SPEED_RESIDUAL_LIMIT_KNOTS = 8.0
REVIEW_MEDIAN_TIME_GAP_LIMIT_MINUTES = 10.0


# ============================================================
# 3. 检查文件是否存在
# ============================================================
for file_path in [INPUT_SUMMARY_FILE, INPUT_TRACK_FILE]:
    if not file_path.exists():
        raise FileNotFoundError(
            f"没有找到文件：{file_path}\n"
            "请先成功运行 03_select_tracks_and_extract_features.py。"
        )


# ============================================================
# 4. 读取候选轨迹统计表和轨迹数据
# ============================================================
summary = pd.read_csv(INPUT_SUMMARY_FILE, low_memory=False)
tracks = pd.read_csv(INPUT_TRACK_FILE, low_memory=False)

tracks["Time"] = pd.to_datetime(tracks["Time"], errors="coerce")

numeric_columns = [
    "Point_Count",
    "Duration_Minutes",
    "Total_Distance_NM",
    "Moving_Point_Ratio",
    "Median_Time_Gap_Minutes",
    "P95_Calculated_Speed_Knots",
    "Mean_Speed_Residual_Knots",
    "Mean_Course_Mismatch_Deg"
]

for column in numeric_columns:
    if column in summary.columns:
        summary[column] = pd.to_numeric(summary[column], errors="coerce")

print("=" * 78)
print("第四阶段开始：候选基线轨迹质量复核")
print(f"输入候选轨迹数量：{len(summary)}")
print(f"输入候选数据点数量：{len(tracks)}")
print("=" * 78)


# ============================================================
# 5. 打印关键指标分布
# ============================================================
audit_metrics = [
    "Point_Count",
    "Duration_Minutes",
    "Total_Distance_NM",
    "Median_Time_Gap_Minutes",
    "P95_Calculated_Speed_Knots",
    "Mean_Speed_Residual_Knots",
    "Mean_Course_Mismatch_Deg"
]

quantile_levels = [0.00, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.00]

print("关键质量指标分位数统计：")

for metric in audit_metrics:
    print("-" * 78)
    print(metric)

    metric_quantiles = summary[metric].quantile(quantile_levels)

    for level, value in metric_quantiles.items():
        if pd.isna(value):
            print(f"  {level:>5.0%}: NaN")
        else:
            print(f"  {level:>5.0%}: {value:.4f}")

print("=" * 78)


# ============================================================
# 6. 标记可能需要人工复核的轨迹
# ============================================================
summary["Flag_High_P95_Calculated_Speed"] = (
    summary["P95_Calculated_Speed_Knots"]
    > REVIEW_P95_SPEED_LIMIT_KNOTS
)

summary["Flag_High_Mean_Speed_Residual"] = (
    summary["Mean_Speed_Residual_Knots"]
    > REVIEW_MEAN_SPEED_RESIDUAL_LIMIT_KNOTS
)

summary["Flag_Sparse_Time_Interval"] = (
    summary["Median_Time_Gap_Minutes"]
    > REVIEW_MEDIAN_TIME_GAP_LIMIT_MINUTES
)

flag_columns = [
    "Flag_High_P95_Calculated_Speed",
    "Flag_High_Mean_Speed_Residual",
    "Flag_Sparse_Time_Interval"
]

summary["Review_Flag_Count"] = summary[flag_columns].sum(axis=1)

summary["Preliminary_Review_Pass"] = (
    summary["Review_Flag_Count"] == 0
)

print("初步复核标记结果：")
print(
    f"P95 推算航速超过 {REVIEW_P95_SPEED_LIMIT_KNOTS} 节的轨迹数量："
    f"{summary['Flag_High_P95_Calculated_Speed'].sum()}"
)
print(
    f"平均速度残差超过 {REVIEW_MEAN_SPEED_RESIDUAL_LIMIT_KNOTS} 节的轨迹数量："
    f"{summary['Flag_High_Mean_Speed_Residual'].sum()}"
)
print(
    f"时间间隔中位数超过 {REVIEW_MEDIAN_TIME_GAP_LIMIT_MINUTES} 分钟的轨迹数量："
    f"{summary['Flag_Sparse_Time_Interval'].sum()}"
)
print("-" * 78)
print(
    f"没有触发上述标记的初步通过轨迹数量："
    f"{summary['Preliminary_Review_Pass'].sum()}"
)
print("=" * 78)


# ============================================================
# 7. 输出最值得检查的轨迹
# ============================================================
audit_table = summary.sort_values(
    [
        "Review_Flag_Count",
        "P95_Calculated_Speed_Knots",
        "Mean_Speed_Residual_Knots"
    ],
    ascending=[False, False, False]
)

print("需要优先检查的前 15 段轨迹：")
print(
    audit_table[
        [
            "Track_ID",
            "VesselName",
            "Point_Count",
            "Total_Distance_NM",
            "Median_Time_Gap_Minutes",
            "P95_Calculated_Speed_Knots",
            "Mean_Speed_Residual_Knots",
            "Mean_Course_Mismatch_Deg",
            "Review_Flag_Count"
        ]
    ]
    .head(15)
    .to_string(index=False)
)
print("=" * 78)


# ============================================================
# 8. 保存复核表和初步通过数据
# ============================================================
audit_table.to_csv(
    OUTPUT_AUDIT_TABLE,
    index=False,
    encoding="utf-8-sig"
)

review_pass_summary = summary[
    summary["Preliminary_Review_Pass"]
].copy()

review_pass_track_ids = review_pass_summary["Track_ID"]

review_pass_tracks = tracks[
    tracks["Track_ID"].isin(review_pass_track_ids)
].copy()

review_pass_summary.to_csv(
    OUTPUT_REVIEW_PASS_SUMMARY,
    index=False,
    encoding="utf-8-sig"
)

review_pass_tracks.to_csv(
    OUTPUT_REVIEW_PASS_TRACKS,
    index=False,
    encoding="utf-8-sig"
)

print(f"完整复核表已保存到：{OUTPUT_AUDIT_TABLE}")
print(f"初步通过轨迹统计表已保存到：{OUTPUT_REVIEW_PASS_SUMMARY}")
print(f"初步通过轨迹数据已保存到：{OUTPUT_REVIEW_PASS_TRACKS}")
print(f"初步通过的数据点数量：{len(review_pass_tracks)}")
print("=" * 78)


# ============================================================
# 9. 绘制速度一致性质量图
# ============================================================
not_flagged = summary[
    summary["Preliminary_Review_Pass"]
]

flagged = summary[
    ~summary["Preliminary_Review_Pass"]
]

plt.figure(figsize=(10, 7))

plt.scatter(
    not_flagged["P95_Calculated_Speed_Knots"],
    not_flagged["Mean_Speed_Residual_Knots"],
    alpha=0.6,
    label="Preliminary review pass"
)

plt.scatter(
    flagged["P95_Calculated_Speed_Knots"],
    flagged["Mean_Speed_Residual_Knots"],
    alpha=0.75,
    label="Flagged for review"
)

plt.axvline(
    REVIEW_P95_SPEED_LIMIT_KNOTS,
    linestyle="--",
    label="Review speed threshold"
)

plt.axhline(
    REVIEW_MEAN_SPEED_RESIDUAL_LIMIT_KNOTS,
    linestyle="--",
    label="Review residual threshold"
)

plt.xlabel("95th Percentile of Calculated Speed (Knots)")
plt.ylabel("Mean Speed Residual (Knots)")
plt.title("Preliminary Quality Review of Candidate Nominal Tracks")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_SPEED_FIGURE, dpi=300)
plt.close()


# ============================================================
# 10. 绘制具有代表性的轨迹，而不是距离最长的轨迹
# ============================================================
plot_source = review_pass_summary.copy()

if len(plot_source) < 5:
    plot_source = summary.copy()

target_quantiles = [0.10, 0.30, 0.50, 0.70, 0.90]
selected_track_ids = []

for quantile in target_quantiles:
    target_distance = plot_source["Total_Distance_NM"].quantile(quantile)

    remaining = plot_source[
        ~plot_source["Track_ID"].isin(selected_track_ids)
    ].copy()

    if len(remaining) == 0:
        break

    remaining["Distance_To_Target"] = (
        remaining["Total_Distance_NM"] - target_distance
    ).abs()

    chosen_track_id = (
        remaining.sort_values("Distance_To_Target")
        .iloc[0]["Track_ID"]
    )

    selected_track_ids.append(chosen_track_id)

representative_tracks = tracks[
    tracks["Track_ID"].isin(selected_track_ids)
].copy()

plt.figure(figsize=(10, 8))

for track_id, track in representative_tracks.groupby("Track_ID"):
    track = track.sort_values("Time")

    vessel_name_values = track["VesselName"].dropna()

    if len(vessel_name_values) > 0:
        vessel_name = str(vessel_name_values.iloc[0]).strip()
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
plt.title("Representative Candidate Nominal AIS Track Segments")
plt.legend(fontsize=7)
plt.grid(True)
plt.axis("equal")
plt.tight_layout()
plt.savefig(OUTPUT_REPRESENTATIVE_FIGURE, dpi=300)
plt.close()

print(f"速度一致性质量图已保存到：{OUTPUT_SPEED_FIGURE}")
print(f"代表性轨迹图已保存到：{OUTPUT_REPRESENTATIVE_FIGURE}")
print("第四阶段完成：候选基线轨迹已经完成初步质量复核。")
