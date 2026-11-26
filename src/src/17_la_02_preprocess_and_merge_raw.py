from pathlib import Path
import zipfile
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# 1. Path settings
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

# external_validation is stored at the project root.
EXT_ROOT = ROOT / "external_validation" / "la_2025"

RAW_DIR = EXT_ROOT / "raw"
PROCESSED_DIR = EXT_ROOT / "processed"
RESULT_DIR = EXT_ROOT / "results"
FIGURE_DIR = EXT_ROOT / "figures"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CLEANED = PROCESSED_DIR / "la_2025_cleaned_ais.csv"
OUTPUT_SUMMARY = RESULT_DIR / "la_step2_preprocess_summary.csv"
OUTPUT_GLOBAL_SUMMARY = RESULT_DIR / "la_step2_global_summary.csv"
OUTPUT_SAMPLE = RESULT_DIR / "la_step2_cleaned_sample_points.csv"
OUTPUT_FIGURE = FIGURE_DIR / "la_step2_cleaned_position_distribution.png"

# LA 下载区域范围
LON_MIN = -118.35
LON_MAX = -117.95
LAT_MIN = 33.55
LAT_MAX = 33.85

CHUNKSIZE = 300000

print("=" * 100)
print("LA 2025 CROSS-REGION VALIDATION: STEP 2")
print("Preprocess and merge raw AIS files")
print("=" * 100)
print("ROOT:", ROOT)
print("EXT_ROOT:", EXT_ROOT)
print("RAW_DIR:", RAW_DIR)
print("OUTPUT_CLEANED:", OUTPUT_CLEANED)

if not RAW_DIR.exists():
    raise FileNotFoundError(f"找不到 raw 文件夹：{RAW_DIR}")


# ============================================================
# 2. Helper functions
# ============================================================

def normalize_col(c):
    return str(c).strip().lower().replace(" ", "_")


def find_col(columns, candidates):
    norm_map = {normalize_col(c): c for c in columns}
    for cand in candidates:
        key = normalize_col(cand)
        if key in norm_map:
            return norm_map[key]
    return None


def list_csv_sources(path):
    """
    普通 CSV: [(path, None)]
    ZIP 文件: [(zip_path, csv_member_name), ...]
    """
    if path.suffix.lower() == ".csv":
        return [(path, None)]

    if path.suffix.lower() == ".zip":
        sources = []
        with zipfile.ZipFile(path, "r") as z:
            for name in z.namelist():
                if name.lower().endswith(".csv"):
                    sources.append((path, name))
        return sources

    return []


def read_csv_head(path, member=None, nrows=5):
    if member is None:
        return pd.read_csv(path, nrows=nrows)
    else:
        with zipfile.ZipFile(path, "r") as z:
            with z.open(member) as f:
                return pd.read_csv(f, nrows=nrows)


def iter_csv_chunks(path, member=None, usecols=None, chunksize=300000):
    if member is None:
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
            yield chunk
    else:
        with zipfile.ZipFile(path, "r") as z:
            with z.open(member) as f:
                for chunk in pd.read_csv(f, usecols=usecols, chunksize=chunksize):
                    yield chunk


def parse_geometry_point(series):
    """
    如果存在 geometry 字段，解析：
    POINT (-118.123 33.456)
    """
    s = series.astype(str)
    extracted = s.str.extract(r"POINT\s*\(\s*([-\d\.]+)\s+([-\d\.]+)\s*\)")
    lon = pd.to_numeric(extracted[0], errors="coerce")
    lat = pd.to_numeric(extracted[1], errors="coerce")
    return lon, lat


def detect_columns(columns):
    """
    自动识别 AIS 关键字段。
    LA 数据当前已知列名：
    MMSI, BaseDateTime, LON, LAT, SOG, COG, Heading, VesselName, VesselType
    """
    return {
        "mmsi_col": find_col(columns, ["MMSI", "mmsi"]),
        "time_col": find_col(columns, ["BaseDateTime", "base_date_time", "Time", "time", "timestamp"]),
        "lon_col": find_col(columns, ["LON", "Lon", "Longitude", "longitude", "X", "x"]),
        "lat_col": find_col(columns, ["LAT", "Lat", "Latitude", "latitude", "Y", "y"]),
        "geom_col": find_col(columns, ["geometry", "Geometry", "geom", "WKT"]),
        "sog_col": find_col(columns, ["SOG", "sog"]),
        "cog_col": find_col(columns, ["COG", "cog"]),
        "heading_col": find_col(columns, ["Heading", "heading"]),
        "vessel_name_col": find_col(columns, ["VesselName", "vessel_name", "Vessel_Name"]),
        "vessel_type_col": find_col(columns, ["VesselType", "vessel_type", "Vessel_Type"]),
    }


def build_usecols(col_map):
    usecols = []
    for c in col_map.values():
        if c is not None and c not in usecols:
            usecols.append(c)
    return usecols


def standardize_chunk(chunk, col_map, source_name):
    """
    将不同原始列名统一成标准列：
    MMSI, Time, SOG, COG, Heading, VesselName, VesselType, Geometry, Longitude, Latitude, Raw_Source
    """
    out = pd.DataFrame()

    # MMSI
    if col_map["mmsi_col"] is not None:
        out["MMSI"] = chunk[col_map["mmsi_col"]]
    else:
        out["MMSI"] = np.nan

    # Time
    if col_map["time_col"] is not None:
        out["Time"] = pd.to_datetime(chunk[col_map["time_col"]], errors="coerce")
    else:
        out["Time"] = pd.NaT

    # Longitude / Latitude
    if col_map["lon_col"] is not None and col_map["lat_col"] is not None:
        out["Longitude"] = pd.to_numeric(chunk[col_map["lon_col"]], errors="coerce")
        out["Latitude"] = pd.to_numeric(chunk[col_map["lat_col"]], errors="coerce")
    elif col_map["geom_col"] is not None:
        lon, lat = parse_geometry_point(chunk[col_map["geom_col"]])
        out["Longitude"] = lon
        out["Latitude"] = lat
    else:
        out["Longitude"] = np.nan
        out["Latitude"] = np.nan

    # SOG
    if col_map["sog_col"] is not None:
        out["SOG"] = pd.to_numeric(chunk[col_map["sog_col"]], errors="coerce")
    else:
        out["SOG"] = np.nan

    # COG
    if col_map["cog_col"] is not None:
        out["COG"] = pd.to_numeric(chunk[col_map["cog_col"]], errors="coerce")
    else:
        out["COG"] = np.nan

    # Heading
    if col_map["heading_col"] is not None:
        out["Heading"] = pd.to_numeric(chunk[col_map["heading_col"]], errors="coerce")
    else:
        out["Heading"] = np.nan

    # VesselName
    if col_map["vessel_name_col"] is not None:
        out["VesselName"] = chunk[col_map["vessel_name_col"]].astype(str)
    else:
        out["VesselName"] = ""

    # VesselType
    if col_map["vessel_type_col"] is not None:
        out["VesselType"] = chunk[col_map["vessel_type_col"]]
    else:
        out["VesselType"] = np.nan

    # Geometry
    out["Geometry"] = (
        "POINT (" +
        out["Longitude"].astype(str) +
        " " +
        out["Latitude"].astype(str) +
        ")"
    )

    out["Raw_Source"] = source_name

    # 统一列顺序
    out = out[
        [
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
            "Raw_Source",
        ]
    ]

    return out


# ============================================================
# 3. Find raw files and CSV sources
# ============================================================

raw_files = []

for pattern in ["*.csv", "*.CSV", "*.zip", "*.ZIP"]:
    raw_files.extend(list(RAW_DIR.glob(pattern)))

# Windows 下去重
raw_files = sorted(set(raw_files), key=lambda p: p.name.lower())

print("\n找到的原始文件：")
for f in raw_files:
    print(" -", f.name, f"{f.stat().st_size / 1024 / 1024:.2f} MB")

if len(raw_files) == 0:
    raise FileNotFoundError(f"在 {RAW_DIR} 中没有找到 CSV 或 ZIP 文件。")


csv_sources = []
for raw_path in raw_files:
    csv_sources.extend(list_csv_sources(raw_path))

csv_sources = sorted(
    set(csv_sources),
    key=lambda x: (str(x[0]).lower(), str(x[1]).lower() if x[1] is not None else "")
)

print("\n检测到的 CSV 数据源：")
for path, member in csv_sources:
    if member is None:
        print(" -", path.name)
    else:
        print(" -", f"{path.name} :: {member}")

if len(csv_sources) == 0:
    raise ValueError("没有检测到可读取的 CSV 数据源。")


# ============================================================
# 4. Remove old output if exists
# ============================================================

if OUTPUT_CLEANED.exists():
    print("\n发现旧的输出文件，正在删除：")
    print(OUTPUT_CLEANED)
    OUTPUT_CLEANED.unlink()


# ============================================================
# 5. Process each source by chunks
# ============================================================

summary_rows = []
sample_points = []

global_raw_rows = 0
global_clean_rows = 0
global_dropped_missing = 0
global_dropped_coordinate = 0

header_written = False

for path, member in csv_sources:
    source_name = path.name if member is None else f"{path.name}::{member}"

    print("\n" + "=" * 100)
    print("开始预处理：", source_name)
    print("=" * 100)

    head = read_csv_head(path, member, nrows=5)
    columns = list(head.columns)
    col_map = detect_columns(columns)
    usecols = build_usecols(col_map)

    print("识别到的字段：")
    for k, v in col_map.items():
        print(f"  {k}: {v}")

    if col_map["mmsi_col"] is None or col_map["time_col"] is None:
        print("[WARNING] MMSI 或时间列未识别，当前文件可能无法正常处理。")

    if (col_map["lon_col"] is None or col_map["lat_col"] is None) and col_map["geom_col"] is None:
        print("[WARNING] 经纬度列和 geometry 字段都未识别，当前文件可能无法正常处理。")

    source_raw_rows = 0
    source_clean_rows = 0
    source_dropped_missing = 0
    source_dropped_coordinate = 0

    source_min_time = None
    source_max_time = None
    source_mmsi_set = set()

    for chunk_idx, chunk in enumerate(
        iter_csv_chunks(path, member, usecols=usecols, chunksize=CHUNKSIZE),
        start=1
    ):
        raw_count = len(chunk)
        source_raw_rows += raw_count
        global_raw_rows += raw_count

        standard = standardize_chunk(chunk, col_map, source_name)

        # 删除缺失关键字段
        before_missing = len(standard)
        standard = standard.dropna(subset=["MMSI", "Time", "Longitude", "Latitude"])
        after_missing = len(standard)

        dropped_missing = before_missing - after_missing
        source_dropped_missing += dropped_missing
        global_dropped_missing += dropped_missing

        # 坐标范围过滤
        before_coord = len(standard)
        standard = standard[
            (standard["Longitude"].between(LON_MIN, LON_MAX)) &
            (standard["Latitude"].between(LAT_MIN, LAT_MAX))
        ].copy()
        after_coord = len(standard)

        dropped_coord = before_coord - after_coord
        source_dropped_coordinate += dropped_coord
        global_dropped_coordinate += dropped_coord

        # MMSI 统一为整数可读格式，无法转换的删除
        standard["MMSI"] = pd.to_numeric(standard["MMSI"], errors="coerce")
        standard = standard.dropna(subset=["MMSI"])
        standard["MMSI"] = standard["MMSI"].astype("int64")

        # SOG/COG/Heading 合理性轻度处理
        # 不在这里做过强过滤，后续轨迹筛选阶段再严格处理
        standard["SOG"] = pd.to_numeric(standard["SOG"], errors="coerce")
        standard["COG"] = pd.to_numeric(standard["COG"], errors="coerce")
        standard["Heading"] = pd.to_numeric(standard["Heading"], errors="coerce")

        # 时间范围和 MMSI
        if len(standard) > 0:
            cmin = standard["Time"].min()
            cmax = standard["Time"].max()

            source_min_time = cmin if source_min_time is None else min(source_min_time, cmin)
            source_max_time = cmax if source_max_time is None else max(source_max_time, cmax)

            source_mmsi_set.update(standard["MMSI"].unique())

            # 抽样点
            temp_sample = standard[["Longitude", "Latitude"]].dropna()
            if len(temp_sample) > 0:
                take_n = min(1000, len(temp_sample))
                sample_points.append(temp_sample.sample(n=take_n, random_state=42 + chunk_idx))

        clean_count = len(standard)
        source_clean_rows += clean_count
        global_clean_rows += clean_count

        # 保存到总 cleaned CSV
        standard.to_csv(
            OUTPUT_CLEANED,
            mode="a",
            header=not header_written,
            index=False,
            encoding="utf-8-sig",
        )
        header_written = True

        print(
            f"  chunk {chunk_idx}: raw={raw_count:,}, "
            f"clean={clean_count:,}, "
            f"drop_missing={dropped_missing:,}, "
            f"drop_coord={dropped_coord:,}"
        )

    summary_rows.append({
        "Source": source_name,
        "Raw_Row_Count": source_raw_rows,
        "Clean_Row_Count": source_clean_rows,
        "Dropped_Missing_Key_Fields": source_dropped_missing,
        "Dropped_Out_Of_LA_Bounds": source_dropped_coordinate,
        "Unique_MMSI_Count": len(source_mmsi_set),
        "Start_Time": source_min_time,
        "End_Time": source_max_time,
    })


# ============================================================
# 6. Save summary
# ============================================================

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUTPUT_SUMMARY, index=False, encoding="utf-8-sig")

print("\n" + "=" * 100)
print("LA STEP 2 PREPROCESS SUMMARY")
print("=" * 100)
print(summary_df.to_string(index=False))

print("\n预处理摘要已保存到：")
print(OUTPUT_SUMMARY)


global_summary = pd.DataFrame([{
    "Global_Raw_Row_Count": global_raw_rows,
    "Global_Clean_Row_Count": global_clean_rows,
    "Global_Dropped_Missing_Key_Fields": global_dropped_missing,
    "Global_Dropped_Out_Of_LA_Bounds": global_dropped_coordinate,
    "Output_Cleaned_File": str(OUTPUT_CLEANED),
}])

global_summary.to_csv(OUTPUT_GLOBAL_SUMMARY, index=False, encoding="utf-8-sig")

print("\n" + "=" * 100)
print("LA STEP 2 GLOBAL SUMMARY")
print("=" * 100)
print(global_summary.to_string(index=False))

print("\n全局摘要已保存到：")
print(OUTPUT_GLOBAL_SUMMARY)


# ============================================================
# 7. Plot sampled cleaned positions
# ============================================================

if len(sample_points) > 0:
    sample_df = pd.concat(sample_points, ignore_index=True)

    if len(sample_df) > 20000:
        sample_df = sample_df.sample(n=20000, random_state=42)

    sample_df.to_csv(OUTPUT_SAMPLE, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(8, 7))
    plt.scatter(
        sample_df["Longitude"],
        sample_df["Latitude"],
        s=2,
        alpha=0.35,
    )
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Cleaned Sampled AIS Positions in LA External Validation Dataset")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURE, dpi=300)
    plt.close()

    print("\n清洗后抽样点已保存到：")
    print(OUTPUT_SAMPLE)

    print("\n清洗后位置分布图已保存到：")
    print(OUTPUT_FIGURE)
else:
    print("\n没有可用于绘图的清洗后点。")


print("\n" + "=" * 100)
print("STEP 2 FINISHED")
print("=" * 100)
print("Cleaned LA AIS data saved to:")
print(OUTPUT_CLEANED)
