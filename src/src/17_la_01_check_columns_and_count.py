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
RESULT_DIR = EXT_ROOT / "results"
FIGURE_DIR = EXT_ROOT / "figures"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 100)
print("LA 2025 CROSS-REGION VALIDATION: STEP 1")
print("=" * 100)
print("ROOT:", ROOT)
print("EXT_ROOT:", EXT_ROOT)
print("RAW_DIR:", RAW_DIR)

if not RAW_DIR.exists():
    raise FileNotFoundError(
        f"找不到 raw 文件夹：{RAW_DIR}\n"
        "请将 LA AIS CSV 或 ZIP 文件放在 external_validation/la_2025/raw。"
    )


# ============================================================
# 2. Find raw files
# ============================================================

raw_files = []
for pattern in ["*.csv", "*.CSV", "*.zip", "*.ZIP"]:
    raw_files.extend(list(RAW_DIR.glob(pattern)))

# Windows 下 glob 可能会把 .csv 和 .CSV 重复匹配同一个文件，因此这里必须去重
raw_files = sorted(set(raw_files), key=lambda p: p.name.lower())

print("\n找到的原始文件：")
for f in raw_files:
    print(" -", f.name, f"{f.stat().st_size / 1024 / 1024:.2f} MB")

if len(raw_files) == 0:
    raise FileNotFoundError(
        f"在 {RAW_DIR} 中没有找到 CSV 或 ZIP 文件。\n"
        f"请把 LA AIS 数据文件放到这个文件夹里。"
    )


# ============================================================
# 3. Helper functions
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
    如果是 CSV：返回 [(path, None)]
    如果是 ZIP：返回 [(zip_path, csv_member_name), ...]
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
    """
    分块读取 CSV。
    支持普通 CSV 和 ZIP 内部 CSV。
    """
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
    解析 geometry 字段，例如：
    POINT (-118.123 33.456)
    """
    s = series.astype(str)
    extracted = s.str.extract(r"POINT\s*\(\s*([-\d\.]+)\s+([-\d\.]+)\s*\)")
    lon = pd.to_numeric(extracted[0], errors="coerce")
    lat = pd.to_numeric(extracted[1], errors="coerce")
    return lon, lat


def safe_min(old, new):
    if pd.isna(old):
        return new
    if pd.isna(new):
        return old
    return min(old, new)


def safe_max(old, new):
    if pd.isna(old):
        return new
    if pd.isna(new):
        return old
    return max(old, new)


# ============================================================
# 4. Detect CSV sources
# ============================================================

csv_sources = []

for raw_path in raw_files:
    sources = list_csv_sources(raw_path)
    csv_sources.extend(sources)

# 对 CSV source 再去重一次，避免 ZIP/CSV 或大小写重复导致重复统计
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
    raise ValueError("没有在 CSV 或 ZIP 中检测到可读取的 CSV 文件。")


# ============================================================
# 5. Inspect columns
# ============================================================

column_reports = []

for path, member in csv_sources:
    source_name = path.name if member is None else f"{path.name}::{member}"

    try:
        head = read_csv_head(path, member, nrows=5)
        columns = list(head.columns)

        print("\n" + "-" * 100)
        print("SOURCE:", source_name)
        print("COLUMNS:")
        print(columns)

        column_reports.append({
            "Source": source_name,
            "Column_Count": len(columns),
            "Columns": " | ".join(columns),
        })

    except Exception as e:
        print("[ERROR] 读取文件头失败：", source_name)
        print(e)

column_report_df = pd.DataFrame(column_reports)
column_report_out = RESULT_DIR / "la_step1_column_report.csv"
column_report_df.to_csv(column_report_out, index=False, encoding="utf-8-sig")

print("\n列名检查结果已保存到：")
print(column_report_out)


# ============================================================
# 6. Count rows and summarize key fields
# ============================================================

summary_rows = []
sample_points = []

global_total_rows = 0
global_min_time = None
global_max_time = None
global_min_lon = np.nan
global_max_lon = np.nan
global_min_lat = np.nan
global_max_lat = np.nan

for path, member in csv_sources:
    source_name = path.name if member is None else f"{path.name}::{member}"

    print("\n" + "=" * 100)
    print("开始处理：", source_name)
    print("=" * 100)

    try:
        head = read_csv_head(path, member, nrows=5)
    except Exception as e:
        print("[ERROR] 无法读取文件头：", source_name)
        print(e)
        continue

    columns = list(head.columns)

    mmsi_col = find_col(columns, ["MMSI", "mmsi"])
    time_col = find_col(columns, ["BaseDateTime", "base_date_time", "Time", "time", "timestamp"])
    lon_col = find_col(columns, ["LON", "Lon", "Longitude", "longitude", "X", "x"])
    lat_col = find_col(columns, ["LAT", "Lat", "Latitude", "latitude", "Y", "y"])
    geom_col = find_col(columns, ["geometry", "Geometry", "geom", "WKT"])

    sog_col = find_col(columns, ["SOG", "sog"])
    cog_col = find_col(columns, ["COG", "cog"])
    heading_col = find_col(columns, ["Heading", "heading"])
    vessel_name_col = find_col(columns, ["VesselName", "vessel_name", "Vessel_Name"])
    vessel_type_col = find_col(columns, ["VesselType", "vessel_type", "Vessel_Type"])

    print("识别到的关键列：")
    print("  MMSI:", mmsi_col)
    print("  Time:", time_col)
    print("  Longitude:", lon_col)
    print("  Latitude:", lat_col)
    print("  Geometry:", geom_col)
    print("  SOG:", sog_col)
    print("  COG:", cog_col)
    print("  Heading:", heading_col)
    print("  VesselName:", vessel_name_col)
    print("  VesselType:", vessel_type_col)

    if mmsi_col is None:
        print("[WARNING] 没有识别到 MMSI 列。")

    if time_col is None:
        print("[WARNING] 没有识别到时间列。")

    if (lon_col is None or lat_col is None) and geom_col is None:
        print("[WARNING] 没有识别到经纬度列，也没有识别到 geometry 列。")

    usecols = []
    for c in [
        mmsi_col,
        time_col,
        lon_col,
        lat_col,
        geom_col,
        sog_col,
        cog_col,
        heading_col,
        vessel_name_col,
        vessel_type_col,
    ]:
        if c is not None and c not in usecols:
            usecols.append(c)

    source_rows = 0
    source_min_time = None
    source_max_time = None
    source_min_lon = np.nan
    source_max_lon = np.nan
    source_min_lat = np.nan
    source_max_lat = np.nan
    source_unique_mmsi = set()
    source_sample = []

    try:
        for chunk_idx, chunk in enumerate(
            iter_csv_chunks(path, member, usecols=usecols, chunksize=300000),
            start=1
        ):
            source_rows += len(chunk)
            global_total_rows += len(chunk)

            # MMSI
            if mmsi_col is not None and mmsi_col in chunk.columns:
                source_unique_mmsi.update(
                    chunk[mmsi_col].dropna().astype(str).unique()
                )

            # Time
            if time_col is not None and time_col in chunk.columns:
                time_values = pd.to_datetime(chunk[time_col], errors="coerce")
                cmin = time_values.min()
                cmax = time_values.max()

                if pd.notna(cmin):
                    source_min_time = cmin if source_min_time is None else min(source_min_time, cmin)
                    global_min_time = cmin if global_min_time is None else min(global_min_time, cmin)

                if pd.notna(cmax):
                    source_max_time = cmax if source_max_time is None else max(source_max_time, cmax)
                    global_max_time = cmax if global_max_time is None else max(global_max_time, cmax)

            # Longitude and Latitude
            if lon_col is not None and lat_col is not None:
                lon = pd.to_numeric(chunk[lon_col], errors="coerce")
                lat = pd.to_numeric(chunk[lat_col], errors="coerce")
            elif geom_col is not None:
                lon, lat = parse_geometry_point(chunk[geom_col])
            else:
                lon = pd.Series(dtype=float)
                lat = pd.Series(dtype=float)

            if len(lon) > 0:
                cmin_lon = lon.min()
                cmax_lon = lon.max()
                cmin_lat = lat.min()
                cmax_lat = lat.max()

                source_min_lon = safe_min(source_min_lon, cmin_lon)
                source_max_lon = safe_max(source_max_lon, cmax_lon)
                source_min_lat = safe_min(source_min_lat, cmin_lat)
                source_max_lat = safe_max(source_max_lat, cmax_lat)

                global_min_lon = safe_min(global_min_lon, cmin_lon)
                global_max_lon = safe_max(global_max_lon, cmax_lon)
                global_min_lat = safe_min(global_min_lat, cmin_lat)
                global_max_lat = safe_max(global_max_lat, cmax_lat)

                temp = pd.DataFrame({
                    "Longitude": lon,
                    "Latitude": lat,
                }).dropna()

                temp = temp[
                    (temp["Longitude"].between(-180, 180)) &
                    (temp["Latitude"].between(-90, 90))
                ]

                if len(temp) > 0:
                    take_n = min(1000, len(temp))
                    source_sample.append(
                        temp.sample(n=take_n, random_state=42 + chunk_idx)
                    )

            print(f"  chunk {chunk_idx}, 已处理行数：{source_rows:,}")

    except Exception as e:
        print("[ERROR] 分块读取失败：", source_name)
        print(e)
        continue

    if len(source_sample) > 0:
        source_sample_df = pd.concat(source_sample, ignore_index=True)
        source_sample_df["Source"] = source_name
        sample_points.append(source_sample_df)

    summary_rows.append({
        "Source": source_name,
        "Record_Count": source_rows,
        "Unique_MMSI_Count": len(source_unique_mmsi),
        "Start_Time": source_min_time,
        "End_Time": source_max_time,
        "Longitude_Min": source_min_lon,
        "Longitude_Max": source_max_lon,
        "Latitude_Min": source_min_lat,
        "Latitude_Max": source_max_lat,
        "MMSI_Column": mmsi_col,
        "Time_Column": time_col,
        "Longitude_Column": lon_col,
        "Latitude_Column": lat_col,
        "Geometry_Column": geom_col,
        "SOG_Column": sog_col,
        "COG_Column": cog_col,
        "Heading_Column": heading_col,
        "VesselName_Column": vessel_name_col,
        "VesselType_Column": vessel_type_col,
    })


# ============================================================
# 7. Save summaries
# ============================================================

summary_df = pd.DataFrame(summary_rows)
summary_out = RESULT_DIR / "la_step1_raw_data_summary.csv"
summary_df.to_csv(summary_out, index=False, encoding="utf-8-sig")

print("\n" + "=" * 100)
print("LA RAW DATA SUMMARY")
print("=" * 100)
print(summary_df.to_string(index=False))

print("\n原始数据摘要已保存到：")
print(summary_out)


global_summary = pd.DataFrame([{
    "Total_Record_Count": global_total_rows,
    "Global_Start_Time": global_min_time,
    "Global_End_Time": global_max_time,
    "Global_Longitude_Min": global_min_lon,
    "Global_Longitude_Max": global_max_lon,
    "Global_Latitude_Min": global_min_lat,
    "Global_Latitude_Max": global_max_lat,
}])

global_summary_out = RESULT_DIR / "la_step1_global_summary.csv"
global_summary.to_csv(global_summary_out, index=False, encoding="utf-8-sig")

print("\n" + "=" * 100)
print("GLOBAL SUMMARY")
print("=" * 100)
print(global_summary.to_string(index=False))

print("\n全局摘要已保存到：")
print(global_summary_out)


# ============================================================
# 8. Plot sample points
# ============================================================

if len(sample_points) > 0:
    sample_df = pd.concat(sample_points, ignore_index=True)

    if len(sample_df) > 20000:
        sample_df = sample_df.sample(n=20000, random_state=42)

    sample_out = RESULT_DIR / "la_step1_sample_points.csv"
    sample_df.to_csv(sample_out, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(8, 7))
    plt.scatter(
        sample_df["Longitude"],
        sample_df["Latitude"],
        s=2,
        alpha=0.35,
    )
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Sampled AIS Positions in LA External Validation Dataset")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fig_out = FIGURE_DIR / "la_step1_sample_position_distribution.png"
    plt.savefig(fig_out, dpi=300)
    plt.close()

    print("\n抽样点数据已保存到：")
    print(sample_out)

    print("\n抽样位置分布图已保存到：")
    print(fig_out)

else:
    print("\n没有可用于绘图的有效经纬度点。")


print("\n" + "=" * 100)
print("STEP 1 FINISHED")
print("=" * 100)
