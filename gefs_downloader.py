import os
import requests
from datetime import datetime, timedelta

def is_data_available(date: str) -> bool:
    """
    检查指定日期的 GEFS 数据目录是否存在。
    """
    test_url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/gens/prod/gefs.{date}/00/atmos/pgrb2ap5/"
    response = requests.head(test_url)
    return response.status_code == 200

def download_file_with_idx(member: str, date: str, forecast_hour: int, save_dir: str):
    """
    下载 .grb2 文件及对应的 .idx 索引文件，如果文件已存在则跳过
    """
    run_hour = "00"
    fxx = f"f{forecast_hour:03d}"
    filename = f"{member}.t{run_hour}z.pgrb2a.0p50.{fxx}"
    base_url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/gens/prod/gefs.{date}/{run_hour}/atmos/pgrb2ap5"

    grib_url = f"{base_url}/{filename}"
    idx_url = f"{grib_url}.idx"

    os.makedirs(save_dir, exist_ok=True)
    local_base = os.path.join(save_dir, f"{date}_{member}_{fxx}.grb2")
    idx_path = local_base + ".idx"

    # ✅ 如果两个文件都已存在则跳过
    if os.path.exists(local_base) and os.path.exists(idx_path):
        print(f"✅ 已存在，跳过: {date} - {fxx}")
        return

    # 下载 GRIB 文件
    print(f"📥 Downloading GRIB: {date}_{member}_{fxx}")
    r = requests.get(grib_url, stream=True)
    if r.status_code == 200:
        with open(local_base, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ Saved: {local_base}")
    else:
        print(f"❌ GRIB not found: {grib_url} (HTTP {r.status_code})")

    # 下载 IDX 文件
    print(f"📥 Downloading IDX: {date}_{member}_{fxx}.idx")
    r_idx = requests.get(idx_url)
    if r_idx.status_code == 200:
        with open(idx_path, 'w') as f:
            f.write(r_idx.text)
        print(f"✅ Saved: {idx_path}\n")
    else:
        print(f"❌ IDX not found: {idx_url} (HTTP {r_idx.status_code})")

from datetime import datetime, timedelta

def download_recent_available_gefs(max_days: int = 4,
                                   control_hours: list = [0, 6],
                                   spread_hours: list = [6],
                                   train_dir: str = "F:/data_train",
                                   spread_dir: str = "F:/data_spread"):
    """
    下载 NOAA 当前还保留的最新 GEFS 数据。
    控制场（gec00）下载多个 forecast hour（默认 0 和 6），
    扰动场（gespr）只下载指定的 hour（默认仅 6）。
    """
    today = datetime.utcnow()
    for i in range(max_days):
        date = (today - timedelta(days=i)).strftime("%Y%m%d")
        print(f"\n🔍 Checking: {date}\n")
        if is_data_available(date):
            for fh in control_hours:
                print(f"📥 Downloading CONTROL (gec00) - forecast_hour: {fh}")
                download_file_with_idx("gec00", date, fh, train_dir)
            for fh in spread_hours:
                print(f"📥 Downloading SPREAD (gespr) - forecast_hour: {fh}")
                download_file_with_idx("gespr", date, fh, spread_dir)
        else:
            print(f"⚠️  No data found for {date} (skipped)")


# ✅ 示例调用：下载最近可用 4 天的 f006 控制+spread 数据
download_recent_available_gefs(max_days=4)
