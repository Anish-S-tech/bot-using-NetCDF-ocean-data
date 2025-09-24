import os, time
import pandas as pd
from erddapy import ERDDAP
from dotenv import load_dotenv

# -------------------------
# Load .env file
# -------------------------
load_dotenv()

ERDDAP_SERVER = os.getenv("ERDDAP_SERVER", "https://erddap.ifremer.fr/erddap")
DATASET_ID = os.getenv("DATASET_ID", "ArgoFloats")
OUT_DIR = os.getenv("PARQUET_DIR", "data/processed")
START_YEAR = int(os.getenv("START_YEAR", 2020))
END_YEAR = int(os.getenv("END_YEAR", 2025))

# Geographic constraints (region around India)
LAT_MIN = float(os.getenv("LAT_MIN", 5))
LAT_MAX = float(os.getenv("LAT_MAX", 25))
LON_MIN = float(os.getenv("LON_MIN", 65))
LON_MAX = float(os.getenv("LON_MAX", 95))

os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------
# Core Argo variables
# -------------------------
VARS = [
    "time", "platform_number", "latitude", "longitude",
    "pres_adjusted", "temp_adjusted", "psal_adjusted"
]

# -------------------------
# ERDDAP connection
# -------------------------
e = ERDDAP(server=ERDDAP_SERVER, protocol="tabledap")
e.dataset_id = DATASET_ID
e.variables = VARS

# -------------------------
# Loop through years/months
# -------------------------
for year in range(START_YEAR, END_YEAR + 1):
    for month in range(1, 13):
        t0 = f"{year}-{month:02d}-01T00:00:00Z"
        if month == 12:
            t1 = f"{year+1}-01-01T00:00:00Z"
        else:
            t1 = f"{year}-{month+1:02d}-01T00:00:00Z"

        e.constraints = {
            "time>=": t0,
            "time<": t1,
            "latitude>=": LAT_MIN,
            "latitude<=": LAT_MAX,
            "longitude>=": LON_MIN,
            "longitude<=": LON_MAX
        }

        try:
            print(f"Fetching CORE {year}-{month:02d}")
            df = e.to_pandas(parse_dates=True)
            if df is None or df.empty:
                print("No rows")
                continue

            df = df.reset_index()

            df = df.rename(columns={
                "time (UTC)": "time",
                "pres_adjusted": "depth_m",
                "temp_adjusted": "temp_c",
                "psal_adjusted": "salinity"
            })

            df["time"] = pd.to_datetime(df["time"], errors="coerce")

            out_file = os.path.join(OUT_DIR, f"core_{year}_{month:02d}.parquet")
            df.to_parquet(out_file, index=False)
            print(f"Saved {out_file} ({len(df)} rows)")

        except Exception as err:
            print("Error:", err)
            time.sleep(5)
