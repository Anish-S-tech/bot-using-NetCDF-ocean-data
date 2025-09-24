import os
import glob
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# ----------------------------
# CONFIG
# ----------------------------
load_dotenv()

PARQUET_DIR = os.getenv("PARQUET_DIR", "data/processed")
DB_URI = os.getenv("DB_URI")
TABLE_NAME = os.getenv("TABLE_NAME", "argo_core_measurements")

# ----------------------------
# Connect to DB
# ----------------------------
engine = create_engine(DB_URI)

def create_table():
    """Create the table if it does not exist"""
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        id SERIAL PRIMARY KEY,
        time TIMESTAMP,
        platform_number BIGINT,
        latitude DOUBLE PRECISION,
        longitude DOUBLE PRECISION,
        depth DOUBLE PRECISION,
        temp_c DOUBLE PRECISION,
        salinity DOUBLE PRECISION
    );
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))
    print(f"Ensured table {TABLE_NAME} exists.")

def load_parquet_files():
    """Load all parquet files into Postgres"""
    parquet_files = glob.glob(os.path.join(PARQUET_DIR, "*.parquet"))
    print(f"Found {len(parquet_files)} parquet files.")

    for file in sorted(parquet_files):
        print(f" Loading {os.path.basename(file)} ...")
        try:
            df = pd.read_parquet(file)

            # Standardize column names
            df = df.rename(columns={
                "time": "time",
                "platform_number": "platform_number",
                "latitude (degrees_north)": "latitude",
                "longitude (degrees_east)": "longitude",
                "pres_adjusted (decibar)": "depth",
                "temp_adjusted (degree_Celsius)": "temp_c",
                "psal_adjusted (PSU)": "salinity"
            })

            # Drop unnecessary columns
            for col in ["id", "index"]:
                if col in df.columns:
                    df = df.drop(columns=[col])

            # Convert time if it's numeric (ms timestamps)
            if pd.api.types.is_integer_dtype(df["time"]):
                df["time"] = pd.to_datetime(df["time"], unit="ms", errors="coerce")

            # Insert into Postgres
            df.to_sql(TABLE_NAME, engine, if_exists="append", index=False)
            print(f"Loaded {len(df)} rows from {os.path.basename(file)}")

        except Exception as e:
            print(f"Error loading {file}: {e}")

if __name__ == "__main__":
    create_table()
    load_parquet_files()
    print("All files loaded successfully.")
