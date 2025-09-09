"""
ingest_argo.py
---------------
Fetch Argo data from ERDDAP, process, and save as Parquet for downstream use.
Executable in VS Code (Python 3.9+).
"""

import pandas as pd
from erddapy import ERDDAP
import os


def fetch_argo_data(
    server: str = "https://erddap.ifremer.fr/erddap",
    dataset_id: str = "ArgoFloats",
    lat_range=(5, 25),
    lon_range=(65, 90),
    time_range=("2020-01-01T00:00:00Z", "2025-12-31T23:59:59Z"),
    variables=None,
) -> pd.DataFrame:
    """Fetch subset of Argo data from ERDDAP into a pandas DataFrame."""

    if variables is None:
        variables = [
            "time",
            "latitude",
            "longitude",
            "pres_adjusted",
            "temp_adjusted",
            "psal_adjusted",
            "doxy",
            "chla",
            "nitrate",
            "turbidity",
        ]

    e = ERDDAP(server=server, protocol="tabledap")
    e.dataset_id = dataset_id

    e.constraints = {
        "time>=": time_range[0],
        "time<=": time_range[1],
        "latitude>=": lat_range[0],
        "latitude<=": lat_range[1],
        "longitude>=": lon_range[0],
        "longitude<=": lon_range[1],
    }

    e.variables = variables

    print("Downloading data from ERDDAP...")
    df = e.to_pandas(
        index_col="time (UTC)",
        parse_dates=True,
    )

    print(f"Downloaded {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_and_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names, handle missing values, enforce data types."""

    # Normalize column names first
    df = df.rename(
        columns=lambda x: (
            x.lower()
            .replace(" (degrees_east)", "")
            .replace(" (degree_east)", "")
            .replace(" (degrees_north)", "")
            .replace(" (degree_north)", "")
            .replace(" (UTC)", "")
            .replace(" ", "_")
        )
    )

    # Debug: print available columns
    print("Normalized columns:", df.columns.tolist())

    # Drop rows with no lat/lon
    if "latitude" in df.columns and "longitude" in df.columns:
        df = df.dropna(subset=["latitude", "longitude"])
    else:
        print("Warning: latitude/longitude not found in dataframe columns!")

    # Convert depth to float if exists
    if "pres_adjusted" in df.columns:
        df["pres_adjusted"] = pd.to_numeric(df["pres_adjusted"], errors="coerce")

    # Force numeric for key vars
    for col in ["temp_adjusted", "psal_adjusted", "doxy", "chla", "nitrate", "turbidity"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print(" Cleaned and normalized data.")
    return df


def save_to_parquet(df: pd.DataFrame, out_dir="data/processed") -> str:
    """Save dataframe into monthly Parquet files for downstream use."""

    os.makedirs(out_dir, exist_ok=True)
    df["month"] = df.index.to_period("M")

    for month, subdf in df.groupby("month"):
        filename = os.path.join(out_dir, f"argo_{month}.parquet")
        subdf.drop(columns="month").to_parquet(filename, index=True)
        print(f"Saved {filename} ({subdf.shape[0]} rows)")

    return out_dir


if __name__ == "__main__":
    df = fetch_argo_data()
    df = clean_and_normalize(df)
    save_to_parquet(df)
