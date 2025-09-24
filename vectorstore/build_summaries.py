# build_summaries.py
import os
import pandas as pd
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()

DATA_DIR = os.getenv("PARQUET_DIR", "data/processed")
CHROMA_PATH = os.getenv("CHROMA_PATH", "vectorstore/chroma_db")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ----------------------------
# Helpers
# ----------------------------
def normalize_columns(df):
    """Standardize column names across parquet files."""
    rename_map = {
        "pres_adjusted (decibar)": "depth",
        "pressure": "depth",
        "temp_adjusted (degree_Celsius)": "temp_c",
        "temperature": "temp_c",
        "psal_adjusted (PSU)": "salinity",
        "psal": "salinity",
        "psal_adjusted": "salinity",
        "temp_adjusted": "temp_c",
    }
    df = df.rename(columns=rename_map)

    needed = ["time", "platform_number", "latitude", "longitude", "depth", "temp_c", "salinity"]
    for col in needed:
        if col not in df.columns:
            df[col] = None

    return df[needed]


def summarize_file(file_path):
    """Generate summary documents for one parquet file."""
    try:
        df = pd.read_parquet(file_path)
        df = normalize_columns(df)

        df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
        df = df.dropna(subset=["time"])
        if df.empty:
            return []

        df["year_month"] = df["time"].dt.to_period("M")

        docs = []
        for (ym, platform), g in df.groupby(["year_month", "platform_number"]):
            if g["temp_c"].notna().any() and g["salinity"].notna().any() and g["depth"].notna().any():
                summary = (
                    f"Platform {platform}, {ym}: "
                    f"Temperature {g['temp_c'].min():.2f}-{g['temp_c'].max():.2f} °C, "
                    f"Salinity {g['salinity'].min():.2f}-{g['salinity'].max():.2f} PSU, "
                    f"Depth range {g['depth'].min():.1f}-{g['depth'].max():.1f} m."
                )
            else:
                summary = f"Platform {platform}, {ym}: No valid measurements available."

            metadata = {
                "platform_number": str(platform),
                "year_month": str(ym),
                "file": os.path.basename(file_path),
            }
            docs.append(Document(page_content=summary, metadata=metadata))

        return docs

    except Exception as e:
        print(f"Error in {file_path}: {e}")
        return []


def main():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    total_docs = 0
    for file in sorted(os.listdir(DATA_DIR)):
        if file.endswith(".parquet"):
            file_path = os.path.join(DATA_DIR, file)
            print(f"Summarizing {file_path} ...")
            docs = summarize_file(file_path)
            if docs:
                db.add_documents(docs)
                total_docs += len(docs)
                print(f"Added {len(docs)} summaries")
            else:
                print(f"No summaries generated for {file_path}")

    print(f"\nFinished! Total summaries stored: {total_docs}")


if __name__ == "__main__":
    main()
