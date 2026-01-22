import polars as pl
import pathlib
import sys

claims_path = pathlib.Path("data/raw/claims.csv")

if not claims_path.exists():
    print(
        "❌ Missing required file: data/raw/claims.csv\n\n"
        "Create it by running:\n\n"
        "  python scripts/generate_synthetic.py --count 10000 "
        "--output data/raw/claims.csv --format csv\n"
    )
    sys.exit(1)

df_raw = pl.read_csv(claims_path)
df_approved = df_raw.filter(pl.col("has_error") == False)

pathlib.Path("data/processed").mkdir(parents=True, exist_ok=True)
df_approved.write_parquet("data/processed/claims_approved.parquet")

print(
    f"✅ Created: data/processed/claims_approved.parquet "
    f"with {len(df_approved)} cases."
)
