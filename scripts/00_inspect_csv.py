from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = ROOT / "dataset" / "csv"

print("CSV_DIR:", CSV_DIR)
files = sorted(CSV_DIR.glob("*.csv"))
print("Found CSV:", [f.name for f in files])

for f in files:
    df = pd.read_csv(f)
    print("\n===", f.name, "===")
    print("rows:", len(df))
    print("cols:", list(df.columns))
    print(df.head(3).to_string(index=False))
