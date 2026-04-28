import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT_DIR / "data" / "raw" / "emails.csv")

# Check some spam samples
print(df[df["spam"] == 1]["text"].head(10))

# Check some non-spam samples
print(df[df["spam"] == 0]["text"].head(10))
