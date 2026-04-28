from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "emails.csv"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"

TEXT_COLUMN = "text"
LABEL_COLUMN = "spam"


def main():
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Missing raw data file: {RAW_DATA_PATH}")

    df = pd.read_csv(RAW_DATA_PATH)
    missing_columns = {TEXT_COLUMN, LABEL_COLUMN} - set(df.columns)
    if missing_columns:
        raise ValueError(f"{RAW_DATA_PATH} is missing columns: {sorted(missing_columns)}")

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        stratify=df[LABEL_COLUMN],
    )

    validation_df, test_df = train_test_split(
        temp_df,
        test_size=1 / 3,
        random_state=42,
        stratify=temp_df[LABEL_COLUMN],
    )

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
    validation_df.to_csv(PROCESSED_DATA_DIR / "validation.csv", index=False)
    test_df.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)

    total_rows = len(df)
    for split_name, split_df in [
        ("Train", train_df),
        ("Validation", validation_df),
        ("Test", test_df),
    ]:
        percent = len(split_df) / total_rows * 100
        spam_rate = split_df[LABEL_COLUMN].mean() * 100
        print(f"{split_name}: {split_df.shape} ({percent:.1f}% of data, {spam_rate:.1f}% spam)")


if __name__ == "__main__":
    main()
