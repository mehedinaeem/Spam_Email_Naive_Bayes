# train_random_forest.py
# Adapted from train_model.py for Random Forest Classifier

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

TEXT_COLUMN = "text"
LABEL_COLUMN = "spam"
LABELS = [0, 1]
TARGET_NAMES = ["Not Spam", "Spam"]

def load_split(split_name):
    path = DATA_DIR / f"{split_name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {split_name} split: {path}")

    df = pd.read_csv(path)
    missing_columns = {TEXT_COLUMN, LABEL_COLUMN} - set(df.columns)
    if missing_columns:
        raise ValueError(f"{path} is missing columns: {sorted(missing_columns)}")

    return df[TEXT_COLUMN].astype(str), df[LABEL_COLUMN].astype(int)

def evaluate_split(model, vectorizer, split_name, x_values, y_values):
    x_vec = vectorizer.transform(x_values)
    predictions = model.predict(x_vec)
    accuracy = accuracy_score(y_values, predictions)
    report = classification_report(
        y_values,
        predictions,
        labels=LABELS,
        target_names=TARGET_NAMES,
        digits=4,
    )
    matrix = confusion_matrix(y_values, predictions, labels=LABELS)

    return (
        f"{split_name.title()} Accuracy: {accuracy:.4f}\n"
        f"{split_name.title()} Classification Report:\n{report}\n"
        f"{split_name.title()} Confusion Matrix [[TN, FP], [FN, TP]]:\n{matrix}\n"
    )

def main():
    x_train, y_train = load_split("train")
    x_validation, y_validation = load_split("validation")
    x_test, y_test = load_split("test")

    vectorizer = CountVectorizer(stop_words="english")
    x_train_vec = vectorizer.fit_transform(x_train)

    model = RandomForestClassifier(n_estimators=100, random_state=42)  # You can tune hyperparameters here
    model.fit(x_train_vec, y_train)

    sections = [
        evaluate_split(model, vectorizer, "train", x_train, y_train),
        evaluate_split(model, vectorizer, "validation", x_validation, y_validation),
        evaluate_split(model, vectorizer, "test", x_test, y_test),
    ]
    report_text = "\n".join(sections)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODELS_DIR / "random_forest_model.pkl")
    joblib.dump(vectorizer, MODELS_DIR / "vectorizer.pkl")  # Reusing the same vectorizer
    (REPORTS_DIR / "random_forest_classification_report.txt").write_text(report_text, encoding="utf-8")

    print(report_text)
    print(f"Saved model to: {MODELS_DIR / 'random_forest_model.pkl'}")
    print(f"Saved vectorizer to: {MODELS_DIR / 'vectorizer.pkl'}")
    print(f"Saved report to: {REPORTS_DIR / 'random_forest_classification_report.txt'}")

if __name__ == "__main__":
    main()