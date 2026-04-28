import argparse
from pathlib import Path

import joblib


ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "naive_bayes_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.pkl"


def load_artifacts():
    missing_paths = [path for path in [MODEL_PATH, VECTORIZER_PATH] if not path.exists()]
    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Missing trained artifact(s): {missing}. Run scripts/train_model.py first.")

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def predict_email(text):
    model, vectorizer = load_artifacts()
    text_vec = vectorizer.transform([text])
    prediction = int(model.predict(text_vec)[0])
    probabilities = model.predict_proba(text_vec)[0]

    return {
        "label": prediction,
        "label_name": "Spam" if prediction == 1 else "Not Spam",
        "not_spam_probability": float(probabilities[0]),
        "spam_probability": float(probabilities[1]),
    }


def main():
    parser = argparse.ArgumentParser(description="Predict whether an email is spam.")
    parser.add_argument("text", help="Email text to classify")
    args = parser.parse_args()

    result = predict_email(args.text)
    print(f"Prediction: {result['label_name']} ({result['label']})")
    print(f"Not spam probability: {result['not_spam_probability']:.4f}")
    print(f"Spam probability: {result['spam_probability']:.4f}")


if __name__ == "__main__":
    main()
