# aiops_ticket_classifier.py
"""
A minimal, self‑contained prototype that shows basic machine‑learning to support‑ticket data.

Features
--------
* One‑file script (easy to read, no packages to install locally).
* Uses scikit‑learn’s TF‑IDF + Logistic Regression – simple, reliable, fast.
* Two sub‑commands:
      ▸ train   – trains and saves a model from a CSV of historical tickets  
      ▸ predict – loads the saved model and classifies a single ticket text
* Clear, plain‑English comments that explain each step.
* Graceful fallback for tiny demo datasets.
"""

import argparse
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(csv_path: str):
    """
    Expect a CSV with at least two columns:
    --------------------------------------
    description   – free‑text problem description
    category      – label / class you want to predict
    """
    if not os.path.exists(csv_path):
        print(f"[ERROR] File '{csv_path}' not found.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    required = {"description", "category"}
    if not required.issubset(df.columns):
        print("[ERROR] CSV must contain 'description' and 'category' columns.")
        sys.exit(1)

    return df[["description", "category"]]


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------
def build_pipeline():
    """TF‑IDF text vectoriser  ➜  Logistic Regression classifier."""
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(stop_words="english")),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------
def train(csv_path: str, model_path: str):
    """Train the model and save it to disk."""
    data = load_data(csv_path)

    # Try a stratified split first (keeps label distribution),
    # fall back to a plain random split if classes are too small.
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            data["description"],
            data["category"],
            test_size=0.2,
            random_state=42,
            stratify=data["category"],
        )
    except ValueError:
        print("[WARN] Some classes too small ➜ using simple random split.")
        X_train, X_test, y_train, y_test = train_test_split(
            data["description"], data["category"], test_size=0.2, random_state=42
        )

    model = build_pipeline()
    print("[INFO] Training model …")
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    print(f"[INFO] Model saved to {model_path}")

    # Quick sanity‑check
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[INFO] Accuracy on held‑out set: {acc:.2%}")
    print("\nDetailed classification report:\n")
    print(classification_report(y_test, y_pred))


# ---------------------------------------------------------------------------
# Single‑ticket prediction helper
# ---------------------------------------------------------------------------
def predict_single(text: str, model_path: str):
    """Predict a category for one ticket description."""
    if not os.path.exists(model_path):
        print(f"[ERROR] Model '{model_path}' not found. Train first.")
        sys.exit(1)

    model = joblib.load(model_path)

    pred = model.predict([text])[0]
    if hasattr(model, "predict_proba"):
        conf = model.predict_proba([text]).max() * 100
        print(f"Prediction ➜ {pred}   ({conf:.1f}% confidence)")
    else:
        print(f"Prediction ➜ {pred}")


# ---------------------------------------------------------------------------
# Command‑line interface
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="AIOps Ticket Classifier • Minimal Prototype"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # train
    t = subparsers.add_parser("train", help="Train a new model")
    t.add_argument("csv", help="CSV file with historical tickets")
    t.add_argument(
        "--out", default="ticket_model.joblib", help="Where to store the model"
    )

    # predict
    p = subparsers.add_parser("predict", help="Predict a single ticket category")
    p.add_argument("text", help='Ticket description (wrap in quotes)')
    p.add_argument(
        "--model", default="ticket_model.joblib", help="Path to trained model file"
    )

    args = parser.parse_args()
    if args.command == "train":
        train(args.csv, args.out)
    elif args.command == "predict":
        predict_single(args.text, args.model)


if __name__ == "__main__":
    main()
