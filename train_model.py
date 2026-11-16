"""Train a sentiment classifier and save it to model.joblib.

Examples
--------
IMDB (binary, text/label columns):
    python train_model.py --data IMDB_Dataset.csv --text-col review --label-col sentiment

Amazon-style star ratings (1-5 -> negative / neutral / positive):
    python train_model.py --data reviews.csv --text-col reviewText --label-col overall
"""
import argparse

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocess import clean_text


def to_label(value):
    """Map text labels or 1-5 star ratings to negative/neutral/positive."""
    s = str(value).strip().lower()
    if s in {"positive", "pos", "1.0_pos"}:
        return "positive"
    if s in {"negative", "neg"}:
        return "negative"
    if s == "neutral":
        return "neutral"
    try:
        stars = float(s)
    except ValueError:
        return None
    if stars <= 2:
        return "negative"
    if stars == 3:
        return "neutral"
    return "positive"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to CSV file")
    p.add_argument("--text-col", default="review")
    p.add_argument("--label-col", default="sentiment")
    p.add_argument("--max-rows", type=int, default=50000, help="Limit rows for speed")
    p.add_argument("--out", default="model.joblib")
    args = p.parse_args()

    df = pd.read_csv(args.data)
    df = df[[args.text_col, args.label_col]].dropna()
    df["label"] = df[args.label_col].map(to_label)
    df = df.dropna(subset=["label"])
    if len(df) > args.max_rows:
        df = df.sample(args.max_rows, random_state=42)

    print("Class distribution:\n", df["label"].value_counts(), "\n")

    X_train, X_test, y_train, y_test = train_test_split(
        df[args.text_col], df["label"],
        test_size=0.2, random_state=42, stratify=df["label"],
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=clean_text, ngram_range=(1, 2),
            min_df=3, max_df=0.9, sublinear_tf=True, max_features=200_000,
        )),
        ("clf", LogisticRegression(max_iter=1000, C=2.0, class_weight="balanced")),
    ])

    print("Training...")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}\n")
    print(classification_report(y_test, preds))

    joblib.dump(model, args.out)
    print(f"Saved model to {args.out}")


if __name__ == "__main__":
    main()
