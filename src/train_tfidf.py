import argparse
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix

def build_pipeline(model_type: str, max_features: int, ngram_max: int):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, ngram_max),
        max_features=max_features
    )

    if model_type == "lr":
        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        )
    elif model_type == "svm":
        clf = LinearSVC(class_weight="balanced")
    else:
        raise ValueError("model_type must be 'lr' or 'svm'")

    return Pipeline([("tfidf", vectorizer), ("clf", clf)])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Cleaned CSV with columns: text,label")
    ap.add_argument("--model", required=True, choices=["lr", "svm"], help="Choose model")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-features", type=int, default=40000)
    ap.add_argument("--ngram-max", type=int, default=2)
    ap.add_argument("--outdir", default="models")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    if not {"text", "label"}.issubset(df.columns):
        raise SystemExit("Expected columns: text,label")

    X = df["text"].astype(str)
    y = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    pipe = build_pipeline(args.model, args.max_features, args.ngram_max)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["FAKE(0)", "REAL(1)"]))
    print("Confusion matrix [ [TN FP], [FN TP] ]:")
    print(confusion_matrix(y_test, y_pred))

    import os
    os.makedirs(args.outdir, exist_ok=True)
    out_path = f"{args.outdir}/pipeline_{args.model}.joblib"
    joblib.dump(pipe, out_path)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
