import argparse
from src.utils import clean_text
from src.predict_tfidf import load_pipeline, predict_text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True, help="models/pipeline_lr.joblib or models/pipeline_svm.joblib")
    ap.add_argument("--text", required=True, help="Text to classify")
    args = ap.parse_args()

    pipe = load_pipeline(args.model_path)
    res = predict_text(pipe, clean_text(args.text))
    print(res)

if __name__ == "__main__":
    main()
