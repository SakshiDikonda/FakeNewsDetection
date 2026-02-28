import joblib
from typing import Dict

def load_pipeline(path: str):
    return joblib.load(path)

def predict_text(pipe, text: str) -> Dict:
    label = int(pipe.predict([text])[0])

    clf = pipe.named_steps["clf"]
    score = None
    if hasattr(clf, "predict_proba"):
        Xv = pipe.named_steps["tfidf"].transform([text])
        proba = clf.predict_proba(Xv)[0]
        score = float(proba[1])  # prob REAL
    elif hasattr(clf, "decision_function"):
        import math
        Xv = pipe.named_steps["tfidf"].transform([text])
        margin = float(clf.decision_function(Xv)[0])
        score = 1.0 / (1.0 + math.exp(-margin))  # pseudo-prob REAL

    return {"label": label, "label_name": "REAL" if label == 1 else "FAKE", "confidence_real": score}
