from typing import Dict
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class BertPredictor:
    def __init__(self, model_dir: str, device: str = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, text: str) -> Dict:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        label = int(probs.argmax())
        return {"label": label, "label_name": "REAL" if label == 1 else "FAKE", "confidence_real": float(probs[1])}
