import argparse
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", pos_label=1)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Cleaned CSV with columns: text,label (1=REAL,0=FAKE)")
    ap.add_argument("--base-model", default="distilbert-base-uncased")
    ap.add_argument("--out", default="models/bert_distilbert")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(
        df, test_size=args.test_size, random_state=args.seed, stratify=df["label"]
    )

    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    test_ds = Dataset.from_pandas(test_df.reset_index(drop=True))

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_len)

    train_ds = train_ds.map(tok, batched=True)
    test_ds = test_ds.map(tok, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=2)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    import os
    os.makedirs(args.out, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=args.out,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        seed=args.seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("\nFinal eval metrics:", metrics)

    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"\nSaved BERT model to: {args.out}")

if __name__ == "__main__":
    main()
