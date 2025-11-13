import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import joblib
import torch

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    DataCollatorWithPadding, TrainingArguments, Trainer, EarlyStoppingCallback
)

# ---------- Minimal cleaners ----------
URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
MD_RE = re.compile(r"\[(.*?)\]\((.*?)\)")

def clean_text(s: str) -> str:
    if not isinstance(s, str): s = str(s)
    s = MD_RE.sub(r"\1", s)          # keep anchor text, drop URL
    s = URL_RE.sub(" ", s)           # drop raw URLs
    s = re.sub(r"\s+", " ", s).strip()
    return s

def map_label(v):
    s = str(v).strip().lower()
    if s in {"1", "suicide", "suicidal", "suicidal_ideation"}: return 1
    if s in {"0", "non-suicide", "nonsuicide", "normal"}: return 0
    try:
        return int(float(s))
    except Exception:
        raise ValueError(f"Unrecognized label: {v}")

# ---------- Metrics ----------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    # prob for class 1 via logit difference
    probs = 1 / (1 + np.exp(-(logits[:, 1] - logits[:, 0])))
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = float("nan")
    acc = (preds == labels).mean()
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "roc_auc": auc}

def main():
    ap = argparse.ArgumentParser(description="Single-command BERT fine-tuning for suicide detection")
    ap.add_argument("--src_csv", default="data/raw/Suicide_Detection.csv", help="Path to Suicide_Detection.csv")
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--label_col", default="class")
    ap.add_argument("--model_name", default="bert-base-uncased")
    ap.add_argument("--max_length", type=int, default=320)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--patience", type=int, default=2, help="Early stopping patience (eval every epoch)")
    ap.add_argument("--out_dir", default="models/bert_base_uncased_single", help="Where to save model & artifacts")
    ap.add_argument("--outputs_dir", default="outputs_single", help="Where to save eval artifacts")
    args = ap.parse_args()

    # --- I/O prep
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir = Path(args.outputs_dir); outputs_dir.mkdir(parents=True, exist_ok=True)

    # --- Load CSV
    src = Path(args.src_csv)
    if not src.exists():
        raise FileNotFoundError(f"CSV not found: {src}. Pass --src_csv to your file.")
    df = pd.read_csv(src, encoding="utf-8", engine="python")

    # --- Select & clean
    cols = list(df.columns)
    if args.text_col not in cols or args.label_col not in cols:
        raise ValueError(f"Expected columns {args.text_col} and {args.label_col} in CSV. Found: {cols}")
    df = df[[args.text_col, args.label_col]].rename(columns={args.text_col: "text", args.label_col: "label"})
    df["clean_text"] = df["text"].map(clean_text)
    df["y"] = df["label"].map(map_label)

    before = len(df)
    df = df.dropna(subset=["clean_text", "y"]).drop_duplicates(subset=["clean_text","y"])
    after = len(df)

    # --- Split
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["y"])
    val_df, test_df  = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["y"])

    # Save a quick stats file
    stats = {
        "rows_before": int(before),
        "rows_after": int(after),
        "train": int(len(train_df)), "val": int(len(val_df)), "test": int(len(test_df)),
        "class_balance_train": train_df["y"].value_counts(normalize=True).to_dict(),
        "max_length": args.max_length
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))

    # --- Build HF Datasets (rename y->labels BEFORE tokenization)
    def to_hfds(pdf):
        hf = Dataset.from_pandas(pdf[["clean_text","y"]].rename(columns={"clean_text":"text"}), preserve_index=False)
        return hf.rename_column("y", "labels")
    ds = DatasetDict({
        "train": to_hfds(train_df),
        "validation": to_hfds(val_df),
        "test": to_hfds(test_df),
    })

    # --- Tokenization (labels are preserved automatically)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    def tok_fn(batch):
        return tokenizer(batch["text"], truncation=True, padding=False, max_length=args.max_length)
    ds_tok = ds.map(tok_fn, batched=True, remove_columns=["text"])

    # --- Model & Trainer
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    # set nice label maps (optional)
    model.config.id2label = {0: "non-suicide", 1: "suicide"}
    model.config.label2id = {"non-suicide": 0, "suicide": 1}

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",           # correct arg name for transformers>=4.57
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=2,
        fp16=False                       # keep False on macOS MPS + torch<2.5
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["validation"],
        tokenizer=tokenizer,             # deprecation warning is OK; can switch to processing_class later
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)]
    )

    trainer.train()

    # --- Save best model & tokenizer
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    # --- Threshold tuning on validation (maximize F1 of positive class)
    val_pred = trainer.predict(ds_tok["validation"])
    val_logits = val_pred.predictions
    val_probs  = 1 / (1 + np.exp(-(val_logits[:,1] - val_logits[:,0])))
    val_labels = np.array(ds_tok["validation"]["labels"])

    best_f1, best_t = -1.0, 0.5
    for t in np.linspace(0.1, 0.9, 81):
        pred_bin = (val_probs >= t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(val_labels, pred_bin, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    joblib.dump(best_t, out_dir / "threshold.joblib")
    (out_dir / "meta.json").write_text(json.dumps({"best_val_f1": float(best_f1), "threshold": float(best_t)}, indent=2))

    # --- Final evaluation on test
    test_pred  = trainer.predict(ds_tok["test"])
    test_logits = test_pred.predictions
    test_probs  = 1 / (1 + np.exp(-(test_logits[:,1] - test_logits[:,0])))
    test_labels = np.array(ds_tok["test"]["labels"])
    test_preds  = (test_probs >= best_t).astype(int)

    report = classification_report(test_labels, test_preds, digits=4)
    try:
        auc = roc_auc_score(test_labels, test_probs)
    except Exception:
        auc = float("nan")
    cm = confusion_matrix(test_labels, test_preds)

    # Save reports
    (outputs_dir / "classification_report.txt").write_text(report + f"\nROC_AUC: {auc:.4f}\n")
    pd.DataFrame({"prob_suicide": test_probs, "pred": test_preds, "label": test_labels}).to_csv(outputs_dir / "test_predictions.csv", index=False)
    (outputs_dir / "confusion_matrix.json").write_text(json.dumps(cm.tolist(), indent=2))

    print("=== Done ===")
    print("Model dir:", out_dir)
    print("Outputs dir:", outputs_dir)
    print("Best threshold (val F1):", best_t)

if __name__ == "__main__":
    main()
