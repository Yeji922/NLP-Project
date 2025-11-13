
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def predict_texts(texts, model_dir, max_length=320, batch_size=64):
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    threshold = joblib.load(Path(model_dir) / "threshold.joblib")
    model.eval()

    all_probs, all_preds = [], []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        enc = tokenizer(batch_texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits
        # probability of class 1 using logit difference
        probs = 1 / (1 + torch.exp(-(logits[:,1] - logits[:,0])))
        preds = (probs >= threshold).long()
        all_probs.append(probs.cpu().numpy())
        all_preds.append(preds.cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)
    preds = np.concatenate(all_preds, axis=0)
    return probs, preds

def main():
    ap = argparse.ArgumentParser(description="Predict suicide risk using a fine-tuned BERT model.")
    ap.add_argument("--model_dir", default="models/bert_base_uncased_single", help="Directory with saved model and threshold.joblib")
    ap.add_argument("--text", default=None, help="Single text to classify")
    ap.add_argument("--csv", default=None, help="CSV with a 'text' or 'clean_text' column")
    ap.add_argument("--out", default="outputs_single/infer_predictions.csv", help="Output CSV for batch inference")
    ap.add_argument("--max_length", type=int, default=320)
    args = ap.parse_args()

    Path("outputs_single").mkdir(exist_ok=True)

    if args.text:
        probs, preds = predict_texts([args.text], args.model_dir, args.max_length)
        print(f"prob_suicide={float(probs[0]):.4f}, pred={int(preds[0])}")
        return

    if args.csv:
        df = pd.read_csv(args.csv)
        col = "clean_text" if "clean_text" in df.columns else "text"
        if col not in df.columns:
            raise SystemExit("CSV must contain 'text' or 'clean_text' column.")
        texts = df[col].astype(str).tolist()
        probs, preds = predict_texts(texts, args.model_dir, args.max_length)
        out_df = pd.DataFrame({"prob_suicide": probs, "pred": preds})
        out_df.to_csv(args.out, index=False)
        print(f"Saved predictions to {args.out}")
        return

    raise SystemExit("Provide --text or --csv.")

if __name__ == "__main__":
    main()
