import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

from baseline_model import (
    vectorize_text,
    tune_logreg,
    evaluate_model,
    save_model
)


# 1. Setup

nltk.download("stopwords")
nltk.download("wordnet")
STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# 2. Load Dataset
DATA_PATH = "Suicide_Detection.csv"
df = pd.read_csv(DATA_PATH)

# adjust columns
if "post_text" in df.columns:
    df["text"] = df["title"].fillna("") + " " + df["post_text"].fillna("")
if "class" in df.columns:
    df["label"] = df["class"]

df = df[["text", "label"]].dropna().reset_index(drop=True)


# 3. Text Cleaning Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = [
        LEMMATIZER.lemmatize(w)
        for w in text.split()
        if w not in STOP_WORDS
    ]
    return " ".join(tokens)

df["clean_text"] = df["text"].apply(clean_text)

# 4. Stratified Splits
X = df["clean_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.20, stratify=y_train, random_state=42
)


# 5. TF-IDF (Improved)
tfidf, X_train_tfidf, X_val_tfidf, X_test_tfidf = vectorize_text(
    X_train, X_val, X_test
)


# 6. Train Logistic Regression using Hyperparameter Tuning
best_model = tune_logreg(X_train_tfidf, y_train)


# 7. Evaluate Model

evaluate_model(best_model, X_val_tfidf, y_val, title="Validation Set")
evaluate_model(best_model, X_test_tfidf, y_test, title="Test Set")


# 8. Save model + TF-IDF

save_model(best_model, tfidf, "improved_baseline")
