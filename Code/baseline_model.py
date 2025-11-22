
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)
from sklearn.model_selection import GridSearchCV
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd



# 1. Improved TF-IDF Vectorization
def vectorize_text(train, val, test):
    tfidf = TfidfVectorizer(
        max_features=100000,
        min_df=3,
        ngram_range=(1, 3),
        sublinear_tf=True
    )

    X_train = tfidf.fit_transform(train)
    X_val = tfidf.transform(val)
    X_test = tfidf.transform(test)

    return tfidf, X_train, X_val, X_test


# 2. Logistic Regression with Hyperparameter Tuning
def tune_logreg(X_train, y_train):

    logreg = LogisticRegression(max_iter=500, class_weight="balanced")

    param_grid = {
        "C": [0.1, 1, 10],
        "solver": ["lbfgs"],
        "penalty": ["l2"]
    }

    grid = GridSearchCV(
        logreg,
        param_grid,
        cv=3,
        scoring="f1_macro",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("\nBest Parameters:", grid.best_params_)
    print("Best F1 Score:", grid.best_score_)

    return grid.best_estimator_


# 3. Evaluate Model
def evaluate_model(model, X, y, title="Evaluation"):
    y_pred = model.predict(X)

    print(f"        {title} Results")
    print(classification_report(y, y_pred))
    print("Accuracy:", accuracy_score(y, y_pred))

    cm = confusion_matrix(y, y_pred, labels=model.classes_)
    sns.heatmap(
        pd.DataFrame(cm, index=model.classes_, columns=model.classes_),
        annot=True, fmt="d", cmap="Blues"
    )
    plt.title(f"Confusion Matrix - {title}")
    plt.show()

# 4. Save Model
def save_model(model, tfidf, prefix="baseline"):
    joblib.dump(model, f"{prefix}_lr.pkl")
    joblib.dump(tfidf, f"{prefix}_tfidf.pkl")
    print(f"\nSaved: {prefix}_lr.pkl | {prefix}_tfidf.pkl")
