"""
train.py
--------
Trains the disease-prediction model for MediScope.

Data: data/Training.csv — 4,920 labeled patient records, 132 binary symptom
features, 41 disease classes (a real, publicly documented symptom-disease
dataset used across healthcare ML coursework and cited in the accompanying
README; not synthetically generated for this project).

We train several standard classifiers, evaluate them with stratified
cross-validation AND a held-out test split, and persist the best one plus
all the metadata the API needs at inference time.

Run:
    python src/train.py
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_DIR / "Training.csv")
    # The source CSV sometimes ships a trailing unnamed index column — drop if present.
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df = df.dropna(axis=1, how="all")

    target_col = "prognosis"
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols].astype(int)

    # Known typo in the source Training.csv: "Peptic ulcer diseae" (missing
    # the 's' in "disease"), which doesn't match the spelling used in
    # description.csv / medications.csv / diets.csv / workout_df.csv.
    # Fixed here, once, at the label level, rather than patched around later.
    LABEL_FIXES = {"Peptic ulcer diseae": "Peptic ulcer disease"}
    y_raw = df[target_col].str.strip().replace(LABEL_FIXES)

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    return X, y, feature_cols, encoder


def build_candidates():
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "SVC": SVC(kernel="linear", probability=True, random_state=42),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "KNeighbors": KNeighborsClassifier(n_neighbors=5),
        "NaiveBayes": GaussianNB(),
    }


def main():
    print("Loading data...")
    X, y, feature_cols, encoder = load_data()
    print(f"  {X.shape[0]} records, {X.shape[1]} symptom features, "
          f"{len(encoder.classes_)} disease classes")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = {}
    fitted_models = {}

    for name, model in build_candidates().items():
        t0 = time.time()
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, n_jobs=-1)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        test_acc = accuracy_score(y_test, preds)
        test_f1 = f1_score(y_test, preds, average="weighted")
        elapsed = time.time() - t0

        results[name] = {
            "cv_mean_accuracy": round(float(cv_scores.mean()), 4),
            "cv_std": round(float(cv_scores.std()), 4),
            "test_accuracy": round(float(test_acc), 4),
            "test_f1_weighted": round(float(test_f1), 4),
            "train_seconds": round(elapsed, 2),
        }
        fitted_models[name] = model
        print(f"  {name:<18} cv_acc={cv_scores.mean():.4f}  "
              f"test_acc={test_acc:.4f}  test_f1={test_f1:.4f}  ({elapsed:.1f}s)")

    best_name = max(results, key=lambda n: results[n]["test_f1_weighted"])
    best_model = fitted_models[best_name]
    print(f"\nBest model: {best_name} "
          f"(test_accuracy={results[best_name]['test_accuracy']})")

    report = classification_report(
        y_test,
        best_model.predict(X_test),
        target_names=encoder.classes_,
        zero_division=0,
    )
    print("\nClassification report (best model, held-out test set):\n", report)

    # Persist everything the API needs.
    joblib.dump(best_model, MODELS_DIR / "model.pkl")
    joblib.dump(encoder, MODELS_DIR / "label_encoder.pkl")
    with open(MODELS_DIR / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(
            {
                "best_model": best_name,
                "results": results,
                "n_records": int(X.shape[0]),
                "n_features": int(X.shape[1]),
                "n_classes": int(len(encoder.classes_)),
            },
            f,
            indent=2,
        )

    print(f"\nSaved model.pkl, label_encoder.pkl, feature_columns.json, "
          f"metrics.json -> {MODELS_DIR}")


if __name__ == "__main__":
    main()
