"""Week 2 — Model Architecture & Training (Varolline AI Engineer internship).

Rebuilds the Titanic features from the raw CSV (same preprocessing logic as
Week 1, vendored in ``src/preprocessing.py``), trains three candidate
classifiers with GridSearchCV (5-fold CV, F1 scoring), evaluates each on a
held-out test set, and persists the winner.

Usage:
    python src/train.py
"""

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import TARGET_COL, build_preprocessor, load_raw

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "titanic.csv"
ART = ROOT / "artifacts"
REP = ROOT / "reports"
SEED = 42

CANDIDATES = {
    "logistic_regression": (
        LogisticRegression(max_iter=2000, random_state=SEED),
        {"clf__C": [0.1, 1.0, 10.0]},
    ),
    "random_forest": (
        RandomForestClassifier(random_state=SEED),
        {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [None, 10, 20],
            "clf__min_samples_split": [2, 5],
        },
    ),
    "hist_gradient_boosting": (
        HistGradientBoostingClassifier(random_state=SEED),
        {
            "clf__max_iter": [100, 200],
            "clf__learning_rate": [0.05, 0.1],
            "clf__max_depth": [None, 5, 10],
        },
    ),
}


def evaluate(y_true, y_pred, y_prob):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def main() -> None:
    df = load_raw(RAW)
    print(f"Loaded {len(df)} rows from {RAW}")

    # Same stratified 70/15/15 protocol as Week 1 (train + test used here).
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=SEED, stratify=df[TARGET_COL]
    )
    _, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=SEED, stratify=temp_df[TARGET_COL]
    )
    X_train = train_df.drop(columns=[TARGET_COL]).reset_index(drop=True)
    y_train = train_df[TARGET_COL].astype(int).reset_index(drop=True)
    X_test = test_df.drop(columns=[TARGET_COL]).reset_index(drop=True)
    y_test = test_df[TARGET_COL].astype(int).reset_index(drop=True)
    print(f"train={len(X_train)} test={len(X_test)}")

    results, best_name, best_f1, best_pipe = {}, None, -1.0, None

    for name, (clf, grid) in CANDIDATES.items():
        pipe = Pipeline([("preprocess", build_preprocessor()), ("clf", clf)])
        search = GridSearchCV(pipe, grid, cv=5, scoring="f1", n_jobs=-1)
        search.fit(X_train, y_train)
        pipe = search.best_estimator_
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        metrics = evaluate(y_test, y_pred, y_prob)
        results[name] = {
            "best_params": {k.replace("clf__", ""): v
                            for k, v in search.best_params_.items()},
            "cv_best_f1": float(search.best_score_),
            "test": metrics,
        }
        print(f"{name}: cv_f1={search.best_score_:.4f} "
              f"test_f1={metrics['f1']:.4f} acc={metrics['accuracy']:.4f}")
        if metrics["f1"] > best_f1:
            best_name, best_f1, best_pipe = name, metrics["f1"], pipe

    print(f"Winner: {best_name} (test F1={best_f1:.4f})")

    # --- persist ---
    ART.mkdir(parents=True, exist_ok=True)
    REP.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipe, ART / "model.joblib")
    # standalone fitted preprocessor (train only) for drift checks / reuse
    preprocessor = build_preprocessor().fit(X_train)
    joblib.dump(preprocessor, ART / "preprocessor.joblib")

    metrics_doc = {
        "winner": best_name,
        "seed": SEED,
        "test_size": len(X_test),
        "models": results,
    }
    (ART / "metrics.json").write_text(json.dumps(metrics_doc, indent=2))

    y_pred = best_pipe.predict(X_test)
    (REP / "classification_report.txt").write_text(
        f"Best model: {best_name}\n\n"
        + classification_report(y_test, y_pred, target_names=["Died", "Survived"])
    )
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Died", "Survived"])
    ax.set_yticks([0, 1], ["Died", "Survived"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix — {best_name}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(REP / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    print(f"Artifacts saved to {ART}/, reports to {REP}/")


if __name__ == "__main__":
    main()
