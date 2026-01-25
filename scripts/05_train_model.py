import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

DATA = Path("data/processed/ao_ml_dataset.csv")
MODEL_OUT = Path("models/ao_logistic_model.pkl")

def main():
    df = pd.read_csv(DATA)

    X = df[["winrate_diff", "odds_diff", "matches_diff"]]
    y = df["a_wins"]

    # Time-safe split: first 80% train, last 20% test
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_proba)

    print("MODEL PERFORMANCE")
    print(f"Accuracy: {acc:.4f}")
    print(f"Log loss: {ll:.4f}")

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_OUT)

    print(f"\nSaved model → {MODEL_OUT}")

if __name__ == "__main__":
    main()
