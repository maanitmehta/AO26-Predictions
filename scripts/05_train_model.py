import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

TOURS = ["atp", "wta"]

BASE_DATA = Path("data/processed")
BASE_MODELS = Path("models")

FEATURES = [
    "winrate_diff",
    "odds_diff",
    "matches_diff",
    "rank_diff",
]

def main():
    for tour in TOURS:
        print(f"\n=== Training model for {tour.upper()} ===")

        data_path = BASE_DATA / tour / "ml_dataset.csv"
        model_out = BASE_MODELS / f"{tour}_logistic_model.pkl"

        if not data_path.exists():
            raise RuntimeError(f"Missing ML dataset: {data_path}")

        df = pd.read_csv(data_path)

        X = df[FEATURES]
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

        BASE_MODELS.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, model_out)

        print(f"Saved model → {model_out}")

if __name__ == "__main__":
    main()
