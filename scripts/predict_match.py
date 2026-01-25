import pandas as pd
import joblib
from pathlib import Path
from scripts.name_utils import canonical_name

MODEL_PATH = Path("models/ao_logistic_model.pkl")
STATS_PATH = Path("data/processed/rolling_player_stats.csv")
REF_DATE = pd.to_datetime("2026-01-14")

# Load once
_model = joblib.load(MODEL_PATH)
_stats = pd.read_csv(
    STATS_PATH,
    parse_dates=["date"],
    engine="python"
)
_strength = pd.read_csv(
    "data/processed/player_strength.csv",
    index_col=0
)["winrate_last10"]


# Keep latest pre-AO stats per player
_stats = (
    _stats[_stats["date"] < REF_DATE]
    .sort_values("date")
    .groupby("player")
    .tail(1)
    .set_index("player")
)

# 🔹 in-memory cache
_match_cache = {}

def safe_diff(x, y, default=0.0):
    if pd.isna(x) or pd.isna(y):
        return default
    return x - y

def predict_match(A, B):
    """
    Probability that A beats B
    """
    A = canonical_name(A)
    B = canonical_name(B)
    if (A, B) in _match_cache:
        return _match_cache[(A, B)]

    if A not in _stats.index or B not in _stats.index:
        p = 0.45
    else:
        a = _stats.loc[A]
        b = _stats.loc[B]

        X = pd.DataFrame([{
            "winrate_diff": safe_diff(a["winrate_last10"], b["winrate_last10"], 0.0),
            "odds_diff": safe_diff(a["avg_odds_last10"], b["avg_odds_last10"], 0.0),
            "matches_diff": safe_diff(a["matches_played_last10"], b["matches_played_last10"], 0.0),
        }])

        if pd.isna(X.values).any():
    	    print(f"⚠️ NaN handled for match {A} vs {B}")

        model_p = float(_model.predict_proba(X)[0, 1])
        sA = _strength.get(A, 0.5)
        sB = _strength.get(B, 0.5)
        strength_p = sA / (sA + sB)
        p = 0.6 * model_p + 0.4 * strength_p

        p = float(p)
        p = min(max(p, 0.05), 0.95)


    _match_cache[(A, B)] = p
    return p
