import pandas as pd
import joblib
from pathlib import Path
from scripts.name_utils import canonical_name

TOURS = ["atp", "wta"]

BASE_MODELS = Path("models")
BASE_DATA = Path("data/processed")

REF_DATE = pd.to_datetime("2026-01-14")

# -----------------------------
# Load everything ONCE (cached)
# -----------------------------
_MODELS = {}
_STATS = {}

for tour in TOURS:
    model_path = BASE_MODELS / f"{tour}_logistic_model.pkl"
    stats_path = BASE_DATA / tour / "rolling_player_stats.csv"

    if not model_path.exists():
        raise RuntimeError(f"Missing model: {model_path}")
    if not stats_path.exists():
        raise RuntimeError(f"Missing stats: {stats_path}")

    _MODELS[tour] = joblib.load(model_path)

    stats = pd.read_csv(stats_path, parse_dates=["date"], engine="python")

    # Keep latest pre-AO stats per player
    stats = (
        stats[stats["date"] < REF_DATE]
        .sort_values("date")
        .groupby("player")
        .tail(1)
        .set_index("player")
    )

    _STATS[tour] = stats

# In-memory cache
_match_cache = {}

def safe_diff(x, y, default=0.0):
    if pd.isna(x) or pd.isna(y):
        return default
    return x - y

def predict_match(A, B, tour="atp"):
    """
    Probability that player A beats player B
    """
    A = canonical_name(A)
    B = canonical_name(B)

    key = (A, B, tour)
    if key in _match_cache:
        return _match_cache[key]

    stats = _STATS[tour]
    model = _MODELS[tour]

    # Fallback if stats missing
    if A not in stats.index or B not in stats.index:
        p = 0.45
    else:
        a = stats.loc[A]
        b = stats.loc[B]

        X = pd.DataFrame([{
            "winrate_diff": safe_diff(a["winrate_lastN"], b["winrate_lastN"]),
            "odds_diff": safe_diff(a["avg_odds_lastN"], b["avg_odds_lastN"]),
            "matches_diff": safe_diff(a["matches_played_lastN"], b["matches_played_lastN"]),
            "rank_diff": 0.0  # already learned in training; no leakage at prediction time
        }])

        if pd.isna(X.values).any():
            print(f"⚠ NaN handled for match {A} vs {B} ({tour.upper()})")

        p = float(model.predict_proba(X)[0, 1])
        p = min(max(p, 0.05), 0.95)

    _match_cache[key] = p
    return p
