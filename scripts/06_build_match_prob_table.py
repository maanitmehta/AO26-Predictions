import pandas as pd
import joblib
from pathlib import Path
import itertools
from scripts.name_utils import normalize_name

MODEL_PATH = Path("models/ao_logistic_model.pkl")
STATS_PATH = Path("data/processed/rolling_player_stats.csv")
DRAW_PATH = Path("data/processed/ao_2026_mens_draw.csv")
OUT_PATH = Path("data/processed/ao_2026_match_probs.csv")

REF_DATE = pd.to_datetime("2026-01-14")

def main():
    model = joblib.load(MODEL_PATH)

    stats = pd.read_csv(
        STATS_PATH,
        parse_dates=["date"],
        engine="python"
    )

    # Keep latest stats before AO
    stats = (
        stats[stats["date"] < REF_DATE]
        .sort_values("date")
        .groupby("player")
        .tail(1)
    )

    # Normalize stat names
    stats["norm_name"] = stats["player"].apply(normalize_name)
    stats = stats.set_index("norm_name")

    draw = pd.read_csv(DRAW_PATH)

    # Normalize draw names
    draw["A_norm"] = draw["player_A"].apply(normalize_name)
    draw["B_norm"] = draw["player_B"].apply(normalize_name)

    players = set(draw["A_norm"]) | set(draw["B_norm"])
    players = [p for p in players if p in stats.index]

    print(f"Matched players: {len(players)}")

    rows = []

    for A, B in itertools.permutations(players, 2):
        a = stats.loc[A]
        b = stats.loc[B]

        X = pd.DataFrame([{
            "winrate_diff": a["winrate_last10"] - b["winrate_last10"],
            "odds_diff": a["avg_odds_last10"] - b["avg_odds_last10"],
            "matches_diff": a["matches_played_last10"] - b["matches_played_last10"],
        }])

        p = model.predict_proba(X)[0, 1]

        rows.append({
            "player_A": A,
            "player_B": B,
            "p_A_wins": p
        })

    probs = pd.DataFrame(rows)
    probs.to_csv(OUT_PATH, index=False)

    print(f"Saved match probability table → {OUT_PATH}")
    print(f"Rows: {len(probs)}")

if __name__ == "__main__":
    main()
