import pandas as pd
import numpy as np
from pathlib import Path
from scripts.name_utils import canonical_name



MATCHES = Path("data/processed/ao_model_base.csv")
STATS = Path("data/processed/rolling_player_stats.csv")
OUTPUT = Path("data/processed/ao_ml_dataset.csv")

rankings = (
    pd.read_csv("data/raw/atp_rankings.csv")
    .sort_values("rank")                 # keep best (lowest) rank
    .drop_duplicates("player", keep="first")
    .assign(log_rank=lambda df: np.log(df["rank"]))
    .set_index("player")
)

def main():
    matches = pd.read_csv(MATCHES, parse_dates=["date"])
    stats = pd.read_csv(STATS, parse_dates=["date"])

    # Keep only numeric rolling stats
    stats = stats[[
        "player",
        "date",
        "winrate_last10",
        "avg_odds_last10",
        "matches_played_last10"
    ]]

    rows = []

    for _, m in matches.iterrows():
        date = m["date"]
        w = canonical_name(m["winner"])
        l = canonical_name(m["loser"])

        w_stats = stats[(stats["player"] == w) & (stats["date"] == date)]
        l_stats = stats[(stats["player"] == l) & (stats["date"] == date)]

        # Require exactly one row for each player
        if len(w_stats) != 1 or len(l_stats) != 1:
            continue

        w_stats = w_stats.iloc[0]
        l_stats = l_stats.iloc[0]

        # Skip if rolling features are missing
        if pd.isna(w_stats["winrate_last10"]) or pd.isna(l_stats["winrate_last10"]):
            continue

        rankA = rankings.loc[w, "log_rank"] if w in rankings.index else np.log(200)
        rankB = rankings.loc[l, "log_rank"] if l in rankings.index else np.log(200)

        # Winner as w
        rows.append({
            "winrate_diff": w_stats["winrate_last10"] - l_stats["winrate_last10"],
            "odds_diff": w_stats["avg_odds_last10"] - l_stats["avg_odds_last10"],
            "matches_diff": w_stats["matches_played_last10"] - l_stats["matches_played_last10"],
            "rank_diff": rankB - rankA,
            "a_wins": 1
        })

        # Loser as A
        rows.append({
            "winrate_diff": l_stats["winrate_last10"] - w_stats["winrate_last10"],
            "odds_diff": l_stats["avg_odds_last10"] - w_stats["avg_odds_last10"],
            "matches_diff": l_stats["matches_played_last10"] - w_stats["matches_played_last10"],
            "rank_diff": rankA - rankB,
            "a_wins": 0
        })

    df = pd.DataFrame(rows)

    # FINAL safety check
    df = df.dropna()
    df["a_wins"] = df["a_wins"].astype(int)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)

    print(f"Saved ML dataset → {OUTPUT}")
    print(f"Rows: {len(df)}")
    print("Class balance:")
    print(df['a_wins'].value_counts(normalize=True))

if __name__ == "__main__":
    main()
