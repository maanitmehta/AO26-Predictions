import pandas as pd
from pathlib import Path

MATCHES = Path("data/processed/ao_model_base.csv")
STATS = Path("data/processed/rolling_player_stats.csv")
OUTPUT = Path("data/processed/ao_ml_dataset.csv")

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
        w = m["winner"]
        l = m["loser"]

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

        # Winner as A
        rows.append({
            "winrate_diff": w_stats["winrate_last10"] - l_stats["winrate_last10"],
            "odds_diff": w_stats["avg_odds_last10"] - l_stats["avg_odds_last10"],
            "matches_diff": w_stats["matches_played_last10"] - l_stats["matches_played_last10"],
            "a_wins": 1
        })

        # Loser as A
        rows.append({
            "winrate_diff": l_stats["winrate_last10"] - w_stats["winrate_last10"],
            "odds_diff": l_stats["avg_odds_last10"] - w_stats["avg_odds_last10"],
            "matches_diff": l_stats["matches_played_last10"] - w_stats["matches_played_last10"],
            "a_wins": 0
        })

    df = pd.DataFrame(rows)

    # FINAL safety check
    df = df.dropna()
    df = df.astype(float)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)

    print(f"Saved ML dataset → {OUTPUT}")
    print(f"Rows: {len(df)}")
    print("Class balance:")
    print(df['a_wins'].value_counts(normalize=True))

if __name__ == "__main__":
    main()
