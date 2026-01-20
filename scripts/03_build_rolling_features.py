import pandas as pd
from pathlib import Path

INPUT = Path("data/processed/player_match_history.csv")
OUTPUT = Path("data/processed/rolling_player_stats.csv")

WINDOW = 10

def main():
    df = pd.read_csv(INPUT, parse_dates=["date"])

    df = df.sort_values(["player", "date"])

    df["matches_played_last10"] = (
        df.groupby("player")["won"]
        .shift(1)
        .rolling(WINDOW, min_periods=1)
        .count()
    )

    df["winrate_last10"] = (
        df.groupby("player")["won"]
        .shift(1)
        .rolling(WINDOW, min_periods=1)
        .mean()
    )

    df["avg_odds_last10"] = (
        df.groupby("player")["odds_for"]
        .shift(1)
        .rolling(WINDOW, min_periods=1)
        .mean()
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)

    print(f"Saved → {OUTPUT}")
    print(f"Rows: {len(df)}")

if __name__ == "__main__":
    main()
