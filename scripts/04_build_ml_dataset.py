import pandas as pd
from pathlib import Path

MATCHES = Path("data/processed/ao_model_base.csv")
STATS = Path("data/processed/rolling_player_stats.csv")
OUTPUT = Path("data/processed/ao_ml_dataset.csv")

def main():
    matches = pd.read_csv(MATCHES, parse_dates=["date"])
    stats = pd.read_csv(STATS, parse_dates=["date"])

    stats = stats[[
        "date",
        "player",
        "winrate_last10",
        "avg_odds_last10",
        "matches_played_last10"
    ]]

    A = matches.merge(
        stats,
        left_on=["winner", "date"],
        right_on=["player", "date"],
        how="left"
    ).rename(columns={
        "winrate_last10": "winrate_A",
        "avg_odds_last10": "odds_A",
        "matches_played_last10": "matches_A"
    })

    AB = A.merge(
        stats,
        left_on=["loser", "date"],
        right_on=["player", "date"],
        how="left"
    ).rename(columns={
        "winrate_last10": "winrate_B",
        "avg_odds_last10": "odds_B",
        "matches_played_last10": "matches_B"
    })

    df = pd.DataFrame({
        "winrate_diff": AB["winrate_A"] - AB["winrate_B"],
        "odds_diff": AB["odds_A"] - AB["odds_B"],
        "matches_diff": AB["matches_A"] - AB["matches_B"],
        "a_wins": 1
    })

    df = df.dropna().reset_index(drop=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)

    print(f"Saved → {OUTPUT}")
    print(f"Rows: {len(df)}")

if __name__ == "__main__":
    main()
