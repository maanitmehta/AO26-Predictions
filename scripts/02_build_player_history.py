import pandas as pd
from pathlib import Path

INPUT = Path("data/processed/ao_model_base.csv")
OUTPUT = Path("data/processed/player_match_history.csv")

def main():
    df = pd.read_csv(INPUT, parse_dates=["date"])

    winners = pd.DataFrame({
        "date": df["date"],
        "season": df["season"],
        "tourney_name": df["tournament"],
        "surface": df["surface"],
        "player": df["winner"],
        "player_rank": df["winner_rank"],
        "opponent": df["loser"],
        "opponent_rank": df["loser_rank"],
        "odds_for": df.get("b365w"),
        "odds_against": df.get("b365l"),
        "won": 1
    })

    losers = pd.DataFrame({
        "date": df["date"],
        "season": df["season"],
        "tourney_name": df["tournament"],
        "surface": df["surface"],
        "player": df["loser"],
        "player_rank": df["loser_rank"],
        "opponent": df["winner"],
        "opponent_rank": df["winner_rank"],
        "odds_for": df.get("b365l"),
        "odds_against": df.get("b365w"),
        "won": 0
    })

    history = pd.concat([winners, losers], ignore_index=True)
    history = history.sort_values("date").reset_index(drop=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(OUTPUT, index=False)

    print(f"Saved → {OUTPUT}")
    print(f"Rows: {len(history)}")

if __name__ == "__main__":
    main()
