import pandas as pd
import numpy as np
from pathlib import Path
from scripts.name_utils import canonical_name

TOURS = ["atp", "wta"]

BASE_PROCESSED = Path("data/processed")
BASE_RAW = Path("data/raw")

def main():
    for tour in TOURS:
        print(f"\n=== Building ML dataset for {tour.upper()} ===")

        matches_path = BASE_PROCESSED / tour / "model_base.csv"
        stats_path = BASE_PROCESSED / tour / "rolling_player_stats.csv"
        out_dir = BASE_PROCESSED / tour
        out_file = out_dir / "ml_dataset.csv"

        rankings_path = BASE_RAW / f"{tour}_rankings.csv"

        if not matches_path.exists():
            raise RuntimeError(f"Missing matches file: {matches_path}")
        if not stats_path.exists():
            raise RuntimeError(f"Missing stats file: {stats_path}")
        if not rankings_path.exists():
            raise RuntimeError(f"Missing rankings file: {rankings_path}")

        matches = pd.read_csv(matches_path, parse_dates=["date"])
        stats = pd.read_csv(stats_path, parse_dates=["date"])

        # Load and sanitise rankings (one row per player)
        rankings = (
            pd.read_csv(rankings_path)
            .sort_values("rank")
            .drop_duplicates("player", keep="first")
            .assign(log_rank=lambda df: np.log(df["rank"]))
            .set_index("player")
        )

        if rankings.index.duplicated().any():
            raise ValueError(f"Duplicate players found in {tour} rankings")

        # Keep only required rolling stats
        stats = stats[[
            "player",
            "date",
            "winrate_lastN",
            "avg_odds_lastN",
            "matches_played_lastN",
        ]]

        rows = []

        for _, m in matches.iterrows():
            date = m["date"]
            w = canonical_name(m["winner"])
            l = canonical_name(m["loser"])

            w_stats = stats[(stats["player"] == w) & (stats["date"] == date)]
            l_stats = stats[(stats["player"] == l) & (stats["date"] == date)]

            # Require exactly one stats row per player
            if len(w_stats) != 1 or len(l_stats) != 1:
                continue

            w_stats = w_stats.iloc[0]
            l_stats = l_stats.iloc[0]

            # Skip if rolling features are missing
            if pd.isna(w_stats["winrate_lastN"]) or pd.isna(l_stats["winrate_lastN"]):
                continue

            rankA = float(rankings.loc[w, "log_rank"]) if w in rankings.index else np.log(200)
            rankB = float(rankings.loc[l, "log_rank"]) if l in rankings.index else np.log(200)

            # Winner row
            rows.append({
                "winrate_diff": w_stats["winrate_lastN"] - l_stats["winrate_lastN"],
                "odds_diff": w_stats["avg_odds_lastN"] - l_stats["avg_odds_lastN"],
                "matches_diff": w_stats["matches_played_lastN"] - l_stats["matches_played_lastN"],
                "rank_diff": rankB - rankA,
                "a_wins": 1,
            })

            # Loser row
            rows.append({
                "winrate_diff": l_stats["winrate_lastN"] - w_stats["winrate_lastN"],
                "odds_diff": l_stats["avg_odds_lastN"] - w_stats["avg_odds_lastN"],
                "matches_diff": l_stats["matches_played_lastN"] - w_stats["matches_played_lastN"],
                "rank_diff": rankA - rankB,
                "a_wins": 0,
            })

        df = pd.DataFrame(rows)

        # Final safety checks
        df = df.dropna()
        df["a_wins"] = df["a_wins"].astype(int)

        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_file, index=False)

        print(f"Saved ML dataset → {out_file}")
        print(f"Rows: {len(df):,}")
        print("Class balance:")
        print(df["a_wins"].value_counts(normalize=True))

if __name__ == "__main__":
    main()
