import pandas as pd
from pathlib import Path

INP = Path("data/processed/atp_all_matches_2000_2025.csv")
OUT = Path("data/processed/ao_model_base.csv")

def main():
    df = pd.read_csv(INP, low_memory=False)

    df.columns = df.columns.str.lower().str.strip()

    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["date", "winner", "loser"])

    df["surface"] = df["surface"].str.capitalize()
    df = df[df["surface"] == "Hard"]

    df = df.sort_values("date")

    keep = [
        "date", "season", "tournament", "surface",
        "winner", "loser", "winner_rank", "loser_rank"
    ]
    odds = [c for c in df.columns if c.startswith("b365")]
    df = df[keep + odds]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    print(f"Saved {len(df)} rows → {OUT}")

if __name__ == "__main__":
    main()
