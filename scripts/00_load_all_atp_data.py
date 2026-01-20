import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw/tennis_data/atp")
OUT = Path("data/processed/atp_all_matches_2000_2025.csv")

def main():
    dfs = []

    files = sorted(
        list(RAW_DIR.glob("*.xls")) +
        list(RAW_DIR.glob("*.xlsx")) +
        list(RAW_DIR.glob("*.csv"))
    )

    if not files:
        raise RuntimeError("No ATP files found")

    print(f"Found {len(files)} files")

    for f in files:
        print(f"Loading {f.name}")
        df = pd.read_excel(f) if f.suffix != ".csv" else pd.read_csv(f)

        df.columns = df.columns.str.strip()

        rename = {
            "Winner": "winner",
            "Loser": "loser",
            "Surface": "surface",
            "Date": "date",
            "Tournament": "tournament",
            "WRank": "winner_rank",
            "LRank": "loser_rank"
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

        year = int("".join(filter(str.isdigit, f.stem))[:4])
        df["season"] = year

        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print(f"Saved {len(out)} rows → {OUT}")

if __name__ == "__main__":
    main()
