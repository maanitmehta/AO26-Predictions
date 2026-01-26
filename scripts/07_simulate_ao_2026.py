import pandas as pd
import numpy as np
from collections import Counter
from scripts.predict_match import predict_match
from scripts.name_utils import canonical_name

DRAW_PATH = "data/processed/ao_2026_mens_draw.csv"
N_SIM = 10000

ROUNDS = ["R128", "R64", "R32", "R16", "QF", "SF", "F"]

def simulate_round(players, round_name, verbose=False):
    winners = []
    for i in range(0, len(players), 2):
        A = players[i]
        B = players[i + 1]
        p = predict_match(A, B)
        winner = A if np.random.rand() < p else B
        winners.append(winner)

        if verbose:
            print(f"{round_name}: {A} vs {B} → {winner} (p={p:.2f})")

    return winners

def simulate_tournament(draw, verbose=False):
    players = []
    for _, row in draw.iterrows():
        players.append(row["player_A"])
        players.append(row["player_B"])

    for r in ROUNDS:
        players = simulate_round(players, r, verbose)
        if verbose:
            print(f"\nWinners after {r}: {players}\n")

    return players[0]

def main():
    draw = pd.read_csv(DRAW_PATH, engine="python")
    draw["player_A"] = draw["player_A"].apply(canonical_name)
    draw["player_B"] = draw["player_B"].apply(canonical_name)

    print("\n🔍 DEBUG: ONE FULL TOURNAMENT RUN\n")
    champ = simulate_tournament(draw, verbose=True)
    print(f"\n🏆 Champion (single run): {champ}\n")

    # Monte Carlo
    counts = Counter()
    for _ in range(N_SIM):
        champ = simulate_tournament(draw, verbose=False)
        counts[champ] += 1

    results = (
        pd.DataFrame.from_dict(counts, orient="index", columns=["wins"])
        .assign(prob=lambda df: df["wins"] / N_SIM)
        .sort_values("prob", ascending=False)
    )

    print("\n🏆 AUSTRALIAN OPEN 2026 — TITLE PROBABILITIES\n")
    print(results.head(15))

    results.to_csv("results/ao_2026_title_probabilities.csv")

if __name__ == "__main__":
    main()
