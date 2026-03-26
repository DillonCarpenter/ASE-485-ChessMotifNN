import pandas as pd
import numpy as np

CSV_PATH = "data/lichess_puzzle_transformed.csv"

print(f"Loading {CSV_PATH}...")
df = pd.read_csv(CSV_PATH, usecols=["Moves"])

# Total moves column includes move[0] (opponent move) + solution moves
# Solution length = total moves - 1 (since move[0] is not part of the solution)
df["total_moves"] = df["Moves"].str.split().str.len()
df["solution_length"] = df["total_moves"] - 1

sol = df["solution_length"]

print(f"\nTotal puzzles: {len(df):,}")
print(f"\nSolution length stats:")
print(f"  Min:    {sol.min()}")
print(f"  Max:    {sol.max()}")
print(f"  Mean:   {sol.mean():.2f}")
print(f"  Median: {sol.median():.0f}")
print(f"  Std:    {sol.std():.2f}")

print(f"\nDistribution:")
counts = sol.value_counts().sort_index()
cumulative = 0
for length, count in counts.items():
    cumulative += count
    pct = count / len(df) * 100
    cum_pct = cumulative / len(df) * 100
    bar = "#" * int(pct / 0.5)
    print(f"  {length:>3} moves: {count:>7,} ({pct:5.1f}%) cum: {cum_pct:5.1f}%  {bar}")

print(f"\nCoverage by cap:")
for cap in [1, 2, 3, 4, 5, 6, 7, 8, 10]:
    pct = (sol <= cap).sum() / len(df) * 100
    print(f"  Cap at {cap}: covers {pct:.1f}% of puzzles")