import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import torch
import torch.nn as nn

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH        = "data/chess_motif_sample_with_moves.csv"
MODEL_PATH      = "models/motif_detector_29_channels_2_residual_blocks.pt"   # your saved checkpoint
SAMPLE_SIZE     = 10_000
THRESHOLD       = 0.62
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABELS = [
    "advancedPawn", "advantage", "anastasiaMate", "arabianMate", "attackingF2F7",
    "attraction", "backRankMate", "bishopEndgame", "bodenMate", "capturingDefender",
    "castling", "clearance", "crushing", "defensiveMove", "deflection",
    "discoveredAttack", "doubleBishopMate", "doubleCheck", "dovetailMate", "enPassant",
    "endgame", "equality", "exposedKing", "fork", "hangingPiece",
    "hookMate", "interference", "intermezzo", "kingsideAttack", "knightEndgame",
    "long", "master", "masterVsMaster", "mate", "mateIn1",
    "mateIn2", "mateIn3", "mateIn4", "mateIn5", "middlegame",
    "oneMove", "opening", "pawnEndgame", "pin", "promotion",
    "queenEndgame", "queenRookEndgame", "queensideAttack", "quietMove", "rookEndgame",
    "sacrifice", "short", "skewer", "smotheredMate", "superGM",
    "trappedPiece", "underPromotion", "veryLong", "xRayAttack", "zugzwang"
]
# ─────────────────────────────────────────────────────────────────────────────

print("Loading CSV...")
df = pd.read_csv(CSV_PATH, usecols=["Themes"])
df = df.dropna(subset=["Themes"])

# Count positives per label across the full dataset
print(f"\nCounting positives across {len(df):,} puzzles...")
label_counts = {label: 0 for label in LABELS}
for themes in df["Themes"]:
    for theme in themes.strip().split():
        if theme in label_counts:
            label_counts[theme] += 1

total = len(df)
print(f"\n{'Label':<30} {'Positives':>10} {'Frequency':>10}")
print("-" * 52)
for label, count in sorted(label_counts.items(), key=lambda x: x[1]):
    freq = count / total * 100
    print(f"{label:<30} {count:>10,} {freq:>9.2f}%")

# Buckets
counts = np.array([label_counts[l] for l in LABELS])
print(f"\nLabel count buckets:")
print(f"  Under 100 positives:   {(counts < 100).sum()} labels")
print(f"  100–500 positives:     {((counts >= 100) & (counts < 500)).sum()} labels")
print(f"  500–2000 positives:    {((counts >= 500) & (counts < 2000)).sum()} labels")
print(f"  Over 2000 positives:   {(counts >= 2000).sum()} labels")

# ── Per-label F1 from saved model (optional, requires model + dataset) ────────
# Comment this section out if you just want the count analysis
print("\nLoading model for per-label F1 analysis...")
try:
    from chess_data_set_v2 import ChessDataset  # replace with your actual import
    from MotifDetector import MotifNet         # replace with your actual import

    dataset = ChessDataset(CSV_PATH)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
    model   = MotifNet(num_blocks=2).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_probs, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(DEVICE)
            logits = model(inputs)
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(targets.numpy())

    all_probs   = np.vstack(all_probs)
    all_targets = np.vstack(all_targets)
    preds       = (all_probs > THRESHOLD).astype(int)

    per_label_f1 = f1_score(all_targets, preds, average=None, zero_division=0)

    print(f"\n{'Label':<30} {'Positives':>10} {'Freq%':>7} {'F1':>8}")
    print("-" * 60)
    for i, label in enumerate(LABELS):
        count = label_counts[label]
        freq  = count / total * 100
        f1    = per_label_f1[i]
        flag  = " ←" if f1 < 0.2 else ""
        print(f"{label:<30} {count:>10,} {freq:>6.2f}% {f1:>8.3f}{flag}")

    print(f"\nMacro F1: {per_label_f1.mean():.4f}")
    print(f"Micro F1: {f1_score(all_targets, preds, average='micro', zero_division=0):.4f}")

    # Correlation between count and F1
    corr = np.corrcoef(counts, per_label_f1)[0, 1]
    print(f"\nCorrelation between positive count and F1: {corr:.3f}")

except ImportError:
    print("Skipping per-label F1 — update the imports at the top of this script.")