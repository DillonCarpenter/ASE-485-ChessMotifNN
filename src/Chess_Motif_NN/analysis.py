import torch
from MotifDetector import MotifNet, ResBlock, load_data
from sklearn.metrics import f1_score, roc_auc_score
import pandas as pd
import numpy as np
# other imports

# same seed, same split
torch.manual_seed(42)

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH        = "data/lichess_puzzle_transformed.csv"
MODEL_PATH      = "models/motif_detector_29_channels_2_residual_blocks.pt"   # your saved checkpoint
THRESHOLD       = 0.12
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

if __name__ == "__main__":
    print("Creating loaders")
    dataset, train_loader, val_loader, test_loader = load_data()
    label_counts = {label: 0 for label in LABELS}
    print("Loading CSV...")
    df = pd.read_csv(CSV_PATH, usecols=["Themes"])
    df = df.dropna(subset=["Themes"])
    for themes in df["Themes"]:
        for theme in themes.strip().split():
            if theme in label_counts:
                label_counts[theme] += 1
    counts = np.array([label_counts[l] for l in LABELS])
    total = len(df)
    print()
    model = MotifNet(num_blocks=2).to(DEVICE)
    model.load_state_dict(torch.load("models/overnight_best_model.pt", map_location=DEVICE))
    model.eval()
    print("Evaluating on test set...")
    all_probs, all_targets = [], []
    all_probs, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
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
    best_thresholds = []
    thresholds = np.linspace(0.0, 1.0, 101)
    for i, label in enumerate(LABELS):
        best_f1 = 0
        best_t = 0.5
        for t in thresholds:
            preds = (all_probs[:, i] > t)
            f1 = f1_score(all_targets[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        entry = {"Label": label, "Best Threshold": best_t, "F1 at Best Threshold": best_f1}
        best_thresholds.append(entry)
    best_thresholds_df = pd.DataFrame(best_thresholds)
    best_thresholds_df.sort_values(
        by="F1 at Best Threshold", ascending=False, inplace=True
    )
    # Correlation between count and F1
    corr = np.corrcoef(counts, per_label_f1)[0, 1]
    print(f"\nCorrelation between positive count and F1: {corr:.3f}")

    print("\nBest Thresholds per Label:")
    print(best_thresholds_df)

    optimized_macro_f1_score = best_thresholds_df["F1 at Best Threshold"].sum()/len(LABELS)
    print(f"\nOptimized Macro F1 Score (using best thresholds): {optimized_macro_f1_score:.4f}")
