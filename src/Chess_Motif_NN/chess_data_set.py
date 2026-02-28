import torch
from torch.utils.data import Dataset
import pandas as pd
import chess
import numpy as np
import random

def fen_to_tensor(fen: str) -> torch.Tensor:
    """
    Convert a FEN string into a 12x8x8 tensor.
    Channels: 6 piece types * 2 colors (white, black)
    """
    piece_map = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,  # White
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11  # Black
    }
    board = chess.Board(fen)
    tensor = np.zeros((13, 8, 8), dtype=np.float32) #13 channels for 12 piece types and Side to move

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            channel = piece_map[piece.symbol()]
            row = 7 - chess.square_rank(square)  # convert rank to row
            col = chess.square_file(square)
            tensor[channel, row, col] = 1.0
    # side to move channel (12)
    tensor[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0

    return torch.from_numpy(tensor)

class ChessDataset(Dataset):
    def __init__(self, csv_path, target_theme="mateIn1", row_limit=10000):
        
        self.target_theme = target_theme
        
        # Load only what we need
        df = self.reservoir_sample(csv_path, row_limit)

        # Convert labels once (vectorized = fast)
        labels = df["Themes"].str.contains(
            fr"\b{target_theme}\b",
            regex=True,
            na=False
        ).astype(float)

        self.fens = df["FEN"].tolist()
        self.labels = labels.tolist()

        positives = int(sum(self.labels))
        negatives = len(self.labels) - positives
        self.positives = positives
        self.negatives = negatives

        print(f"""
            Dataset Loaded:
            Rows: {len(self.labels)}
            Positives ({target_theme}): {positives}
            Negatives: {negatives}
        """)

    def __len__(self):
        return len(self.fens)

    def __getitem__(self, idx):
        fen = self.fens[idx]
        label = self.labels[idx]

        tensor = fen_to_tensor(fen)

        target = torch.tensor([label], dtype=torch.float32)

        return tensor, target
    
    def reservoir_sample(self, csv_path, sample_size):
        reservoir = []

        for chunk in pd.read_csv(csv_path, usecols=["FEN", "Themes"], chunksize=50000):
            for _, row in chunk.iterrows():
                if len(reservoir) < sample_size:
                    reservoir.append(row)
                else:
                    # Reservoir replacement rule
                    i = random.randint(0, len(reservoir) - 1)
                    if random.random() < (sample_size / (len(reservoir) + 1)):
                        reservoir[i] = row

        return pd.DataFrame(reservoir)
    
