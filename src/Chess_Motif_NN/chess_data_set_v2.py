import torch
from torch.utils.data import Dataset
import chess
import numpy as np
import pandas as pd
theme_to_index = {
    "advancedPawn": 0,
    "advantage": 1,
    "anastasiaMate": 2,
    "arabianMate": 3,
    "attackingF2F7": 4,
    "attraction": 5,
    "backRankMate": 6,
    "bishopEndgame": 7,
    "bodenMate": 8,
    "capturingDefender": 9,
    "castling": 10,
    "clearance": 11,
    "crushing": 12,
    "defensiveMove": 13,
    "deflection": 14,
    "discoveredAttack": 15,
    "doubleBishopMate": 16,
    "doubleCheck": 17,
    "dovetailMate": 18,
    "enPassant": 19,
    "endgame": 20,
    "equality": 21,
    "exposedKing": 22,
    "fork": 23,
    "hangingPiece": 24,
    "hookMate": 25,
    "interference": 26,
    "intermezzo": 27,
    "kingsideAttack": 28,
    "knightEndgame": 29,
    "long": 30,
    "master": 31,
    "masterVsMaster": 32,
    "mate": 33,
    "mateIn1": 34,
    "mateIn2": 35,
    "mateIn3": 36,
    "mateIn4": 37,
    "mateIn5": 38,
    "middlegame": 39,
    "oneMove": 40,
    "opening": 41,
    "pawnEndgame": 42,
    "pin": 43,
    "promotion": 44,
    "queenEndgame": 45,
    "queenRookEndgame": 46,
    "queensideAttack": 47,
    "quietMove": 48,
    "rookEndgame": 49,
    "sacrifice": 50,
    "short": 51,
    "skewer": 52,
    "smotheredMate": 53,
    "superGM": 54,
    "trappedPiece": 55,
    "underPromotion": 56,
    "veryLong": 57,
    "xRayAttack": 58,
    "zugzwang": 59,
}

def fen_to_tensor(fen: str) -> torch.Tensor:
    """
    Convert a FEN string into a 19x8x8 tensor.
    Channels: 6 piece types * 2 colors (white, black) + 1 channel for side to move + 4 channels for castling rights + 
    1 channel for en passant target square + 1 channel for halfmove clock
    """
    piece_map = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,  # White
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11  # Black
    }
    board = chess.Board(fen)
    tensor = np.zeros((19, 8, 8), dtype=np.float32)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            channel = piece_map[piece.symbol()]
            row = 7 - chess.square_rank(square)  # convert rank to row
            col = chess.square_file(square)
            tensor[channel, row, col] = 1.0
    # side to move channel (12)
    tensor[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
    #castling rights channels (13-16)
    tensor[13, :, :] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    tensor[14, :, :] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    tensor[15, :, :] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    tensor[16, :, :] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
    # en passant target square channel (17)
    tensor[17, :, :] = 0.0
    if board.ep_square is not None:
        row = chess.square_rank(board.ep_square)
        col = chess.square_file(board.ep_square)
        tensor[17, row, col] = 1.0
    # halfmove clock channel (18)
    tensor[18, :, :] = board.halfmove_clock / 100.0 # normalize to [0,1]. Max is 100 due to half move and 50 move rule
    return torch.from_numpy(tensor)
def labels_to_tensor(labels: str) -> torch.Tensor: #Create a multi-hot vector for the labels
    indices = [theme_to_index[label] for label in labels.split() if label in theme_to_index]
    tensor = torch.zeros(len(theme_to_index), dtype=torch.float32)
    tensor[indices] = 1.0
    return tensor


class ChessDataset(Dataset):
    def __init__(self, csv_path):
        
        
        # Load only what we need
        df = pd.read_csv(csv_path, usecols=["FEN", "Themes"])

        # Convert labels once (vectorized = fast)
        labels = df["Themes"]
        self.fens = df["FEN"].tolist()
        self.labels = labels.tolist()

    def __len__(self):
        return len(self.fens)

    def __getitem__(self, idx):
        fen = self.fens[idx]
        labels = self.labels[idx]

        tensor = fen_to_tensor(fen)

        target = labels_to_tensor(labels)

        return tensor, target