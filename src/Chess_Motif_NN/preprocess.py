import chess
import numpy as np
import pandas as pd


PIECE_MAP = {
    "P": 0,
    "N": 1,
    "B": 2,
    "R": 3,
    "Q": 4,
    "K": 5,
    "p": 6,
    "n": 7,
    "b": 8,
    "r": 9,
    "q": 10,
    "k": 11,
}

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


def square_to_rc(square: int) -> tuple[int, int]:
    return (
        7 - chess.square_rank(square),
        chess.square_file(square),
    )
def labels_to_tensor(labels: str) -> np.ndarray:
    indices = [theme_to_index[label] for label in labels.split() if label in theme_to_index]
    tensor = np.zeros(len(theme_to_index), dtype=np.uint8)
    tensor[indices] = 1
    return tensor

def fen_to_29_channels(fen: str, moves: list[str]) -> np.ndarray:
    """
    Converts a puzzle into a compact uint8 tensor.

    Channels:
        0-11  Piece planes
        12    Side to move
        13-16 Castling rights
        17    En passant square
        18    Halfmove clock (0-100)
        19-28 Solution move planes (from/to for next 5 moves)
    """

    board = chess.Board(fen)

    # Puzzle starts after the first move
    board.push_uci(moves[0])

    tensor = np.zeros((29, 8, 8), dtype=np.uint8)

    # Piece planes
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue

        channel = PIECE_MAP[piece.symbol()]
        row, col = square_to_rc(square)
        tensor[channel, row, col] = 1

    # Side to move
    tensor[12].fill(board.turn)

    # Castling rights
    tensor[13].fill(board.has_kingside_castling_rights(chess.WHITE))
    tensor[14].fill(board.has_queenside_castling_rights(chess.WHITE))
    tensor[15].fill(board.has_kingside_castling_rights(chess.BLACK))
    tensor[16].fill(board.has_queenside_castling_rights(chess.BLACK))

    # En passant
    if board.ep_square is not None:
        row, col = square_to_rc(board.ep_square)
        tensor[17, row, col] = 1

    # Halfmove clock (store raw value)
    tensor[18].fill(board.halfmove_clock)

    # Remaining solution moves (max 5)
    for i, move in enumerate(moves[1:6]):
        uci = chess.Move.from_uci(move)

        fr, fc = square_to_rc(uci.from_square)
        tr, tc = square_to_rc(uci.to_square)

        tensor[19 + i * 2, fr, fc] = 1
        tensor[20 + i * 2, tr, tc] = 1

    return tensor


def preprocess_csv(csv_path="data/lichess_puzzle_transformed.csv"):
    print("Loading CSV...")

    df = pd.read_csv(csv_path)

    n = len(df)

    print(f"Found {n:,} puzzles.")

    inputs = np.lib.format.open_memmap(
        "data/inputs_packed.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(n, 29, 8, 8),
    )

    targets = np.lib.format.open_memmap(
        "data/targets_packed.npy",
        mode="w+",
        dtype=np.float32,
        shape=(n, 60),
    )

    print("Processing...")

    for i, row in enumerate(df.itertuples(index=False)):
        # Convert string representation into list of moves
        moves =row.Moves.split()

        inputs[i] = fen_to_29_channels(row.FEN, moves)

        # TODO:
        # Replace this with however your 60-dimensional target vector is stored.
        #
        # Example:
        #
        # targets[i] = ast.literal_eval(row.target)
        targets[i] = labels_to_tensor(row.Themes)

        if i % 100000 == 0:
            print(f"{i:,}/{n:,}")

    inputs.flush()
    targets.flush()

    print("Finished preprocessing.")


if __name__ == "__main__":
    preprocess_csv()