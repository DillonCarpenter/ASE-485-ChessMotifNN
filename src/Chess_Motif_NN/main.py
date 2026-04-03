import torch
import chess
import chess.engine
from MotifDetector import MotifNet
import numpy as np
import pandas as pd

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
def decode_predictions(outputs, labels, threshold=0.5):
    probs = torch.sigmoid(outputs)
    preds = probs > threshold
    
    indices = preds.nonzero(as_tuple=True)[1]
    
    results = [
        (labels[i], probs[0, i].item())
        for i in indices
    ]
    
    return sorted(results, key=lambda x: x[1], reverse=True)
def fen_to_tensor(fen: str, moves: list[str]) -> torch.Tensor:
    """
    Convert a FEN string into a 9x8x8 tensor.
    Channels: 6 piece types * 2 colors (white, black) + 1 channel for side to move + 4 channels for castling rights + 
    1 channel for en passant target square + 1 channel for halfmove clock
    10 channels for solution sequence (from and to squares for 5 moves). We're stopping at 5 moves deep after move[0] 
    since according to the distribution over 90% of puzzles have a solution length of 5 or less.
    """

    """
    This function is basically the same from the ChessDataset class, but we don't apply move[0] to the board since the FEN 
    string in the dataset already represents the position the user inputs.
    """
    piece_map = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,  # White
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11  # Black
    }
    board = chess.Board(fen)
    """
    Because in the lichess dataset, the FEN string only contains the position before the move is made, 
    we need to apply the move to get the correct position for motif detection. 
    This is because the puzzle doesn't actually start until move[0] is applied.
    """
    
    tensor = np.zeros((29, 8, 8), dtype=np.float32)
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
        row = 7 - chess.square_rank(board.ep_square)
        col = chess.square_file(board.ep_square)
        tensor[17, row, col] = 1.0
    # halfmove clock channel (18)
    tensor[18, :, :] = board.halfmove_clock / 100.0 # normalize to [0,1]. Max is 100 due to half move and 50 move rule
    solution_moves = moves[0:5] # take only the first 5 moves in the solution sequence
    for i, move in enumerate(solution_moves):
        uci = chess.Move.from_uci(move)
        from_row = 7 - chess.square_rank(uci.from_square)
        from_col = chess.square_file(uci.from_square)
        to_row = 7 - chess.square_rank(uci.to_square)
        to_col = chess.square_file(uci.to_square)
        tensor[19 + i*2,     from_row, from_col] = 1.0  # from plane
        tensor[19 + i*2 + 1, to_row,   to_col]   = 1.0  # to plane
    return torch.from_numpy(tensor)

def menu():
    print("=" * 50)
    print("        Welcome to the Chess Motif Network        ")
    print("=" * 50)
    print("Type a FEN string to test the neural network.")
    print("Or enter -1 to exit the program.")
    print("=" * 50)

    choice = input("Your choice: ").strip()

    if not choice:
        print("You didn't enter anything. Please try again.")
        return menu()

    return choice

def main():
    settings = {
        "MultiPV": 1,
        "Depth": 10
    }
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MotifNet(num_blocks=2)
    model.to(device)
    model.load_state_dict(torch.load("models/overnight_best_model.pt", map_location=device))
    model.eval()
    engine = chess.engine.SimpleEngine.popen_uci("stockfish")
    while True:
        choice = menu()
        #Handle choice
        if choice == "-1":
            print("Exiting...")
            engine.quit()
            return
        else:
            print("Processing FEN...")
            try:
                board = chess.Board(choice)
                info = engine.analyse(board, chess.engine.Limit(depth=settings["Depth"]), multipv=settings["MultiPV"])
                predictions = []
                for i in info:
                   pv = i["pv"] #List of move objects. For simplicity, convert them to UCI strings.
                   pv = [move.uci() for move in pv]
                   with torch.no_grad(): #No gradient needed
                       #Take only the first 5 moves from the pv. This will be changed to sliding window
                       input_tensor = fen_to_tensor(choice, pv[0:5]).unsqueeze(0).to(device) #Basically add batch size of 1 to tensor as that's expected by the model. Also move to device.
                       output = model(input_tensor)
                       predictions = decode_predictions(output, LABELS, threshold=0.5)
                       print(predictions)
            except (ValueError) as e:
                print(f"Error processing FEN: {e}")
                continue
                    

if __name__ == "__main__":
    print("Starting Chess Motif Neural Network...")
    main()