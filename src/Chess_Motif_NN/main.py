import torch
import chess
import chess.engine
from MotifDetector import MotifNet
import numpy as np
import pandas as pd
from chess.engine import MateGiven

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
def sliding_window(fen: str, moves: list[str], window_size=5) -> list[torch.Tensor]:
    """
    Generate a list of tensors for each position in the solution sequence using a sliding window approach.
    For example, if the solution sequence is [move1, move2, move3, move4, move5], we will generate tensors for:
    - Position after move1 (using moves[0:5])
    - Position after move2 (using moves[1:6])
    - Position after move3 (using moves[2:7])
    - Position after move4 (using moves[3:8])
    - Position after move5 (using moves[4:9])
    This way, we can capture the motifs that may arise at different points in the solution sequence.
    """
    board = chess.Board(fen)
    tensors = []
    current_fen = fen
    if len(moves) < window_size:
        # We can only make one tensor
        return [fen_to_tensor(fen, moves)]
    for i in range(len(moves) - window_size + 1):
        # Classic sliding window
        window_moves = moves[i:i+window_size]
        tensor = fen_to_tensor(current_fen, window_moves)
        tensors.append(tensor)
        #However, now we need to apply the first move in the window to the board to get the correct FEN for the next window
        board.push_uci(moves[i])
        current_fen = board.fen()
    return tensors

def menu():
    print("=" * 60)
    print("        Welcome to the Chess Motif Network        ")
    print("=" * 60)
    print("Type a FEN string to test the neural network.")
    print("For example: r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 3")
    print("Or enter settings to change settings")
    print("Or enter -1 to exit the program.")
    print("=" * 60)

    choice = input("Your choice: ").strip()

    if not choice:
        print("You didn't enter anything. Please try again.")
        return menu()

    return choice
def load_model(device):
    model = MotifNet(num_blocks=2)
    model.to(device)
    model.load_state_dict(torch.load("models/overnight_best_model.pt", map_location=device))
    model.eval()
    return model

def load_engine():
    return chess.engine.SimpleEngine.popen_uci("stockfish")

def analyze_position(engine: chess.engine.SimpleEngine, board: chess.Board, settings: dict):
    return engine.analyse(
        board,
        chess.engine.Limit(depth=settings["Depth"]),
        multipv=settings["Number of Lines"]
    )

def evaluate_pv(model, tensors, device):
    outputs = []
    with torch.no_grad():
        for tensor in tensors:
            tensor = tensor.unsqueeze(0).to(device)
            outputs.append(model(tensor).cpu())

    stacked = torch.stack(outputs, dim=0)
    return torch.max(stacked, dim=0).values

def process_pv(choice, pv, model, device, threshold=0.5):
    pv_moves = [move.uci() for move in pv]
    tensors = sliding_window(choice, pv_moves)

    final = evaluate_pv(model, tensors, device)
    return decode_predictions(final, LABELS, threshold=threshold)

def confidence_label(p):
    if p >= 0.90:
        return "very confident"
    elif p >= 0.75:
        return "confident"
    elif p >= 0.50:
        return "somewhat confident"
    else:
        return "low confidence"

def print_results(multipv: int, pv: list[str], predictions: list[tuple[str, float]], pov_score: chess.engine.PovScore):
    print("\n" + "=" * 60)
    print(f"Evaluation & Top Motifs — PV{multipv}")
    print(f"Line: {' '.join(move.uci() for move in pv)}")
    print("=" * 60)

    score = pov_score.relative
    pov = "White" if pov_score.turn else "Black"

    if score == MateGiven:
        eval_str = f"{pov} has an immediate mate (MateGiven)"

    elif score.is_mate():
        mate = score.mate()
        if mate > 0:
            eval_str = f"{pov} mates in {mate}"
        else:
            eval_str = f"{pov} is getting mated in {abs(mate)}"

    else:
        cp = score.score()
        eval_str = f"{pov} eval: {cp/100:+.2f} cp"

    print(eval_str)

    print("\nTop Motifs:")
    for i, (motif, prob) in enumerate(predictions, start=1):
        print(f"{i:>2}. {motif:<30} {prob:.2f} ({confidence_label(prob)})")

    print("=" * 60)
def validate_fen(fen):
    try:
        chess.Board(fen)
        return True
    except ValueError:
        return False
def change_settings(settings):
    print("\nCurrent settings:")
    for key, value in settings.items():
        print(f"{key}: {value}")
    print("\nEnter the setting you want to change (Number of Lines, Depth, Minimum Threshold) or type 'back' to return to the menu.")
    setting_choice = input("Your choice: ").strip()
    if setting_choice.lower() == "back":
        return
    elif setting_choice in settings:
        new_value = input(f"Enter new value for {setting_choice}: ").strip()
        try:
            if setting_choice in ["Number of Lines", "Depth"]:
                new_value = int(new_value)
            else:
                new_value = float(new_value)
            settings[setting_choice] = new_value
            print(f"{setting_choice} updated to {new_value}.")
        except ValueError:
            print("Invalid value. Please enter a valid number.")
    else:
        print("Invalid setting. Please try again.")

def main():
    settings = {
        "Number of Lines": 3,
        "Depth": 10,
        "Minimum Threshold": 0.5
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    engine = load_engine()
    while True:
        choice = menu()
        #Handle choice
        if choice == "-1":
            print("Exiting...")
            engine.quit()
            return
        elif choice.lower() == "settings":
            change_settings(settings)
        else:
            if not validate_fen(choice):
                print("Invalid FEN string. Please try again.")
                continue
            print(f"Processing FEN {choice} with Stockfish depth {settings['Depth']} and {settings['Number of Lines']} Lines...")
            try:
                board = chess.Board(choice)
                info = analyze_position(engine, board, settings)
                for i in info:
                    pv = i["pv"] #List of move objects. For simplicity, convert them to UCI strings.
                    predictions = process_pv(choice, pv, model, device, settings["Minimum Threshold"])
                    multipv = i["multipv"]
                    print_results(multipv, pv, predictions, i["score"])
                    
            except (ValueError) as e:
                print(f"Error processing FEN: {e}")
                continue
                    

if __name__ == "__main__":
    print("Starting Chess Motif Neural Network...")
    main()