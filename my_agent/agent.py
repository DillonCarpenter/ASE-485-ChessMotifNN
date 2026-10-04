from google.adk.agents.llm_agent import Agent
from src.Chess_Motif_NN.main import validate_fen, load_engine, load_model, process_pv, analyze_position
import torch
import chess
from src.Chess_Motif_NN.MotifDetector import MotifNet

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


def detect_motifs(FEN: str)-> dict:
    """
    Analyze a chess position and return the predicted motifs for
    each principal variation.

    Returns:
    {
        "pv1": {
            "moves": ["e2e4", "e7e5", ...],
            "motifs": {
                "fork": 0.91,
                "pin": 0.73,
                ...
            }
        },
        "pv2": {
            "moves": [...],
            "motifs": {...}
        }
    }
    """
    #First, validate FEN
    if not validate_fen(FEN):
        raise ValueError("Invalid FEN string")
    dictionary = dict()
    settings ={
        "Number of Lines": 3,
        "Depth": 10,
        "Minimum Threshold": 0.5
    }
    engine = load_engine()
    device = torch.device("cpu")
    model = load_model(device)
    board = chess.Board(FEN)
    info = analyze_position(engine, board, settings)
    for i in info:
        pv = i["pv"] #List of move objects. For simplicity, convert them to UCI strings.
        #Returns list of tuples where each tuple is (motif, probability)
        predictions = process_pv(FEN, pv, model, device, settings["Minimum Threshold"])
        multipv = i["multipv"]
        dictionary[f"pv{multipv}"] = {
            "moves": [move.uci() for move in pv],
            "motifs": {
                motif: prob for motif, prob in predictions
            }
        }
    return dictionary
    
root_agent = Agent(
    model='gemini-3.5-flash',
    name='root_agent',
    description='A helpful chess assistant for user questions.',
    instruction="""
    You are a chess motif analysis assistant.

    When the user provides a FEN string, call the detect_motifs tool.

    Use the tool output to explain the principal variations and the
    predicted chess motifs.

    The tool's Stockfish analysis is authoritative for the engine lines.
    Do not invent or speculate about engine reasoning that is not supported
    by the tool output.

    It is completely acceptable to say that Stockfish chose a move because
    it is the engine's strongest line without providing a deeper human-readable
    justification.

    Do not invent motifs, probabilities, moves, or strategic explanations.
    If the tool does not provide enough information to answer a question,
    say so rather than guessing.
    """,
    tools= [detect_motifs]
)
