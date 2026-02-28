import torch
import torch.nn as nn
from chess_data_set import fen_to_tensor
from MateDetector import MateNet


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
    while True:
        choice = menu()
        #Handle choice
        if choice == "-1":
            print("Exiting...")
            return
        else:
            # Load the trained model
            model = MateNet()
            model.load_state_dict(torch.load("../../models/mate_detector(advancedPawn).pt", map_location=torch.device('cpu')))
            model.eval()
            print(f"Testing FEN: {choice}")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            tensor = fen_to_tensor(choice).unsqueeze(0).to(device)  # Add batch dimension and move to device
            with torch.no_grad():
                output = model(tensor)
                prob = torch.sigmoid(output).item()
                print(f"Predicted probability of Advanced Pawn motif: {prob:.4f}")

if __name__ == "__main__":
    print("Starting Chess Motif Neural Network...")
    main()