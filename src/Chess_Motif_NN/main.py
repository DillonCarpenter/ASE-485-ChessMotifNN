import torch
import torch.nn as nn
from chess_data_set import fen_to_tensor
from MateDetector import MateNet


def menu():
    print("Welcome to the Chess Motif Neural Network!")
    choice = input("Enter FEN to test neural network or -1 to exit:")
    return choice

def main():
    choice = menu()
    #Handle choice
    if choice == "-1":
        print("Exiting...")
        return
    else:
        # Load the trained model
        model = MateNet()
        model.load_state_dict(torch.load("models/mate_detector(advancedPawn).pt", map_location=torch.device('cpu')))
        model.eval()
        print(f"Testing FEN: {choice}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tensor = fen_to_tensor(choice).unsqueeze(0).to(device)  # Add batch dimension and move to device
        with torch.no_grad():
            output = model(tensor)
            prob = torch.sigmoid(output).item()
            print(f"Predicted probability of motif: {prob:.4f}")


print("Starting Chess Motif Neural Network...")
main()