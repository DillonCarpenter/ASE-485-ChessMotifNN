import torch
import torch.nn as nn
import torch.optim as optim
from chess_data_set import ChessDataset
from torch.utils.data import DataLoader

# ----------------------------
# Hyperparameters
# ----------------------------
channels = 12      # 6 piece types * 2 colors
board_size = 8     # 8x8 chessboard
batch_size = 4
learning_rate = 0.0025
epochs = 20         # number of training iterations

# ----------------------------
# Sample Data (simulate FEN → tensor)
# ----------------------------
dataset = ChessDataset(csv_path = "data/lichess_puzzle_transformed.csv", row_limit=10_000)

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2  # increase if CPU allows
)

# ----------------------------
# Define the CNN Network
# ----------------------------
class ChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(64 * board_size * board_size, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

# Instantiate the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChessNet().to(device)

# ----------------------------
# Loss function and optimizer
# ----------------------------
pos_weight = torch.tensor([9181 / 819], device=device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ----------------------------
# Training loop
# ----------------------------
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(dataloader):.4f}")
# ----------------------------
# Save the trained model
# ----------------------------
torch.save(model.state_dict(), "models/basic_chess_model.pt")
print("Model saved to chess_model.pth")

# ----------------------------
# Evaluation (simple accuracy)
# ----------------------------
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()

        correct += (preds == targets).sum().item()
        total += targets.numel()

accuracy = correct / total
print(f"Accuracy: {accuracy * 100:.2f}%")

# How many positives does it check?
model.eval()
pred_pos = 0
total = 0

with torch.no_grad():
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()

        pred_pos += preds.sum().item()
        total += preds.numel()

print(f"Predicted positives: {pred_pos} / {total}")