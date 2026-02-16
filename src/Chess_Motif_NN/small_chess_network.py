import torch
import torch.nn as nn
import torch.optim as optim

# ----------------------------
# Hyperparameters
# ----------------------------
channels = 12      # 6 piece types * 2 colors
board_size = 8     # 8x8 chessboard
batch_size = 4
learning_rate = 0.001
epochs = 5         # number of training iterations

# ----------------------------
# Sample Data (simulate FEN → tensor)
# ----------------------------
# Normally you'd convert FEN strings to 8x8x12 tensors
inputs = torch.randn(batch_size, channels, board_size, board_size)
targets = torch.randint(0, 2, (batch_size, 1), dtype=torch.float)  # 1 = mate-in-1, 0 = not

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
        self.sigmoid = nn.Sigmoid()  # probability output

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Instantiate the model
model = ChessNet()

# ----------------------------
# Loss function and optimizer
# ----------------------------
criterion = nn.BCELoss()  # binary cross-entropy for mate-in-1
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ----------------------------
# Training loop
# ----------------------------
for epoch in range(epochs):
    # Forward pass
    outputs = model(inputs)
    
    # Compute loss
    loss = criterion(outputs, targets)
    
    # Backward pass
    loss.backward()
    
    # Update weights
    optimizer.step()
    
    # Reset gradients for next step
    optimizer.zero_grad()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# ----------------------------
# Test forward pass (optional)
# ----------------------------
with torch.no_grad():  # no gradients needed
    test_input = torch.randn(1, channels, board_size, board_size)
    test_output = model(test_input)
    print(f"\nTest input shape: {test_input.shape}")
    print(f"Predicted mate-in-1 probability: {test_output.item():.4f}")
