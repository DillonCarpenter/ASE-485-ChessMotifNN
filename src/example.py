import torch
import torch.nn as nn
import torch.optim as optim

# ----------------------------
# Hyperparameters
# ----------------------------
input_size = 20    # length of each input tensor
hidden1 = 10       # size of first hidden layer
hidden2 = 5        # size of second hidden layer
output_size = 1    # regression output
batch_size = 4
learning_rate = 0.01
epochs = 5         # number of training iterations

# ----------------------------
# Sample Data (batch of 1D tensors)
# ----------------------------
# Normally you'd load real data here
inputs = torch.randn(batch_size, input_size)
targets = torch.randn(batch_size, output_size)

# ----------------------------
# Define the Network
# ----------------------------
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden2, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# Instantiate the model
model = SimpleNet()

# ----------------------------
# Loss function and optimizer
# ----------------------------
criterion = nn.MSELoss()           # mean squared error for regression
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

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
with torch.no_grad():  # no need to track gradients
    test_input = torch.randn(1, input_size)
    test_output = model(test_input)
    print(f"\nTest input: {test_input}")
    print(f"Network output: {test_output}")
