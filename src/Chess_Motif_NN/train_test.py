import torch
import torch.nn as nn
import torch.optim as optim

# Simple dataset: y = 2x + 1
x = torch.tensor([[1.0],[2.0],[3.0],[4.0]])
y = torch.tensor([[3.0],[5.0],[7.0],[9.0]])

# Simple linear model
model = nn.Linear(1,1)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
x, y = x.to(device), y.to(device)

# Training loop
for epoch in range(200):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()

print('Trained model parameters:', model.weight.item(), model.bias.item())

# Save model
torch.save(model.state_dict(), 'models/simple_model.pt')
