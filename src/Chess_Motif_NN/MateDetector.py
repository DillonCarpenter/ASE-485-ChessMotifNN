import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from chess_data_set import ChessDataset
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import precision_score, recall_score, f1_score

# ----------------------------
# Hyperparameters
# ----------------------------
channels = 13      # 6 piece types * 2 colors + Side to move
board_size = 8     # 8x8 chessboard
batch_size = 4
learning_rate = 0.0001
epochs = 200         # number of training iterations
threshold = 0.35

def load_data():
    # ----------------------------
    # Sample Data (simulate FEN → tensor)
    # ----------------------------
    dataset = ChessDataset(csv_path = "data/lichess_puzzle_transformed.csv", row_limit=20_000, target_theme="advancedPawn")

    # Define sizes and split
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=128, shuffle=False)
    test_loader  = DataLoader(test_set, batch_size=128, shuffle=False)

    return dataset, train_loader, val_loader, test_loader

# ----------------------------
# Define the CNN Network
# ----------------------------
class MateNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.input = nn.Sequential(
            nn.Conv2d(13, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.resblocks = nn.Sequential(
            *[ResBlock(64) for _ in range(4)]
        )

        self.head = nn.Sequential(
            nn.Conv2d(64, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.input(x)
        x = self.resblocks(x)
        x = self.head(x)
        return x
    
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.dropout(x, p=0.3, training=self.training) #Regularization technique
        return F.relu(x + residual)

if __name__ == "__main__":
    dataset, train_loader, val_loader, test_loader = load_data()
    # Instantiate the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MateNet().to(device)

    # ----------------------------
    # Loss function and optimizer
    # ----------------------------
    pos_weight = torch.tensor([dataset.negatives / dataset.positives], device=device) * .75
    #pos_weight = torch.tensor([4.0], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    best_f1 = 0.0
    best_avg_val_loss = 100000
    patience = 15
    patience_counter = 0
    for epoch in range(epochs):
        # ----------------------------
        # Training
        # ----------------------------
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # ----------------------------
        # Validation
        # ----------------------------
        model.eval()
        val_loss = 0.0
        all_targets = []
        #all_probs = []
        all_preds = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                #loss
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                #metrics
                probs = torch.sigmoid(outputs)
                preds = (probs > threshold).float()
                
                all_targets.append(targets.cpu())
                #all_probs.append(probs.cpu())
                all_preds.append(preds.cpu())
        avg_val_loss = val_loss / len(val_loader)

        all_targets = torch.cat(all_targets)
        all_preds = torch.cat(all_preds)
        #all_probs = torch.cat(all_probs)

        #Looking for best threshold
        """
        f1 = 0.0
        best_t = 0.0
        for t in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
            preds = (all_probs > t).float()
            thresholds_f1 = f1_score(all_targets, preds)
            if thresholds_f1 > f1:
                f1 = thresholds_f1
                best_t = t
        print("best threshold:", best_t, "f1:", f1)
        """
        f1 = f1_score(all_targets, all_preds)
        
        """
        if(f1 > best_f1):
            best_f1 = f1
            patience_counter = 0
            best_model_weights = model.state_dict().copy()
        elif(f1 < best_f1):
            patience_counter += 1
        if(patience_counter >= patience):
            break
        """
        if(avg_val_loss < best_avg_val_loss):
            best_avg_val_loss = avg_val_loss
            patience_counter = 0
            best_model_weights = model.state_dict().copy()
        elif(avg_val_loss > best_avg_val_loss):
            patience_counter += 1
        if(patience_counter > patience):
            break
        print(f"Epoch {epoch+1}/{epochs}, "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, "
            f"F1: {f1:.4f}"
            )
    # ----------------------------
    # Save the trained model
    # ----------------------------
    model.load_state_dict(best_model_weights)
    torch.save(model.state_dict(), "models/mate_detector(advancedPawn).pt")
    print("Model saved to models/mate_detector(advancedPawn).pt")

    # ----------------------------
    # Evaluation (simple accuracy)
    # ----------------------------
    model.eval()


    all_targets = []
    all_preds = []

    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            preds = (probs > threshold).float()

            all_targets.append(targets.cpu())
            all_preds.append(preds.cpu())

    all_targets = torch.cat(all_targets)
    all_preds = torch.cat(all_preds)

    precision = precision_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds)

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    """
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
    """