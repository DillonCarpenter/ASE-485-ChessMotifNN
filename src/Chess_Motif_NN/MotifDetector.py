import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from chess_data_set_v2 import ChessDataset
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import precision_score, recall_score, f1_score


torch.manual_seed(42)
# ----------------------------
# Hyperparameters
# ----------------------------
channels = 19      # 6 piece types * 2 colors + Side to move + castling rights + en passant + halfmove clock
board_size = 8     # 8x8 chessboard
learning_rate = 0.0001
epochs = 200         # number of training iterations
thresholds = np.arange(0.0, 1.0, 0.01) # Thresholds to evaluate for F1 score
thresholds = thresholds.tolist()

def load_data():
    # ----------------------------
    # Sample Data (simulate FEN → tensor)
    # ----------------------------
    dataset = ChessDataset(csv_path = "data/chess_motif_sample_bigger.csv")

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
class MotifNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.input = nn.Sequential(
            nn.Conv2d(channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.resblocks = nn.Sequential(
            *[ResBlock(64) for _ in range(4)]
        )

        self.head = nn.Sequential(
            nn.Conv2d(64, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 60) # 60 possible motifs
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
    model = MotifNet().to(device)

    # ----------------------------
    # Loss function and optimizer
    # ----------------------------
    all_labels = []
    for _, target in dataset:  
        all_labels.append(target)

    y_train = torch.stack(all_labels)  # shape [num_samples, num_classes]
    num_pos = y_train.sum(dim=0)
    num_neg = y_train.shape[0] - num_pos
    pos_weight = num_neg / (num_pos + 1e-5)  # avoid division by zero
    pos_weight = pos_weight.float()
    pos_weight = pos_weight.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,       # halve the lr on plateau
        patience=5,        # wait 5 epochs of no improvement before reducing
        min_lr=1e-6
    )
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
                
                all_targets.append(targets.cpu())
        avg_val_loss = val_loss / len(val_loader)

        all_targets = torch.cat(all_targets)
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
            )
    # ----------------------------
    # Save the trained model
    # ----------------------------
    model.load_state_dict(best_model_weights)
    torch.save(model.state_dict(), "models/motif_detector4.pt")
    print("Model saved to models/motif_detector4.pt")

    # ----------------------------
    # Evaluation (simple accuracy)
    # ----------------------------
    model.eval()


    all_targets = []
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            probs = torch.sigmoid(outputs)

            all_targets.append(targets.cpu())
            all_probs.append(probs.cpu())

    all_targets = torch.cat(all_targets)
    all_targets_np = all_targets.numpy()
    all_probs = torch.cat(all_probs)

    for threshold in thresholds:
        preds = (all_probs > threshold).float().cpu().numpy()

        f1_macro = f1_score(all_targets_np, preds, average='macro')
        f1_micro = f1_score(all_targets_np, preds, average='micro')

        print(f"Macro F1: {f1_macro:.4f} at threshold {threshold:.2f}")
        print(f"Micro F1: {f1_micro:.4f} at threshold {threshold:.2f}")

        per_class_precision = precision_score(all_targets_np, preds, average=None)
        per_class_recall = recall_score(all_targets_np, preds, average=None)
    