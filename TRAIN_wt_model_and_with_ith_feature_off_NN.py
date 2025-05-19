import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import mean_squared_error
import numpy as np
import os
import random

# -----------------------------
# Config (Global Variables)
# -----------------------------
CSV_PATH = 'data/train/triplet_answers_corrected_balanced.csv'  # replace with your path
BATCH_SIZE = 4
LR = 1e-3
EPOCHS = 50
VAL_SPLIT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = 'models/NN'
CHECKPOINT_FEATURE_DIR = 'models/NN_features'
INPUT_DIM = 10
SEED = 42
PATIENCE = 5

# -----------------------------
# Set Seed
# -----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# -----------------------------
# Custom Dataset
# -----------------------------
class CustomDataset(Dataset):
    def __init__(self, csv_path, feature_off_idx=None):
        data = pd.read_csv(csv_path)
        X = data.iloc[:, :-1].values
        if feature_off_idx is not None:
            X[:, feature_off_idx] = 0  # turn off the feature
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(data.iloc[:, -1].values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# -----------------------------
# Neural Network
# -----------------------------
class RegressionNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

# -----------------------------
# Training Loop
# -----------------------------
def train_model(feature_off_idx=None, max_epochs=EPOCHS, checkpoint_path='best_model.pt', early_stopping=True):
    dataset = CustomDataset(CSV_PATH, feature_off_idx)
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    model = RegressionNN(input_dim=INPUT_DIM).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(DEVICE), y_val.to(DEVICE)
                pred = model(X_val)
                val_preds.extend(pred.cpu().numpy())
                val_targets.extend(y_val.cpu().numpy())

        val_loss = mean_squared_error(val_targets, val_preds)
        print(f"Epoch {epoch+1}/{max_epochs}, Train Loss: {avg_train_loss:.4f}, Val MSE: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), checkpoint_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early Stopping
        if early_stopping and epochs_no_improve >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    return best_epoch

# -----------------------------
# Main Run
# -----------------------------
if __name__ == '__main__':
    set_seed(SEED)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_FEATURE_DIR, exist_ok=True)

    # Step 1: Train on full features
    print("\nTraining on full features...")
    ground_truth_path = os.path.join(CHECKPOINT_DIR, 'best_ground_truth_model_deepseek.pt')
    best_epoch = train_model(feature_off_idx=None, max_epochs=EPOCHS, checkpoint_path=ground_truth_path, early_stopping=True)
    print(f"Best epoch on full features: {best_epoch}")

    # Step 2: Train with each feature off
    for i in range(INPUT_DIM):
        print(f"\nTraining with feature {i} turned off...")
        feature_checkpoint_path = os.path.join(CHECKPOINT_FEATURE_DIR, f'{i+1}_model_deepseek.pt')
        train_model(feature_off_idx=i, max_epochs=best_epoch, checkpoint_path=feature_checkpoint_path, early_stopping=False)