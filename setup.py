import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# 1. THE SIAMESE (TRIPLET) NETWORK
# ---------------------------------------------------------
class EmbeddingNet(nn.Module):
    def __init__(self, input_dim, output_dim=32):
        super(EmbeddingNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, output_dim) 
            # No activation at the end - we want raw coordinates in 32D space
        )

    def forward(self, x):
        return self.net(x)

# ---------------------------------------------------------
# 2. CUSTOM DATASET FOR TRIPLET MINING
# ---------------------------------------------------------
class TripletDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
        # Separate indices by class
        self.index_1 = np.where(y == 1)[0]
        self.index_0 = np.where(y == 0)[0]
        
        # We define length by the minority class (Class 1)
        # to ensure every Class 1 is seen as an 'Anchor'
        self.length = len(self.index_1)

    def __getitem__(self, index):
        # 1. ANCHOR: Always a Class 1 sample
        anchor_idx = self.index_1[index]
        
        # 2. POSITIVE: Another Random Class 1 sample
        # (different from anchor if possible)
        pos_idx = np.random.choice(self.index_1)
        while pos_idx == anchor_idx and len(self.index_1) > 1:
            pos_idx = np.random.choice(self.index_1)
            
        # 3. NEGATIVE: A Random Class 0 sample
        # (Improvement: In a more advanced version, we would pick 'Hard' negatives here)
        neg_idx = np.random.choice(self.index_0)

        return (self.X[anchor_idx], self.X[pos_idx], self.X[neg_idx])

    def __len__(self):
        return self.length

# ---------------------------------------------------------
# 3. TRIPLET LOSS FUNCTION
# ---------------------------------------------------------
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Euclidean Distance
        dist_pos = (anchor - positive).pow(2).sum(1)
        dist_neg = (anchor - negative).pow(2).sum(1)
        
        # Loss = max(0, dist_pos - dist_neg + margin)
        losses = torch.relu(dist_pos - dist_neg + self.margin)
        return losses.mean()
