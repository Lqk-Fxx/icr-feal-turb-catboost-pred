# -*- coding: utf-8 -*-
"""
Train and save FE NN 5-Seed Ensemble models.
Architecture: 2x128 + BN + Dropout(0.1), same as run_ensemble.py
"""
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
MODEL_DIR = os.path.join(BASE, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEEDS = [42, 123, 456, 789, 1024]

# Load data
DATA_8 = os.path.join(ROOT, '数据', '8_合并的数据.xlsx')
DATA_2 = os.path.join(ROOT, '数据', '2_合并的数据.xlsx')
df8 = pd.read_excel(DATA_8, sheet_name='FE_编码归一化')
df2 = pd.read_excel(DATA_2, sheet_name='FE_编码归一化')
cols = list(df8.columns)
TARGET_IDX = 7
feature_cols = [c for i, c in enumerate(cols) if i != TARGET_IDX]
X_train = df8[feature_cols].values.astype(np.float32)
y_train = df8[cols[TARGET_IDX]].values.astype(np.float32).reshape(-1, 1)

print(f'FE NN Ensemble Training  Device: {DEVICE}')
print(f'Data: {X_train.shape[0]} train rows, {X_train.shape[1]} features')

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, neurons, dropout, use_bn=True):
        super().__init__()
        layers = []
        prev = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(prev, neurons))
            if use_bn:
                layers.append(nn.BatchNorm1d(neurons))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = neurons
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

for seed in SEEDS:
    print(f'  Training seed={seed}...', end='', flush=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLP(17, 2, 128, 0.1, True).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.MSELoss()
    dl = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=16, shuffle=True
    )

    for _ in range(500):
        model.train()
        for bx, by in dl:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(bx), by).backward()
            optimizer.step()

    # Save
    save_path = os.path.join(MODEL_DIR, f'fe_nn_seed{seed}.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'seed': seed,
        'architecture': '2x128+BN+Drop0.1',
        'input_dim': 17,
        'hidden_layers': 2,
        'neurons': 128,
        'dropout': 0.1,
        'use_bn': True
    }, save_path)
    print(f' saved fe_nn_seed{seed}.pt')

print(f'\nDone! {len(SEEDS)} FE NN models saved to {MODEL_DIR}')
