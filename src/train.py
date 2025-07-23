# File: src/train.py

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import LEVIRCDDataset as ChangeDetectionDataset
from model import UNet

def train(
    data_dir: str,
    epochs: int = 20,
    batch_size: int = 4,
    lr: float = 1e-3,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
):
    # Dataset and DataLoader
    train_dataset = ChangeDetectionDataset(root_dir=data_dir, split='train')
    val_dataset = ChangeDetectionDataset(root_dir=data_dir, split='val')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1)

    # Model
    model = UNet(in_channels=6, out_channels=1).to(device)

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        loop = tqdm(train_loader, desc=f'Epoch [{epoch}/{epochs}]')
        for img1, img2, mask in loop:
            inputs = torch.cat([img1, img2], dim=1).to(device)  # shape: [B, 6, H, W]
            mask = mask.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, mask)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        print(f'Validation at Epoch {epoch}')
        validate(model, val_loader, device)

        # Save checkpoint
        torch.save(model.state_dict(), f'model_epoch_{epoch}.pth')

def validate(model, val_loader, device):
    model.eval()
    criterion = nn.BCELoss()
    val_loss = 0.0

    with torch.no_grad():
        for img1, img2, mask in val_loader:
            inputs = torch.cat([img1, img2], dim=1).to(device)
            mask = mask.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, mask)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f'Validation Loss: {avg_val_loss:.4f}')
