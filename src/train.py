# src/train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.data.dataset import OCTDataset
from src.models.resnet_model import ResNetOCT
from tqdm import tqdm

# ---------------- CONFIG ----------------
DATA_ROOT = "data/raw/OCT2017"
CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]
BATCH_SIZE = 16
EPOCHS = 5
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- DATA ----------------
train_dataset = OCTDataset(
    root_dir=f"{DATA_ROOT}/train",
    classes=CLASSES,
    augment=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

# ---------------- MODEL ----------------
model = ResNetOCT(num_classes=len(CLASSES)).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ---------------- TRAIN ----------------
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

# ---------------- SAVE ----------------
torch.save(model.state_dict(), "experiments/resnet_oct.pth")
print("Model saved to experiments/resnet_oct.pth")
