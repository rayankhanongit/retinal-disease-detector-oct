# src/evaluate.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.data.dataset import OCTDataset
from src.models.resnet_model import ResNetOCT
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# ---------------- CONFIG ----------------
DATA_ROOT = "data/raw/OCT2017"
CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- DATA ----------------
test_dataset = OCTDataset(
    root_dir=f"{DATA_ROOT}/test",
    classes=CLASSES,
    augment=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# ---------------- MODEL ----------------
model = ResNetOCT(num_classes=len(CLASSES)).to(DEVICE)
model.load_state_dict(torch.load("experiments/resnet_oct.pth", map_location=DEVICE))
model.eval()

# ---------------- EVALUATION ----------------
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ---------------- METRICS ----------------
print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds, target_names=CLASSES))

print("\nConfusion Matrix:\n")
print(confusion_matrix(all_labels, all_preds))
