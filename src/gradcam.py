# src/gradcam.py

import torch
import cv2
import numpy as np
from torchvision import transforms
from src.models.resnet_model import ResNetOCT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]

# ---------- Load model ----------
model = ResNetOCT(num_classes=4)
model.load_state_dict(torch.load("experiments/resnet_oct.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ---------- Hook storage ----------
gradients = None
activations = None

def backward_hook(module, grad_in, grad_out):
    global gradients
    gradients = grad_out[0]

def forward_hook(module, input, output):
    global activations
    activations = output

# Register hooks on last conv layer
target_layer = model.model.layer4[-1].conv2
target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)

# ---------- Load OCT image ----------
img_path = "data/raw/OCT2017/test/CNV/CNV-1016042-1.jpeg"
orig = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
orig = cv2.resize(orig, (224, 224))

img = orig / 255.0
img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
img = img.to(DEVICE)

# ---------- Forward ----------
output = model(img)
pred_class = torch.argmax(output).item()

# ---------- Backward ----------
model.zero_grad()
output[0, pred_class].backward()

# ---------- Grad-CAM ----------
weights = gradients.mean(dim=(2, 3), keepdim=True)
cam = (weights * activations).sum(dim=1).squeeze()
cam = cam.detach().cpu().numpy()
cam = np.maximum(cam, 0)
cam = cam / cam.max()
cam = cv2.resize(cam, (224, 224))

# ---------- Overlay ----------
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
overlay = cv2.addWeighted(cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR), 0.6, heatmap, 0.4, 0)

cv2.imshow(f"Grad-CAM ({CLASSES[pred_class]})", overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()
