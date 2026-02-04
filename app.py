import streamlit as st
import torch
import cv2
import numpy as np
from matplotlib import cm

from src.models.resnet_model import ResNetOCT
from src.gradcam_utils import GradCAM

# ---------------- CONFIG ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]

st.set_page_config(page_title="Retinal Disease Detector", layout="centered")

st.title("👁️ Retinal Disease Detector (OCT)")
st.write("Upload an OCT image to classify retinal disease.")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = ResNetOCT(num_classes=4)
    model.load_state_dict(
        torch.load("experiments/resnet_oct.pth", map_location=DEVICE)
    )
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload OCT Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if image is None:
        st.error("Invalid image file.")
    else:
        st.image(image, caption="Uploaded OCT Image", width=700)

        # -------- Preprocess --------
        img = cv2.resize(image, (224, 224))
        img = img / 255.0
        img = torch.tensor(img, dtype=torch.float32)
        img = img.unsqueeze(0).unsqueeze(0)
        img = img.to(DEVICE)

        # -------- Prediction --------
        with torch.no_grad():
            outputs = model(img)
            probs = torch.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item()

        st.markdown("---")
        st.subheader("🧠 Prediction Result")
        st.write(f"**Disease:** {CLASSES[pred_class]}")
        st.write(f"**Confidence:** {confidence:.2%}")

        # -------- GRAD-CAM --------
        target_layer = model.model.layer4[-1]
        gradcam = GradCAM(model, target_layer)

        cam = gradcam.generate(img, pred_class)

        heatmap = cm.jet(cam)[:, :, :3]
        heatmap = np.uint8(255 * heatmap)

        orig = cv2.resize(image, (224, 224))
        orig = cv2.cvtColor(orig, cv2.COLOR_GRAY2RGB)

        overlay = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)

        st.markdown("---")
        st.subheader("🔍 Model Attention (Grad-CAM)")
        st.image(overlay, width=700)
