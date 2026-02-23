import json
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import timm
from torchvision import transforms

import streamlit as st

# Page config 
st.set_page_config(
    page_title="Eye Disease Detection",
    layout="centered"
)

# Model configs per disease
DISEASE_CONFIG = {
    "Glaucoma": {
        "model_path":  "glaucoma_model.pth",
        "config_path": "g_model_config.json",
        "class_names": ["Normal", "Glaucoma"],
        "description": "Detects glaucoma from retinal fundus images by analyzing the optic nerve region."
    },
    "Cataract": {
        "model_path":  "cataract_model.pth",
        "config_path": "c_model_config.json",
        "class_names": ["Normal", "Cataract"],
        "description": "Detects cataract from eye images by analyzing lens clarity."
    }
}

IMG_SIZE = 224
MEAN     = [0.485, 0.456, 0.406]
STD      = [0.229, 0.224, 0.225]
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Shared model architecture for both diseases (EfficientNet-B3 backbone with custom classifier head)
class EyeDiseaseClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout=0.4):
        super().__init__()
        self.backbone   = timm.create_model("efficientnet_b3", pretrained=False, num_classes=0)
        in_features     = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.backbone(x))


# Load model (cached per disease)
@st.cache_resource
def load_model(disease: str):
    cfg        = DISEASE_CONFIG[disease]
    model_path = cfg["model_path"]

    if not Path(model_path).exists():
        return None, cfg["class_names"]

    file_config = {}
    if Path(cfg["config_path"]).exists():
        with open(cfg["config_path"]) as f:
            file_config = json.load(f)

    num_classes = file_config.get("num_classes", 2)
    class_names = file_config.get("class_names", cfg["class_names"])

    m = EyeDiseaseClassifier(num_classes=num_classes)
    m.load_state_dict(torch.load(model_path, map_location=DEVICE))
    m.eval()
    m.to(DEVICE)
    return m, class_names


# Inference 
def predict(image: Image.Image, model, class_names: list, threshold: float) -> dict:
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()

    disease_prob = float(probs[1])
    normal_prob  = float(probs[0])
    predicted    = class_names[1] if disease_prob >= threshold else class_names[0]
    confidence   = disease_prob if predicted == class_names[1] else normal_prob

    return {
        "predicted":    predicted,
        "disease_prob": disease_prob,
        "normal_prob":  normal_prob,
        "confidence":   confidence,
        "class_names":  class_names
    }


# Grad-CAM Implementation (for visual explanations)
class GradCAM:
    def __init__(self, model, target_layer):
        self.model       = model
        self.gradients   = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, tensor, class_idx=None):
        self.model.eval()
        output = self.model(tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        self.model.zero_grad()
        output[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam     = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam     = (cam - cam.min()) / (cam.max() + 1e-8)
        return cam.squeeze().cpu().numpy()


def overlay_cam(image: Image.Image, cam: np.ndarray) -> np.ndarray:
    import cv2
    import matplotlib.pyplot as plt
    cam_resized = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    heatmap     = plt.cm.jet(cam_resized)[:, :, :3]
    img_arr     = np.array(image.resize((IMG_SIZE, IMG_SIZE))) / 255.0
    overlay     = np.clip(0.5 * img_arr + 0.5 * heatmap, 0, 1)
    return (overlay * 255).astype(np.uint8)


def run_gradcam(image: Image.Image, model):
    try:
        target_layer = model.backbone.blocks[-1][-1].conv_pw
        gcam         = GradCAM(model, target_layer)
        transform    = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
        tensor = transform(image).unsqueeze(0).to(DEVICE)
        tensor.requires_grad_(True)
        cam = gcam.generate(tensor)
        return overlay_cam(image, cam), None
    except Exception as e:
        return None, str(e)


# UI 
st.title("Eye Disease Detection")
st.caption("Select a disease, upload a retinal image, and get an instant prediction.")

# Sidebar
st.sidebar.header("Settings")

selected_disease = st.sidebar.radio(
    "Select Disease to Detect",
    options=list(DISEASE_CONFIG.keys()),
    index=0
)

threshold    = st.sidebar.slider(
    "Decision Threshold", 0.1, 0.9, 0.5, 0.05,
    help="Probability above which the image is classified as the selected disease."
)
show_gradcam = st.sidebar.checkbox("Show Grad-CAM Heatmap", value=True)

st.sidebar.divider()
st.sidebar.markdown(f"**About {selected_disease} Detection**")
st.sidebar.caption(DISEASE_CONFIG[selected_disease]["description"])

# Model file status indicators in sidebar
st.sidebar.divider()
st.sidebar.markdown("**Model File Status**")
for disease, cfg in DISEASE_CONFIG.items():
    if Path(cfg["model_path"]).exists():
        st.sidebar.success(f"{disease}: `{cfg['model_path']}` found")
    else:
        st.sidebar.error(f"{disease}: `{cfg['model_path']}` not found")

# Load selected model 
model, class_names = load_model(selected_disease)

if model is None:
    st.error(
        f"Model file `{DISEASE_CONFIG[selected_disease]['model_path']}` not found. "
        "Place it in the same folder as `app.py` and restart."
    )
    st.stop()

# File uploader 
st.subheader(f"Detect {selected_disease}")
uploaded = st.file_uploader(
    "Upload a fundus image",
    type=["jpg", "jpeg", "png", "bmp"],
    label_visibility="collapsed"
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    if show_gradcam:
        with col2:
            st.subheader("Grad-CAM")
            with st.spinner("Generating heatmap..."):
                cam_result, cam_error = run_gradcam(image, model)
            if cam_error:
                st.warning(f"Grad-CAM unavailable: {cam_error}")
            else:
                st.image(cam_result, use_container_width=True, caption="Region of attention")

    result       = predict(image, model, class_names, threshold)
    predicted    = result["predicted"]
    disease_prob = result["disease_prob"]
    normal_prob  = result["normal_prob"]
    confidence   = result["confidence"]
    disease_name = class_names[1]

    st.divider()
    st.subheader("Prediction Result")

    if predicted == disease_name:
        st.error(f"Prediction: **{predicted}**  |  Confidence: **{confidence:.1%}**")
    else:
        st.success(f"Prediction: **{predicted}**  |  Confidence: **{confidence:.1%}**")

    st.subheader("Class Probabilities")
    prob_col1, prob_col2 = st.columns(2)

    with prob_col1:
        st.metric("Normal", f"{normal_prob:.1%}")
        st.progress(normal_prob)

    with prob_col2:
        st.metric(disease_name, f"{disease_prob:.1%}")
        st.progress(disease_prob)

    st.divider()
    st.caption(
        "This tool is for research purposes only and is not a substitute "
        "for professional medical diagnosis."
    )