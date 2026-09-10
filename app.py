import streamlit as st
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import timm
from torchvision import transforms
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# ── Config ────────────────────────────────────────────────────────────
# Confirmed against the MUMDMC2025 training notebook: IMG_SIZE=224,
# direct Resize((224,224)) (no crop), ImageNet normalization, classes
# sorted alphabetically by folder name (note: folder is
# "Potassium_Feldspar", which sorts AFTER Plagioclase, not the "K-feldspar"
# shorthand used elsewhere). Checkpoint is a timm state_dict, no custom head.
APP_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = APP_DIR / "models" / "ResNet_50.pth"

IMG_SIZE = 224
CLASS_NAMES = ["Biotite", "Hornblende", "Plagioclase", "Potassium Feldspar", "Quartz"]
NUM_CLASSES = len(CLASS_NAMES)
DEVICE = torch.device("cpu")

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

st.set_page_config(page_title="Mineral XAI", page_icon="🪨")


@st.cache_resource
def load_model():
    model = timm.create_model("resnet50", pretrained=False, num_classes=NUM_CLASSES)
    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def predict_and_explain(image: Image.Image):
    model = load_model()
    image = image.convert("RGB")
    resized = image.resize((IMG_SIZE, IMG_SIZE))
    rgb_float = np.array(resized).astype(np.float32) / 255.0

    tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].numpy()

    pred_idx = int(probs.argmax())

    target_layer = model.layer4[-1]
    cam = GradCAMPlusPlus(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(pred_idx)])[0]
    overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

    return probs, pred_idx, overlay


# ── UI ───────────────────────────────────────────────────────────────
st.title("🪨 Mineral XAI")
st.caption(
    "Explainable mineral classification from petrographic thin-section images. "
    "Upload a thin-section photo to see the model's prediction and, via "
    "Grad-CAM++, which regions of the image drove that prediction."
)

uploaded_file = st.file_uploader(
    "Upload a petrographic thin-section image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    with st.spinner("Running inference and generating Grad-CAM++..."):
        probs, pred_idx, overlay = predict_and_explain(image)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)
    with col2:
        st.image(overlay, caption="Grad-CAM++ explanation", use_container_width=True)

    st.metric("Prediction", CLASS_NAMES[pred_idx], f"{probs[pred_idx]:.1%} confidence")

    st.write("Full breakdown:")
    for name, p in sorted(zip(CLASS_NAMES, probs), key=lambda x: -x[1]):
        st.write(f"{name}: {p:.1%}")
        st.progress(float(p))
else:
    st.info("Upload a petrographic thin-section image above to get started.")

st.divider()
with st.expander("About this model"):
    st.write(
        "A ResNet-50 model trained on the MUMDMC2025 photomicrographic "
        "dataset to classify five mineral classes from petrographic "
        "thin-section images: Biotite, Hornblende, Plagioclase, Potassium "
        "Feldspar, and Quartz. Reaches 98.74% test accuracy "
        "(cross-validated 99.12% ± 0.85%). Grad-CAM++ highlights the image "
        "regions that most influenced the model's prediction, part of an "
        "explainability analysis accepted at ICCOMTECH 2026."
    )