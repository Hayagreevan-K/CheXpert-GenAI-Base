import os
import io
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import models
import numpy as np
import cv2
from PIL import Image
import streamlit as st
import traceback
import json
import warnings
from openai import OpenAI

warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# ======================
# CONFIG
# ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"
]

# ======================
# MODEL LOADING
# ======================
@st.cache_resource
def load_model(checkpoint_path):
    model = models.densenet121(weights="IMAGENET1K_V1")
    in_feats = model.classifier.in_features
    model.classifier = torch.nn.Linear(in_feats, len(LABELS))
    model.to(DEVICE)

    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=DEVICE)
        if "model_state" in state:
            state = state["model_state"]
        model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print("No checkpoint found. Using ImageNet weights.")
    model.eval()
    return model


# ======================
# PREPROCESS
# ======================
@st.cache_resource
def get_preprocess():
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


# ======================
# GRAD-CAM FUNCTION
# ======================
def compute_gradcam_overlay_fixed(pil: Image.Image, model, label_index=None, image_weight=0.5, heatmap_weight=0.7):
    """
    Safe Grad-CAM — avoids autograd in-place errors by cloning tensors.
    """
    target = None
    for name, m in reversed(list(model.named_modules())):
        if "denseblock" in name or "features" in name:
            target = m
            break

    feats, grads = {"feat": None}, {"grad": None}
    hooks = []

    def _fwd(module, inp, out):
        feats["feat"] = out.clone().detach().cpu() if isinstance(out, torch.Tensor) else None

    def _bwd(module, grad_input, grad_output):
        if grad_output and grad_output[0] is not None:
            grads["grad"] = grad_output[0].clone().detach().cpu()

    if hasattr(target, "register_full_backward_hook"):
        hooks.append(target.register_full_backward_hook(_bwd))
    else:
        hooks.append(target.register_backward_hook(_bwd))
    hooks.append(target.register_forward_hook(_fwd))

    try:
        inp = get_preprocess()(pil).unsqueeze(0).to(DEVICE)
        logits = model(inp)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        if label_index is None:
            label_index = int(np.argmax(probs))
        score = logits[0, label_index]
        model.zero_grad()
        score.backward()

        feat = feats["feat"].numpy()[0]
        grad = grads["grad"].numpy()[0]
        weights = np.mean(grad, axis=(1, 2))
        cam = np.maximum(np.sum(weights[:, None, None] * feat, axis=0), 0)
        cam /= cam.max() + 1e-8
        cam = cv2.resize(cam, pil.size)
        heat_uint8 = np.uint8(cam * 255)
        cmap = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(np.array(pil.convert("RGB")), image_weight, cmap, heatmap_weight, 0)
        return Image.fromarray(overlay), None
    except Exception as e:
        return pil, f"Grad-CAM error: {e}"
    finally:
        for h in hooks:
            try:
                h.remove()
            except:
                pass


# ======================
# OPENAI REPORT GENERATION
# ======================
def generate_openai_report(api_key, preds, top_labels):
    try:
        client = OpenAI(api_key=api_key)
        pred_json = json.dumps(dict(zip(top_labels, preds[:len(top_labels)])), indent=2)
        prompt = f"""
        You are a radiologist. Based on these CheXpert model predictions:
        {pred_json}
        Write a professional chest X-ray report with sections:
        - Exam
        - Findings
        - Impression
        - Confidence level.
        """
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"OpenAI generation error:\n{e}"


# ======================
# STREAMLIT UI
# ======================
st.set_page_config(page_title="CheXpert — Base model + Grad-CAM + OpenAI report", layout="wide")
st.title("🩻 CheXpert — Base model + Grad-CAM + OpenAI report")

# MODEL LOADING
st.sidebar.header("Model / weights")
checkpoint_path = st.sidebar.text_input("Local checkpoint path:", "model_state_final.pth")
use_local = st.sidebar.checkbox("Use local checkpoint if present", value=True)
model = load_model(checkpoint_path if use_local else "")

# OPENAI
st.sidebar.header("OpenAI")
api_key = st.sidebar.text_input("OpenAI API Key (optional; not saved)", type="password")
use_openai = st.sidebar.checkbox("Generate OpenAI report", value=False)

# IMAGE UPLOAD
st.header("🩻 Upload chest X-ray")
uploaded = st.file_uploader("Upload chest X-ray (JPEG/PNG)", type=["jpg", "jpeg", "png"])

show_gradcam = st.checkbox("Compute Grad-CAM overlay", value=True)
topk = st.slider("Show top-k predictions", 1, 14, 3)

if uploaded:
    pil_img = Image.open(uploaded).convert("RGB")
    st.image(pil_img, caption="Uploaded X-ray", use_container_width=True)

    with st.spinner("Running inference..."):
        inp = get_preprocess()(pil_img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(inp)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        top_idx = np.argsort(probs)[::-1][:topk]
        top_labels = [LABELS[i] for i in top_idx]
        top_probs = [float(probs[i]) for i in top_idx]

        st.subheader("Inference Results")
        result_text = "\n".join([f"{lbl}: {p:.3f}" for lbl, p in zip(top_labels, top_probs)])
        st.write(result_text)

    # Grad-CAM
    if show_gradcam:
        with st.spinner("Computing Grad-CAM..."):
            overlay, err = compute_gradcam_overlay_fixed(pil_img, model, top_idx[0])
            if err:
                st.error(err)
            else:
                st.image(overlay, caption="Grad-CAM overlay", use_container_width=True)

    # OpenAI Report
    if use_openai and api_key.strip():
        with st.spinner("Generating report via OpenAI..."):
            report = generate_openai_report(api_key, top_probs, top_labels)
            st.markdown("### 🧠 OpenAI Report")
            st.write(report)
    elif use_openai:
        st.warning("⚠️ Please enter your OpenAI API key.")
else:
    st.info("Upload a chest X-ray to begin.")
