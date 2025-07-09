import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import os
import gdown

st.set_page_config(page_title="Chest X‑Ray Diagnosis", layout="centered")
st.title("🩺 Pneumonia vs Normal Chest X‑Ray Classifier")

MODEL_PATH = "resnet18_pretrained.pth"
DRIVE_FILE_ID = "14OuvtJ2279QSmGyCLtrjn2NdiYraLa2R"  # ✅ YOUR FILE ID

def download_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(f"https://drive.google.com/uc?id={DRIVE_FILE_ID}", MODEL_PATH, quiet=False)

@st.cache_resource
def load_model():
    download_model()
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

labels = ["Normal", "Pneumonia"]

uploaded = st.file_uploader("Upload a Chest X‑Ray (.jpg, .png)", type=["jpg", "jpeg", "png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Chest X‑Ray", use_container_width=True)

    tensor = transform(img).unsqueeze(0)
    with st.spinner("Analyzing X‑ray..."):
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]
        _, pred = torch.max(probs, 0)

    st.success(f"🧠 Prediction: **{labels[pred.item()]}**")
    st.write(f"🟢 Confidence (Normal): `{probs[0]*100:.2f}%`")
    st.write(f"🔴 Confidence (Pneumonia): `{probs[1]*100:.2f}%`")
