import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn

st.set_page_config(page_title="Chest X‑Ray Diagnosis", layout="centered")
st.title("🩺 Pneumonia vs Normal Chest X‑Ray Classifier")

# Load Model
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: Normal and Pneumonia
    model.load_state_dict(torch.load("resnet18_pretrained.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# Updated Transform with 3-channel ImageNet normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # standard ImageNet means
        std=[0.229, 0.224, 0.225]    # standard ImageNet stds
    )
])

labels = ["Normal", "Pneumonia"]

# Upload
uploaded = st.file_uploader("Upload a Chest X‑Ray (.jpg, .png)", type=["jpg", "jpeg", "png"])
if uploaded:
    try:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded Chest X‑Ray", use_container_width=True)

        tensor = transform(img).unsqueeze(0)

        with st.spinner("Analyzing X‑ray..."):
            output = model(tensor)
            probabilities = torch.softmax(output, dim=1)[0]
            _, pred = torch.max(output, 1)

        st.success(f"🧠 **Prediction: {labels[pred.item()]}**")
        st.write(f"🟢 Confidence (Normal): `{probabilities[0]*100:.2f}%`")
        st.write(f"🔴 Confidence (Pneumonia): `{probabilities[1]*100:.2f}%`")

    except Exception as e:
        st.error("⚠️ Something went wrong while processing the image.")
        st.exception(e)
