# app_effnet.py
import streamlit as st
from predict_effnet import load_model, predict_image
from PIL import Image
import os

st.set_page_config(page_title="Retina EfficientNet Detector", layout="centered")
st.title("Retinal Disease Detector — EfficientNet-B0")
st.write("Upload a retinal fundus image. The model was trained on RetinaMNIST (demo).")

uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

@st.cache_resource
def get_model():
    ckpt = os.path.join("outputs", "best_effnet.pth")
    if not os.path.exists(ckpt):
        return None, None, None
    try:
        model, device, info = load_model(ckpt)
        return model, device, info
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

model, device, info = get_model()
if model is None:
    st.warning("Model not found. Run `python train_effnet.py` to train and generate outputs/best_effnet.pth")
else:
    if uploaded_file is not None:
        try:
            img = Image.open(uploaded_file).convert("RGB")
        except Exception:
            st.error("Unable to open image. Upload a valid image file.")
            st.stop()

        st.image(img, caption="Uploaded image", use_container_width=True)
        st.write("Predicting ...")
        pred, prob, probs, label_names = predict_image(img, model, device, info)
        label = label_names[pred] if label_names and pred < len(label_names) else str(pred)
        st.success(f"Predicted class: **{label}**  (probability: {prob:.3f})")

        st.subheader("All class probabilities")
        for i, p in enumerate(probs):
            name = label_names[i] if label_names and i < len(label_names) else str(i)
            st.write(f"{name}: {p:.3f}")
    else:
        st.info("Upload an image to see predictions.")
