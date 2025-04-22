import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import io
import os

# Import the model class
from lowlightenhancement import enhance_net_nopool

# --- Configuration ---
MODEL_WEIGHTS_PATH = './models/enhance_net_epoch_100.pth' # Load the final or best model
IMAGE_SIZE = 256 # Must match the training image size

# --- Load Model ---
@st.cache_resource # Cache the model loading
def load_model(model_path, device):
    model = enhance_net_nopool().to(device)
    if not os.path.exists(model_path):
        st.error(f"Model weights not found at {model_path}. Please ensure the model is trained and saved correctly.")
        return None
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval() # Set model to evaluation mode
        st.success("Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        return None

# --- Image Preprocessing and Enhancement ---
def enhance_image(model, image_bytes, device):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(), # Converts to [0, 1] tensor
    ])

    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device) # Add batch dimension and send to device

    with torch.no_grad(): # No need to track gradients during inference
        _, enhanced_tensor, _ = model(img_tensor)

    # Convert tensor back to PIL Image
    enhanced_tensor = enhanced_tensor.squeeze(0).cpu() # Remove batch dim and move to CPU
    # Clamp values just in case they go slightly out of bounds
    enhanced_tensor = torch.clamp(enhanced_tensor, 0.0, 1.0)
    enhanced_image = transforms.ToPILImage()(enhanced_tensor)

    return img, enhanced_image # Return original PIL and enhanced PIL

# --- Streamlit App ---
st.set_page_config(layout="wide")
st.title("🌃 Low-Light Image Enhancement")
st.write("Upload a low-light image to enhance it using the Zero-DCE inspired model.")

# Device selection
if torch.cuda.is_available():
    device = torch.device("cuda")
    st.sidebar.success("CUDA (GPU) available, using GPU.")
else:
    device = torch.device("cpu")
    st.sidebar.warning("CUDA not available, using CPU (may be slow).")

# Load the model
model = load_model(MODEL_WEIGHTS_PATH, device)

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if model is not None and uploaded_file is not None:
    # Read image bytes
    image_bytes = uploaded_file.getvalue()

    # Enhance the image
    with st.spinner("Enhancing image..."):
        try:
            original_pil, enhanced_pil = enhance_image(model, image_bytes, device)

            # Display images side-by-side
            col1, col2 = st.columns(2)
            with col1:
                st.image(original_pil, caption='Original Image', use_column_width=True)
            with col2:
                st.image(enhanced_pil, caption='Enhanced Image', use_column_width=True)

            # Option to download the enhanced image
            buf = io.BytesIO()
            enhanced_pil.save(buf, format="PNG")
            byte_im = buf.getvalue()
            st.download_button(
                label="Download Enhanced Image",
                data=byte_im,
                file_name=f"enhanced_{uploaded_file.name}",
                mime="image/png"
            )

        except Exception as e:
            st.error(f"An error occurred during enhancement: {e}")
            st.error("Please ensure the uploaded file is a valid image.")

elif model is None:
    st.warning("Model could not be loaded. Please check the 'MODEL_WEIGHTS_PATH' in app.py and ensure training was successful.")

else:
    st.info("Please upload an image file to get started.")

st.sidebar.markdown("---")
st.sidebar.header("About")
st.sidebar.info(
    "This app uses a neural network inspired by the Zero-DCE model "
    "to enhance images taken in low-light conditions. The model learns "
    "light-enhancement curves directly from the input image."
)
st.sidebar.markdown(f"**Model File:** `{MODEL_WEIGHTS_PATH}`")
st.sidebar.markdown(f"**Processing Device:** `{device}`")