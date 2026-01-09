import streamlit as st
import numpy as np
from PIL import Image
import io

# 1. Cached function: Uses the array to avoid "Unhashable" error
@st.cache_data
def get_svd_data(img_array):
    svd_results = []
    # Loop through R, G, B channels
    for i in range(3):
        U, S, Vt = np.linalg.svd(img_array[:, :, i], full_matrices=False)
        svd_results.append((U, S, Vt))
    return svd_results

def reconstruct(svd_results, k):
    reconstructed_channels = []
    for U, S, Vt in svd_results:
        # Fast matrix multiplication
        low_rank = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
        reconstructed_channels.append(low_rank)
    
    compressed = np.stack(reconstructed_channels, axis=2)
    compressed = np.clip(compressed, 0, 255).astype(np.uint8)
    return Image.fromarray(compressed)

# --- UI SETUP ---
st.set_page_config(page_title="SVD Compressor", layout="wide")
st.title("🖼️ Image Compressor")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image and ensure it's RGB
    image = Image.open(uploaded_file).convert("RGB")
    
    # CRITICAL: Resize if too large to prevent the "Oh no" crash (Memory limit)
    if max(image.size) > 1200:
        image.thumbnail((1200, 1200))
    
    img_array = np.array(image, dtype=float)

    # Calculation (Runs once)
    with st.spinner("Calculating SVD..."):
        try:
            svd_data = get_svd_data(img_array)
            
            # Slider
            max_k = len(svd_data[0][1])
            k = st.slider("Select k", 1, max_k, min(100, max_k))

            # Display
            col1, col2 = st.columns(2)
            comp_img = reconstruct(svd_data, k)

            with col1:
                st.image(image, caption="Original", use_container_width=True)
            with col2:
                st.image(comp_img, caption=f"Compressed (k={k})", use_container_width=True)

            # Download
            buf = io.BytesIO()
            comp_img.save(buf, format="JPEG", quality=95)
            byte_im = buf.getvalue()

            # 4. Display the new size so you can see it working
            st.metric("Compressed File Size", f"{len(byte_im) / 1024:.2f} KB")

            st.download_button(
                label="📥 Download JPEG",
                data=byte_im,
                file_name=f"compressed_k_{k}.jpg",
                mime="image/jpeg"
)

        except Exception as e:
            st.error(f"Something went wrong: {e}")