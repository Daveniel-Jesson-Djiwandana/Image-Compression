import streamlit as st
import numpy as np
from PIL import Image
import io

# ---------- CACHED MATH ----------

@st.cache_data
def get_svd(img_array):
    """SVD is slow, so we cache it using the numpy array as the key."""
    svd_results = []
    # Process R, G, and B
    for i in range(3):
        U, S, Vt = np.linalg.svd(img_array[:, :, i], full_matrices=False)
        svd_results.append((U, S, Vt))
    return svd_results

def reconstruct_image(svd_results, k):
    """This is fast matrix math, no need to cache."""
    reconstructed_channels = []
    for U, S, Vt in svd_results:
        # Reconstruct: U_k * S_k * Vt_k
        low_rank = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
        reconstructed_channels.append(low_rank)
    
    compressed = np.stack(reconstructed_channels, axis=2)
    compressed = np.clip(compressed, 0, 255).astype(np.uint8)
    return Image.fromarray(compressed)

# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="SVD Image Compression", layout="wide")

st.title("🖼️ Fast SVD Image Compression")
st.write("Upload an image. The SVD calculation happens once, then the slider is instant.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and convert to array immediately
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image, dtype=float)

    # Calculate SVD (this will only run once per unique image)
    with st.spinner("Performing SVD math..."):
        svd_data = get_svd(img_array)

    # Slider logic
    max_k = len(svd_data[0][1]) 
    k = st.slider("Compression level (k)", 1, max_k, min(100, max_k))

    col1, col2 = st.columns(2)

    # Reconstruct based on k
    compressed_image = reconstruct_image(svd_data, k)

    with col1:
        st.subheader("Original")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader(f"Compressed (k = {k})")
        st.image(compressed_image, use_container_width=True)

   