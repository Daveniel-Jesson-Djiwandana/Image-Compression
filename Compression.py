import streamlit as st
import numpy as np
from PIL import Image
import io

@st.cache_data
def get_svd(image):
    image = image.convert("RGB")
    A = np.array(image, dtype=float)
    
    # Pre-calculate SVD for each channel once
    svd_results = []
    for i in range(3):
        U, S, Vt = np.linalg.svd(A[:, :, i], full_matrices=False)
        svd_results.append((U, S, Vt))
    return svd_results

def reconstruct_image(svd_results, k):
    reconstructed_channels = []
    for U, S, Vt in svd_results:
        # Just multiply the top k components
        low_rank = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
        reconstructed_channels.append(low_rank)
    
    compressed = np.stack(reconstructed_channels, axis=2)
    compressed = np.clip(compressed, 0, 255).astype(np.uint8)
    return Image.fromarray(compressed)

# ---------- SVD COMPRESSION FUNCTION ----------
def compress_channel(channel, k):
    U, S, Vt = np.linalg.svd(channel, full_matrices=False)
    return U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

def compress_image_color(image, k):
    image = image.convert("RGB")
    A = np.array(image, dtype=float)

    # Split channels
    R, G, B = A[:, :, 0], A[:, :, 1], A[:, :, 2]

    # Compress each channel
    R_k = compress_channel(R, k)
    G_k = compress_channel(G, k)
    B_k = compress_channel(B, k)

    # Stack back into RGB image
    compressed = np.stack([R_k, G_k, B_k], axis=2)

    # Clip values to valid range
    compressed = np.clip(compressed, 0, 255).astype(np.uint8)

    return Image.fromarray(compressed)

# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="SVD Image Compression", layout="wide")

st.title("🖼️ Image Compression using SVD (Color)")
st.write("Upload an image and compress it using Singular Value Decomposition.")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    with st.spinner("Analyzing image structure..."):
        svd_data = get_svd(image)

    max_k = min(image.size)
    k = st.slider(
        "Compression level (k)",
        min_value=1,
        max_value=max_k,
        value=min(100, max_k),
        step=1
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader(f"Compressed Image (k = {k})")
        compressed_image = reconstruct_image(svd_data, k)
        st.image(compressed_image)

    # Download button
    buf = io.BytesIO()
    compressed_image.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(
        label="📥 Download Compressed Image",
        data=byte_im,
        file_name="compressed.png",
        mime="image/png"
    )
