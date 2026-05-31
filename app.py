import streamlit as st
import torch
import imageio
from PIL import Image
from diffusers import StableVideoDiffusionPipeline
import tempfile
import os

st.set_page_config(
    page_title="AI Image to Video Generator",
    page_icon="🎬",
    layout="wide"
)

# CSS
st.markdown("""
<style>
.main {
    background-color: #87CEEB;
}

.stButton > button {
    background-color: #FFA500;
    color: white;
    border-radius: 10px;
    height: 50px;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #FF8C00;
}

.stTextInput > div > div > input {
    background-color: #D3D3D3;
}

.title {
    text-align:center;
    color:#1E3A8A;
    font-size:40px;
    font-weight:bold;
}

.subtitle {
    text-align:center;
    color:#555;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="title">🎬 AI Image To Video Generator</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload an image and generate a video using Stable Video Diffusion</p>', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "stabilityai/stable-video-diffusion-img2vid"
    
    try:
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            variant="fp16" if device == "cuda" else None
        )

        if device == "cuda":
            try:
                # This requires 'accelerate' and a GPU
                pipe.enable_model_cpu_offload()
                st.info("GPU detected: Model CPU offloading enabled.")
            except Exception as offload_err:
                st.warning(f"Could not enable CPU offload, moving model to GPU: {offload_err}")
                pipe.to("cuda")
        else:
            st.warning("No GPU detected. Running on CPU (this will be very slow).")
            pipe.to("cpu")

        # Slicing helps with memory even on CPU
        pipe.enable_attention_slicing()

        return pipe
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["png", "jpg", "jpeg"]
)

video_text = st.text_input(
    "Text Overlay",
    value="AI Generated Video"
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image")

    if st.button("🚀 Generate Video"):

        with st.spinner("Generating video..."):
            try:
                pipe = load_model()
                
                if pipe is None:
                    st.error("Failed to initialize the pipeline. Please check if 'accelerate' is installed.")
                else:
                    image = image.resize((512, 288))

                    frames = pipe(
                        image,
                        num_frames=14, 
                        decode_chunk_size=2
                    ).frames[0]

                    output_path = tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".mp4"
                    ).name

                    imageio.mimsave(
                        output_path,
                        frames,
                        fps=7
                    )

                    st.success("Video Generated Successfully!")
                    st.video(output_path)

                    with open(output_path, "rb") as file:
                        st.download_button(
                            "📥 Download Video",
                            file,
                            file_name="generated_video.mp4",
                            mime="video/mp4"
                        )
            except Exception as e:
                st.error(f"An error occurred during generation: {e}")