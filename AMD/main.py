import streamlit as st
import os
from roboflow import Roboflow
from PIL import Image
from datetime import datetime
from dotenv import load_dotenv
# ========== Load Environment Variables ==========
load_dotenv()
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")  # Make sure this is stored in a .env file

# ========== Configuration ==========
WORKSPACE = "amddetection"
PROJECT_NAME = "amd-segmentation-app"
MODEL_VERSION = 3
IMAGE_DIR = "images"
SAMPLE_IMAGES = {
    "Sample 1": "sample1.jpg",
    "Sample 2": "sample2.jpg"
}

# ========== Initialize Roboflow ==========
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace(WORKSPACE).project(PROJECT_NAME)
model = project.version(MODEL_VERSION).model


# ========== Styling ==========
st.set_page_config(
    page_title="Dry AMD - Drusen Classification",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    html, body, .stApp {
        background-color: #ffffff;
        color: #1A1A1A;
        font-family: 'Segoe UI', sans-serif;
    }
    h1, h2 {
        color: #005EB8;
    }
    .stButton>button {
        background-color: #005EB8;
        color: white;
        font-weight: bold;
        border-radius: 6px;
        padding: 0.5rem 1rem;
    }
    .stSpinner {
        color: #005EB8;
    }
    st.markdown("""""", unsafe_allow_html=True)

    </style>

""", unsafe_allow_html=True)

# ========== Sidebar ==========
st.sidebar.image("https://wpstaq-ap-southeast-2-media.s3.amazonaws.com/irisphoto/wp-content/uploads/media/2022/01/Infinity-8--scaled.jpg", width=300)
st.sidebar.title("🧠 Drusen Segmentation")
st.sidebar.info("""
Upload OCT retinal images and automatically detect **drusen** associated with **Dry AMD**.

This demo leverages AI segmentation from a Roboflow-trained model.
""")
st.sidebar.markdown("**Version:** 3.0")
st.sidebar.markdown("**Last Updated:** April 2025")
st.sidebar.markdown("---")
st.sidebar.markdown("📍 **Bradford, United Kingdom**")
st.sidebar.markdown("Need Support? Email: mahme121@bradford.ac.uk")

# ========== Main Interface ==========
st.title("Dry AMD - Drusen Segmentation")
st.write("This pilot tool enables automated segmentation of drusen using Coloured Fundus Images.")

uploaded_file = st.file_uploader("📤 Upload an OCT image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_dir = "images"
    os.makedirs(image_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = os.path.join(image_dir, f"{timestamp}_{uploaded_file.name}")
    
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.subheader("🖼️ Uploaded Image")
    st.image(image_path, use_container_width=True)

    st.subheader("📊 Segmentation Output")

    with st.spinner("🧠 Analyzing image for drusen..."):
        try:
            prediction = model.predict(image_path, confidence=40, overlap=30)
            output_path = os.path.join(image_dir, f"prediction_{timestamp}.jpg")
            prediction.save(output_path)

            st.image(output_path, caption="Detected Drusen", use_container_width=True)
            st.success("✅ Segmentation complete.")
        except Exception as e:
            st.error("❌ Prediction failed. Please ensure the image is a valid OCT scan.")
            st.exception(e)

    with st.expander("ℹ️ What is Drusen?"):
        st.write("""
        Drusen are yellow deposits under the retina and are often early signs of **Age-related Macular Degeneration (AMD)**.
        
        This app helps visualize drusen by segmenting them from OCT images using a trained AI model.
        """)

     # ========== Feedback Form ==========

st.markdown("---")
st.subheader("💬 User Feedback")

with st.form("feedback_form"):
    st.write("Help us improve! Please provide feedback below:")

    rating = st.radio("How would you rate your experience?", [1, 2, 3, 4, 5], horizontal=True)
    comments = st.text_area("Any suggestions or issues you encountered?")
    submit_button = st.form_submit_button("Submit Feedback")

    if submit_button:
        feedback_path = "feedback_log.csv"
        feedback_entry = f"{datetime.now()},{rating},{comments}\n"

        # Create the file if it doesn't exist
        if not os.path.exists(feedback_path):
            with open(feedback_path, "w") as f:
                f.write("timestamp,rating,comments\n")

        # Append feedback
        with open(feedback_path, "a") as f:
            f.write(feedback_entry)

        st.success("✅ Thank you for your feedback!")


# ========== Footer ==========
st.markdown("---")
st.markdown("""
**About the Owner**  
- **Name:** Moshood Abubakar (Updated by Riaz Ahmed)  
- **Role:** Data Scientist  
- **Location:** Bradford, UK  

**Need Support? email: M.ahmed121@bradford.ac.uk**  


- [GitHub Profile](https://github.com/yourusername)
""")