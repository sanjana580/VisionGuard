
import os
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

st.set_page_config(
    page_title="VisionGuard | AI Inspection",
    page_icon="🔍",
    layout="wide"
)

# ---------- Styling ----------
st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
.hero {
    background: linear-gradient(120deg, #14213d, #244a73);
    padding: 28px;
    border-radius: 18px;
    color: white;
    margin-bottom: 24px;
}
.hero h1 {
    color: white;
    margin-bottom: 8px;
}
.metric-card {
    background: white;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #e3e8ef;
}
.small-note {
    color: #526174;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# ---------- Model ----------
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "visionguard_model.pth"
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

@st.cache_resource
def load_model():
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )

    classes = checkpoint["classes"]

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(
        model.fc.in_features, len(classes)
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, classes

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------- Sidebar ----------
with st.sidebar:
    st.title("🔍 VisionGuard")
    st.caption("Industrial AI Inspection")
    page = st.radio(
        "Navigation",
        ["Dashboard", "Defect Guide", "About Project"]
    )
    st.divider()
    st.caption("Deep learning · Computer vision")

# ---------- Dashboard ----------
if page == "Dashboard":
    st.markdown("""
    <div class="hero">
        <h1>VisionGuard</h1>
        <p>AI-powered industrial surface defect classification</p>
        <p>Upload a surface image to analyze its visual defect category.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Defect Categories", "6")

    with col2:
        st.metric("Validation Images", "360")

    with col3:
        st.metric("Validation Accuracy", "97.50%")

    st.caption(
        "Validation accuracy is from the current experiment; "
        "it is not a guarantee for every uploaded image."
    )

    st.subheader("Surface Image Analysis")

    uploaded_file = st.file_uploader(
        "Upload an industrial surface image",
        type=["jpg", "jpeg", "png", "bmp"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        left, right = st.columns([1, 1])

        with left:
            st.image(
                image,
                caption="Uploaded surface image",
                use_container_width=True
            )

        with right:
            st.markdown("#### Prediction")

            if st.button("Analyze Surface", type="primary"):
                if not os.path.isfile(MODEL_PATH):
                    st.error("Trained model file was not found.")
                else:
                    try:
                        model, classes = load_model()
                        tensor = transform(image).unsqueeze(0).to(device)

                        with torch.no_grad():
                            logits = model(tensor)
                            probabilities = torch.softmax(
                                logits, dim=1
                            )[0]

                        confidence, index = probabilities.max(dim=0)
                        label = classes[index.item()].replace(
                            "_", " "
                        ).title()
                        score = confidence.item() * 100

                        st.success("Analysis completed")
                        st.markdown(f"### {label}")
                        st.metric("Model Confidence", f"{score:.2f}%")

                        st.markdown("#### Class probabilities")

                        for i, category in enumerate(classes):
                            value = probabilities[i].item()
                            st.write(
                                f"{category.replace('_', ' ').title()}"
                            )
                            st.progress(value)

                        st.info(
                            "This model selects among six known defect "
                            "categories. It may misclassify defect-free "
                            "or unfamiliar images. Human inspection is "
                            "recommended."
                        )

                    except Exception as error:
                        st.error(f"Prediction failed: {error}")

# ---------- Defect Guide ----------
elif page == "Defect Guide":
    st.title("Defect Category Guide")
    st.write(
        "VisionGuard was trained to classify images into these six "
        "categories from the NEU-DET dataset."
    )

    descriptions = {
        "Crazing": "Fine, network-like cracking patterns.",
        "Inclusion": "Unwanted inclusions within the material.",
        "Patches": "Irregular patch-like surface patterns.",
        "Pitted Surface": "Small pits or cavities on the surface.",
        "Rolled-In Scale": "Scale irregularities embedded in the steel.",
        "Scratches": "Linear marks or grooves on the surface."
    }

    for name, description in descriptions.items():
        with st.expander(name):
            st.write(description)

# ---------- About ----------
else:
    st.title("About VisionGuard")
    st.write(
        "VisionGuard is a final-year AI/ML project exploring "
        "deep learning for industrial surface defect classification."
    )

    st.subheader("Technology Stack")
    st.markdown("""
    - Python
    - PyTorch and ResNet18 transfer learning
    - Pillow and torchvision for image preprocessing
    - Streamlit for the user interface
    - Scikit-learn for model evaluation
    """)

    st.subheader("Methodology")
    st.markdown("""
    1. Load and inspect the industrial defect dataset.
    2. Resize and normalize images.
    3. Train a pretrained ResNet18 classifier.
    4. Evaluate predictions using validation metrics.
    5. Integrate the model into an image-upload application.
    """)

    st.subheader("Current Limitations")
    st.write(
        "The current version classifies an image into one of six "
        "known categories. It does not localize defects or reliably "
        "identify images outside the training categories."
    )
