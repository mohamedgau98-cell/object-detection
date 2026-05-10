from ultralytics import YOLO
import cv2
import streamlit as st
from PIL import Image
import numpy as np

st.set_page_config(page_title="OBJECT DETECTOR", page_icon="🤖", layout="wide")

st.markdown("""
<style>
div.main-header {
    border:2px solid blue;
    border-radius:20px;
    width:100%;
    background-color:violet;
    text-align:center;
    padding:10px;
}
</style>
<div class="main-header">
    <center>
    <h2>🤖 OBJECT DETECTOR</h2>
    <p>Upload image of object and let AI find the object for you</p>
    </center>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# SIDE BAR
# ─────────────────────────────────────────
st.sidebar.header("⚙️ Control Panel")

# FIX 1: Tumia jina tofauti - "model_name" si "model"
# (awali model=YOLO(...) ilikuwa inabadilisha model=selectbox)
model_name = st.sidebar.selectbox("Choose Model", ["yolov8n", "yolov8s", "yolov8m"])

file_uploaded = st.sidebar.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
confidence = st.sidebar.slider("Confidence Threshold", 0, 100, 50) / 100
max_det = st.sidebar.selectbox("Max Detections", [5, 10, 20])
run = st.sidebar.button("Run Detection")

st.sidebar.markdown("""
<style>
    div.stButton>button:first-child{
        background-color:blue;
        color:white;
        border-radius:10px;
        width:100%;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# MAIN COLUMNS
# ─────────────────────────────────────────
col1, col2, col3 = st.columns(3, gap="large")

# ── Column 1: Input Preview ──
with col1:
    st.subheader("📂 Input Preview")
    if file_uploaded:
        st.success("File uploaded successfully")
        st.write(f"**File Name:** {file_uploaded.name}")
        st.write(f"**File Type:** {file_uploaded.type}")
        st.write(f"**File Size:** {file_uploaded.size} bytes")

        # FIX 2: Onyesha image katika col1
        image = Image.open(file_uploaded)
        st.image(image, caption="Uploaded Image", use_container_width=True)
    else:
        st.info("Upload an Image from the side bar")
        st.markdown("""
### Instructions
1. Upload an image
2. Choose a model
3. Set Confidence level
4. Click **Run Detection**
        """)

# ── Column 2: Detection Window ──
with col2:
    st.subheader("🔍 Image Display Window")

    if file_uploaded:
        if run:
            # FIX 3: Tumia model_name (si model) kupakia YOLO
            with st.spinner("Running Detection..."):
                model = YOLO(f"{model_name}.pt")

            # FIX 4: Soma image kutoka file_uploaded - SI camera
            image = Image.open(file_uploaded)
            img_array = np.array(image)  # convert PIL -> numpy (cv2 format)

            # FIX 5: Fanya detection kwenye image - SI video loop
            results = model(
                img_array,
                conf=confidence,
                max_det=max_det
            )

            # FIX 6: Pata annotated image na uionyeshe
            annotated = results[0].plot()  # numpy array na boxes
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            st.image(annotated_rgb, caption="Detection Result", use_container_width=True)
            st.success("Detection complete!")

        else:
            st.info("Press **Run Detection** to start")
    else:
        st.warning("No image uploaded yet")

# ── Column 3: Analysis & Output ──
with col3:
    st.subheader("📊 Analysis and Output")

    if file_uploaded and run:
        image = Image.open(file_uploaded)
        img_array = np.array(image)
        model = YOLO(f"{model_name}.pt")

        results = model(img_array, conf=confidence, max_det=max_det)
        detections = results[0].boxes

        if detections is not None and len(detections) > 0:
            st.write(f"**Total Objects Found:** {len(detections)}")
            st.markdown("---")

            # FIX 7: Onyesha kila object iliyopatikana
            names = model.names
            for i, box in enumerate(detections):
                cls_id = int(box.cls[0])
                conf_score = float(box.conf[0])
                label = names[cls_id]
                st.write(f"**{i+1}. {label}** — Confidence: `{conf_score:.2%}`")
        else:
            st.warning("No objects detected. Try lowering the confidence threshold.")
    else:
        st.info("Results will appear here after detection")

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.markdown("""
<style>
    .footer{
        position:relative;
        left:0;
        bottom:0;
        width:fit-content;
        border-radius:20px;
        background-color:blue;
        color:white;
        text-align:center;
        padding:10px;
        font-size:12px;
        margin-top:20px;
    }
</style>
<div class="footer">
    Developed by BLECA<sup style="color:red">TM</sup>
</div>
""", unsafe_allow_html=True)