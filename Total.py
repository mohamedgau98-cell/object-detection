from ultralytics import YOLO
import cv2
import streamlit as st
from PIL import Image
import numpy as np
import base64
from pathlib import Path
from collections import Counter

st.set_page_config(page_title="OBJECT DETECTOR", page_icon="🤖", layout="wide")

# ══════════════════════════════════════════════════
#   WEKA JINA LA PICHA YAKO HAPA ↓
#   (iweke kwenye folder moja na sum.py)
# ══════════════════════════════════════════════════
MY_PHOTO = "Mohamed.jpg"   # ← JINA LA PICHA YAKO
# ══════════════════════════════════════════════════

# ─── Helper: picha -> base64 ───
def image_to_base64(image_path):
    path = Path(image_path)
    if path.exists():
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

# ─── Load YOLO model (cached) ───
@st.cache_resource
def load_model(name):
    return YOLO(f"{name}.pt")

# ─── Piga picha moja kutoka camera ───
def capture_from_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return None, "❌ Camera haikufunguka — angalia kama camera imeunganishwa"
    
    # Subiri camera iwe tayari
    for _ in range(5):
        cap.read()
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret or frame is None:
        return None, "❌ Imeshindwa kupiga picha — jaribu tena"
    
    # Convert BGR -> RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    return img, None

# ════════════════════════════════
# HEADER
# ════════════════════════════════
st.markdown("""
<style>
.main-header {
    border:2px solid green;
    border-radius:30px;
    background-color:green;
    text-align:center;
    padding:12px;
    color:white;
    margin-bottom:10px;
}
</style>
<div class="main-header">
    <h2>🔍 OBJECT DETECTOR</h2>
    <p>Upload image OR use Camera — let AI find objects for you</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ════════════════════════════════
# SIDEBAR
# ════════════════════════════════
st.sidebar.header("⚙️ Control Panel")

model_name = st.sidebar.selectbox("Choose Model", ["yolov8n", "yolov8s", "yolov8m"])

source = st.sidebar.radio("📷 Input Source", ["Upload Image", "Use Camera"])

file_uploaded = None

if source == "Upload Image":
    file_uploaded = st.sidebar.file_uploader(
        "Upload Image", type=["jpg", "png", "jpeg"]
    )
else:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📸 Camera")
    capture_btn = st.sidebar.button("📸 Piga Picha (Capture)")
    st.sidebar.markdown("*Bonyeza kitufe hapo juu kupiga picha*")

confidence = st.sidebar.slider("Confidence Threshold", 0, 100, 25) / 100
max_det = st.sidebar.selectbox("Max Detections", [5, 10, 20, 50])
run = st.sidebar.button("🚀 Run Detection")

# ════════════════════════════════
# CAMERA CAPTURE LOGIC
# ════════════════════════════════
if source == "Use Camera":
    if capture_btn:
        with st.spinner("📸 Inafungua camera..."):
            img_captured, error = capture_from_camera()
            if error:
                st.sidebar.error(error)
                st.session_state["camera_img"] = None
            else:
                st.session_state["camera_img"] = img_captured
                st.sidebar.success("✅ Picha imepigwa!")

# ════════════════════════════════
# Pata image kutoka chanzo
# ════════════════════════════════
def get_image():
    if source == "Upload Image" and file_uploaded:
        return Image.open(file_uploaded).convert("RGB")
    elif source == "Use Camera":
        return st.session_state.get("camera_img", None)
    return None

# ════════════════════════════════
# COLUMNS
# ════════════════════════════════
col1, col2, col3 = st.columns(3, gap="large")

# ── COL 1: Input Preview ──
with col1:
    st.subheader("📂 Input Preview")
    img = get_image()

    if img:
        if source == "Upload Image" and file_uploaded:
            st.success("✅ File uploaded successfully")
            st.write(f"**Name:** {file_uploaded.name}")
            st.write(f"**Type:** {file_uploaded.type}")
            st.write(f"**Size:** {file_uploaded.size} bytes")
        else:
            st.success("✅ Camera photo captured!")
            st.write(f"**Size:** {img.size[0]}x{img.size[1]} px")

        st.image(img, caption="Input Image", use_container_width=True)
    else:
        st.info("Hakuna image bado")
        if source == "Upload Image":
            st.markdown("""
**Instructions:**
1. Upload image kutoka sidebar
2. Choose YOLO model
3. Set Confidence Threshold
4. Click **🚀 Run Detection**
            """)
        else:
            st.markdown("""
**Instructions:**
1. Bonyeza **📸 Piga Picha** kwenye sidebar
2. Picha itaonekana hapa
3. Bonyeza **🚀 Run Detection**
            """)

# ── COL 2: Detection Window ──
with col2:
    st.subheader("🖼️ Image Display Window")
    img = get_image()

    if img and run:
        with st.spinner("⏳ Detecting objects..."):
            img_array = np.array(img)
            model = load_model(model_name)
            results = model(img_array, conf=confidence, max_det=max_det)

            annotated = results[0].plot()
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            st.image(annotated_rgb, caption="🔍 Detection Result", use_container_width=True)
            st.success("✅ Detection Complete!")

            # Hifadhi results
            st.session_state["results"] = results
            st.session_state["model_names"] = model.names

    elif img and not run:
        st.image(img, caption="Preview — Press 🚀 Run Detection", use_container_width=True)
    else:
        if source == "Use Camera":
            st.warning("⚠️ Piga picha kwanza kisha bonyeza Run Detection")
        else:
            st.warning("⚠️ Upload image kwanza")

# ── COL 3: Analysis & Output ──
with col3:
    st.subheader("📊 Analysis and Output")

    if run and "results" in st.session_state:
        results = st.session_state["results"]
        names = st.session_state["model_names"]
        detections = results[0].boxes

        if detections is not None and len(detections) > 0:
            detected_labels = []
            for box in detections:
                cls_id = int(box.cls[0])
                detected_labels.append(names[cls_id])

            counts = Counter(detected_labels)

            st.markdown(f"### 🎯 Total Objects: `{len(detections)}`")
            st.markdown("---")

            st.markdown("#### 📋 Object Count:")
            for obj, count in counts.items():
                bar = "🟦" * count
                st.markdown(f"**{obj}**: {bar} `{count}`")

            st.markdown("---")

            st.markdown("#### 🔍 Detection Details:")
            for i, box in enumerate(detections):
                cls_id = int(box.cls[0])
                conf_score = float(box.conf[0])
                label = names[cls_id]

                if conf_score >= 0.7:
                    emoji = "🟢"
                elif conf_score >= 0.4:
                    emoji = "🟡"
                else:
                    emoji = "🔴"

                st.write(f"{emoji} **{i+1}. {label}** — `{conf_score:.1%}`")

        else:
            st.warning("⚠️ No objects detected!")
            st.info("💡 Lower the Confidence Threshold (try 10–15%)")
    else:
        st.info("📊 Results will appear here after detection")

# ════════════════════════════════
# FOOTER NA PICHA YAKO
# ════════════════════════════════
img_base64 = image_to_base64(MY_PHOTO)

if img_base64:
    footer_img = f'<img src="data:image/png;base64,{img_base64}">'
else:
    footer_img = "👤"

st.markdown(f"""
<style>
.footer {{
    border-radius:20px;
    background-color:blue;
    color:white;
    text-align:center;
    padding:10px;
    font-size:13px;
    margin-top:25px;
}}
.footer img {{
    width:45px;
    height:45px;
    border-radius:50%;
    vertical-align:middle;
    margin-right:8px;
    border:2px solid white;
    object-fit:cover;
}}
</style>
<div class="footer">
    {footer_img}
    Developed by <b>BLECA</b><sup style="color:yellow">TM</sup>
</div>
""", unsafe_allow_html=True)

if not img_base64:
    st.sidebar.warning(f"⚠️ Picha '{MY_PHOTO}' haijapatikana!\nIweke kwenye folder moja na sum.py")
