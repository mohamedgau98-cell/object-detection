from ultralytics import YOLO
import cv2
import streamlit as st
from PIL import Image
import numpy as np
from collections import Counter

st.set_page_config(page_title="OBJECT DETECTOR", page_icon="🤖", layout="wide")

# ─── Load YOLO ───
@st.cache_resource
def load_model(name):
    return YOLO(f"{name}.pt")

# ─── Piga picha kutoka camera ───
def capture_camera_frame():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return None, "❌ Camera haikufunguka"
    for _ in range(10):
        cap.read()
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None, "❌ Imeshindwa kupiga picha"
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb), None

# ─── Generate Caption ───
def generate_caption(counts):
    if not counts:
        return "Hakuna kitu kilichopatikana."
    parts = []
    for obj, count in counts.items():
        if count == 1:
            parts.append(f"one {obj}")
        else:
            parts.append(f"{count} {obj}s")
    if len(parts) == 1:
        return f"📸 This image contains {parts[0]}."
    elif len(parts) == 2:
        return f"📸 This image contains {parts[0]} and {parts[1]}."
    else:
        return f"📸 This image contains {', '.join(parts[:-1])}, and {parts[-1]}."

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
capture_btn = False

if source == "Upload Image":
    file_uploaded = st.sidebar.file_uploader("📁 Upload Image", type=["jpg", "png", "jpeg"])
else:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📸 Camera")
    capture_btn = st.sidebar.button("📸 Piga Picha (Capture)")

confidence = st.sidebar.slider("Confidence Threshold", 0, 100, 20) / 100
max_det = st.sidebar.selectbox("Max Detections", [5, 10, 20, 50])
run = st.sidebar.button("🚀 Run Detection")

# ════════════════════════════════
# CAMERA CAPTURE
# ════════════════════════════════
if capture_btn:
    with st.spinner("📸 Camera inafunguka..."):
        cam_img, err = capture_camera_frame(1)
        if err:
            st.sidebar.error(err)
            st.session_state["camera_img"] = None
        else:
            st.session_state["camera_img"] = cam_img
            st.sidebar.success("✅ Picha imepigwa!")

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
**Hatua:**
1. Upload image kwenye sidebar
2. Choose YOLO model
3. Set Confidence
4. Click **🚀 Run Detection**
            """)
        else:
            st.markdown("""
**Hatua:**
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
            st.session_state["results"] = results
            st.session_state["model_names"] = model.names
    elif img and not run:
        st.image(img, caption="Preview — Press 🚀 Run Detection", use_container_width=True)
    else:
        if source == "Use Camera":
            st.warning("⚠️ Bonyeza 📸 Piga Picha kwenye sidebar kwanza")
        else:
            st.warning("⚠️ Upload image kwanza kutoka sidebar")

# ── COL 3: Analysis + Caption ──
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

            caption = generate_caption(counts)
            st.markdown(f"""
<div style="
    background-color:#1a1a2e;
    border-left:4px solid #00ff00;
    border-radius:10px;
    padding:12px;
    color:white;
    font-size:15px;
    margin-bottom:15px;
">
{caption}
</div>
""", unsafe_allow_html=True)

            st.markdown(f"### 🎯 Total Objects: `{len(detections)}`")
            st.markdown("---")

            st.markdown("#### 📋 Object Count:")
            for obj, count in counts.items():
                bar = "🟦" * min(count, 10)
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
            st.info("💡 Punguza Confidence Threshold mpaka 10–15%")
    else:
        st.info("📊 Results na Caption zitaonekana hapa")

# ════════════════════════════════
# FOOTER — Clean, bila picha
# ════════════════════════════════
st.markdown("""
<style>
.footer {
    border-radius:20px;
    background-color:blue;
    color:white;
    text-align:center;
    padding:10px;
    font-size:13px;
    margin-top:25px;
}
</style>
<div class="footer">
    Developed by <b>Mo de Great</b><sup style="color:yellow">TM</sup>
</div>
""", unsafe_allow_html=True)