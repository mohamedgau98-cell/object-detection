import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import io
from ultralytics import YOLO
import tempfile
import os

# ─────────────────────────────────────────
#  Page Configuration
# ─────────────────────────────────────────
st.set_page_config(
    page_title="🎯 YOLO Object Detector",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
#  Custom CSS – Dark Futuristic Theme
# ─────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap');

  /* ── Global ── */
  html, body, [class*="css"] {
    font-family: 'Share Tech Mono', monospace;
    background-color: #0a0e1a;
    color: #e0e6f0;
  }

  /* ── App background ── */
  .stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0f1629 50%, #0a0e1a 100%);
  }

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1526 0%, #111d35 100%);
    border-right: 1px solid #1e3a5f;
  }
  section[data-testid="stSidebar"] * {
    color: #a8c4e0 !important;
  }

  /* ── Header ── */
  .hero-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 2.8rem;
    font-weight: 900;
    background: linear-gradient(90deg, #00d4ff, #7b61ff, #00d4ff);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 3s linear infinite;
    text-align: center;
    letter-spacing: 4px;
    margin-bottom: 0.2rem;
  }
  .hero-sub {
    text-align: center;
    color: #4a7fa5;
    font-size: 0.85rem;
    letter-spacing: 3px;
    margin-bottom: 2rem;
  }
  @keyframes shimmer {
    0%   { background-position: 0%   center; }
    100% { background-position: 200% center; }
  }

  /* ── Metric cards ── */
  .metric-card {
    background: linear-gradient(135deg, #111d35, #0d1526);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1rem 1.4rem;
    text-align: center;
    box-shadow: 0 0 20px rgba(0,212,255,0.08);
  }
  .metric-label {
    font-size: 0.7rem;
    letter-spacing: 2px;
    color: #4a7fa5;
    text-transform: uppercase;
  }
  .metric-value {
    font-family: 'Orbitron', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #00d4ff;
  }

  /* ── Tabs ── */
  .stTabs [role="tablist"] {
    gap: 8px;
    border-bottom: 1px solid #1e3a5f;
  }
  .stTabs [role="tab"] {
    font-family: 'Orbitron', sans-serif;
    font-size: 0.75rem;
    letter-spacing: 2px;
    background: #111d35;
    border: 1px solid #1e3a5f;
    border-radius: 8px 8px 0 0;
    color: #4a7fa5;
    padding: 0.6rem 1.4rem;
  }
  .stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #0d2240, #1a3a60) !important;
    color: #00d4ff !important;
    border-color: #00d4ff !important;
  }

  /* ── Buttons ── */
  .stButton > button {
    font-family: 'Orbitron', sans-serif;
    font-size: 0.8rem;
    letter-spacing: 2px;
    background: linear-gradient(135deg, #0d2240, #1a3a60);
    color: #00d4ff;
    border: 1px solid #00d4ff;
    border-radius: 8px;
    padding: 0.6rem 2rem;
    transition: all 0.3s ease;
    width: 100%;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #00d4ff22, #7b61ff22);
    box-shadow: 0 0 20px rgba(0,212,255,0.3);
    transform: translateY(-1px);
  }

  /* ── File uploader ── */
  [data-testid="stFileUploader"] {
    background: #111d35;
    border: 2px dashed #1e3a5f;
    border-radius: 12px;
    padding: 1rem;
  }

  /* ── Detection box ── */
  .detection-box {
    background: #111d35;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1.2rem;
    margin-top: 1rem;
  }
  .detection-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.4rem 0;
    border-bottom: 1px solid #1e3a5f22;
    font-size: 0.85rem;
  }
  .det-label { color: #a8c4e0; }
  .det-conf  { color: #00d4ff; font-family: 'Orbitron', sans-serif; font-size: 0.8rem; }

  /* ── Selectbox / slider ── */
  .stSlider > div { padding: 0; }

  /* ── Status badge ── */
  .status-badge {
    display: inline-block;
    padding: 0.25rem 0.8rem;
    border-radius: 20px;
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
  }
  .badge-live   { background: #00d4ff22; border: 1px solid #00d4ff; color: #00d4ff; }
  .badge-ready  { background: #7b61ff22; border: 1px solid #7b61ff; color: #7b61ff; }

  /* ── Divider ── */
  hr { border-color: #1e3a5f; }

  /* ── Camera frame ── */
  .camera-frame {
    border: 2px solid #1e3a5f;
    border-radius: 12px;
    overflow: hidden;
  }

  /* ── Info box ── */
  .info-block {
    background: #111d35;
    border-left: 3px solid #00d4ff;
    padding: 0.8rem 1.2rem;
    border-radius: 0 8px 8px 0;
    font-size: 0.82rem;
    color: #a8c4e0;
    margin: 0.5rem 0;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
#  Load Model (cached)
# ─────────────────────────────────────────
@st.cache_resource
def load_model(model_name: str):
    """Download & cache the YOLO model."""
    return YOLO(model_name)


# ─────────────────────────────────────────
#  Detect Objects Helper
# ─────────────────────────────────────────
def detect_objects(model, image: np.ndarray, conf_thresh: float):
    """Run inference and return annotated image + results list."""
    results = model(image, conf=conf_thresh, verbose=False)
    annotated = results[0].plot()          # BGR annotated frame
    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label  = model.names[cls_id]
        conf   = float(box.conf[0])
        detections.append({"label": label, "confidence": conf})
    return annotated, detections


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


# ─────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("<div style='font-family:Orbitron,sans-serif;font-size:1.1rem;color:#00d4ff;letter-spacing:3px;margin-bottom:1rem;'>⚙ SETTINGS</div>", unsafe_allow_html=True)

    model_choice = st.selectbox(
        "YOLO Model",
        ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"],
        help="n=nano (haraka), s=small, m=medium, l=large (sahihi zaidi)"
    )

    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1, max_value=1.0, value=0.5, step=0.05,
        help="Acha detections chini ya thamani hii"
    )

    st.markdown("---")
    st.markdown("""
    <div class='info-block'>
    <b>Jinsi ya kutumia:</b><br>
    📁 <b>Upload Image</b> – Chagua picha kutoka kwenye kompyuta yako<br><br>
    📷 <b>Live Camera</b> – Capture picha moja kwa webcam yako
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem;color:#4a7fa5;line-height:1.8;'>
    🧠 Model: YOLOv8 (Ultralytics)<br>
    🖥 Framework: Streamlit<br>
    🎯 Classes: 80 (COCO dataset)<br>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────
#  Header
# ─────────────────────────────────────────
st.markdown("<div class='hero-title'>🎯 YOLO DETECTOR</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-sub'>REAL-TIME OBJECT DETECTION SYSTEM · POWERED BY YOLOv8</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  Load Model
# ─────────────────────────────────────────
with st.spinner(f"⏳ Inapakia model {model_choice} ..."):
    model = load_model(model_choice)

st.markdown(f"<div style='text-align:center;margin-bottom:1.5rem;'><span class='status-badge badge-ready'>✓ MODEL READY · {model_choice.upper()}</span></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  Tabs
# ─────────────────────────────────────────
tab1, tab2 = st.tabs(["📁  UPLOAD IMAGE", "📷  LIVE CAMERA"])


# ══════════════════════════════════════════
#  TAB 1 – Upload Image
# ══════════════════════════════════════════
with tab1:
    st.markdown("<div style='font-family:Orbitron,sans-serif;font-size:0.85rem;color:#4a7fa5;letter-spacing:2px;margin-bottom:1rem;'>PAKIA PICHA YAKO KWA DETECTION</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Chagua picha (JPG, PNG, BMP, WEBP)",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file)

        col_orig, col_result = st.columns(2, gap="medium")

        with col_orig:
            st.markdown("<div class='metric-label' style='margin-bottom:0.5rem;'>ORIGINAL IMAGE</div>", unsafe_allow_html=True)
            st.image(pil_img, use_column_width=True)
            w, h = pil_img.size
            st.markdown(f"<div style='font-size:0.75rem;color:#4a7fa5;text-align:center;'>{w} × {h} px</div>", unsafe_allow_html=True)

        with col_result:
            st.markdown("<div class='metric-label' style='margin-bottom:0.5rem;'>DETECTION RESULT</div>", unsafe_allow_html=True)
            with st.spinner("🔍 Inafanya detection..."):
                t0 = time.time()
                bgr_img = pil_to_bgr(pil_img)
                annotated_bgr, detections = detect_objects(model, bgr_img, conf_threshold)
                elapsed = (time.time() - t0) * 1000

            result_img = bgr_to_pil(annotated_bgr)
            st.image(result_img, use_column_width=True)

        # ── Metrics row ──
        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Objects Detected</div><div class='metric-value'>{len(detections)}</div></div>", unsafe_allow_html=True)
        with m2:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Inference Time</div><div class='metric-value'>{elapsed:.0f}<span style='font-size:1rem'>ms</span></div></div>", unsafe_allow_html=True)
        with m3:
            avg_conf = (sum(d['confidence'] for d in detections) / len(detections) * 100) if detections else 0
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Avg Confidence</div><div class='metric-value'>{avg_conf:.0f}<span style='font-size:1rem'>%</span></div></div>", unsafe_allow_html=True)

        # ── Detection list ──
        if detections:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='metric-label' style='margin-bottom:0.5rem;'>DETECTED OBJECTS</div>", unsafe_allow_html=True)
            items_html = "".join(
                f"<div class='detection-item'>"
                f"<span class='det-label'>{'●'} {d['label'].upper()}</span>"
                f"<span class='det-conf'>{d['confidence']*100:.1f}%</span>"
                f"</div>"
                for d in sorted(detections, key=lambda x: x['confidence'], reverse=True)
            )
            st.markdown(f"<div class='detection-box'>{items_html}</div>", unsafe_allow_html=True)

        # ── Download button ──
        st.markdown("<br>", unsafe_allow_html=True)
        buf = io.BytesIO()
        result_img.save(buf, format="PNG")
        st.download_button(
            label="⬇  DOWNLOAD RESULT IMAGE",
            data=buf.getvalue(),
            file_name="detection_result.png",
            mime="image/png"
        )
    else:
        st.markdown("""
        <div style='text-align:center;padding:3rem;color:#4a7fa5;'>
            <div style='font-size:3rem;margin-bottom:1rem;'>📂</div>
            <div style='font-family:Orbitron,sans-serif;letter-spacing:2px;font-size:0.8rem;'>
                PAKIA PICHA KWA KUANZA
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════
#  TAB 2 – Live Camera (Capture)
# ══════════════════════════════════════════
with tab2:
    st.markdown("<div style='font-family:Orbitron,sans-serif;font-size:0.85rem;color:#4a7fa5;letter-spacing:2px;margin-bottom:0.5rem;'>TUMIA CAMERA KUCHUKUA PICHA</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-block'>
    📷 Bonyeza <b>"Take Photo"</b> kupiga picha, kisha <b>"Detect Objects"</b> kufanya uchambuzi.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Streamlit's built-in camera widget
    camera_image = st.camera_input(
        label="Camera",
        label_visibility="collapsed",
        key="camera_capture"
    )

    if camera_image is not None:
        pil_cam = Image.open(camera_image)

        col_c1, col_c2 = st.columns(2, gap="medium")
        with col_c1:
            st.markdown("<div class='metric-label' style='margin-bottom:0.5rem;'>CAPTURED IMAGE</div>", unsafe_allow_html=True)
            st.image(pil_cam, use_column_width=True)

        with col_c2:
            st.markdown("<div class='metric-label' style='margin-bottom:0.5rem;'>DETECTION RESULT</div>", unsafe_allow_html=True)
            with st.spinner("🔍 Inafanya detection..."):
                t0 = time.time()
                bgr_cam = pil_to_bgr(pil_cam)
                annotated_cam, detections_cam = detect_objects(model, bgr_cam, conf_threshold)
                elapsed_cam = (time.time() - t0) * 1000
            result_cam = bgr_to_pil(annotated_cam)
            st.image(result_cam, use_column_width=True)

        # ── Metrics ──
        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Objects Detected</div><div class='metric-value'>{len(detections_cam)}</div></div>", unsafe_allow_html=True)
        with m2:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Inference Time</div><div class='metric-value'>{elapsed_cam:.0f}<span style='font-size:1rem'>ms</span></div></div>", unsafe_allow_html=True)
        with m3:
            avg_c = (sum(d['confidence'] for d in detections_cam) / len(detections_cam) * 100) if detections_cam else 0
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Avg Confidence</div><div class='metric-value'>{avg_c:.0f}<span style='font-size:1rem'>%</span></div></div>", unsafe_allow_html=True)

        # ── Detection list ──
        if detections_cam:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='metric-label' style='margin-bottom:0.5rem;'>DETECTED OBJECTS</div>", unsafe_allow_html=True)
            items_html2 = "".join(
                f"<div class='detection-item'>"
                f"<span class='det-label'>{'●'} {d['label'].upper()}</span>"
                f"<span class='det-conf'>{d['confidence']*100:.1f}%</span>"
                f"</div>"
                for d in sorted(detections_cam, key=lambda x: x['confidence'], reverse=True)
            )
            st.markdown(f"<div class='detection-box'>{items_html2}</div>", unsafe_allow_html=True)

        # ── Download ──
        st.markdown("<br>", unsafe_allow_html=True)
        buf2 = io.BytesIO()
        result_cam.save(buf2, format="PNG")
        st.download_button(
            label="⬇  DOWNLOAD RESULT IMAGE",
            data=buf2.getvalue(),
            file_name="camera_detection_result.png",
            mime="image/png"
        )
    else:
        st.markdown("""
        <div style='text-align:center;padding:3rem;color:#4a7fa5;'>
            <div style='font-size:3rem;margin-bottom:1rem;'>📷</div>
            <div style='font-family:Orbitron,sans-serif;letter-spacing:2px;font-size:0.8rem;'>
                RUHUSU CAMERA · KISHA PIGA PICHA
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ──
st.markdown("---")
st.markdown("""
<div style='text-align:center;font-size:0.72rem;color:#1e3a5f;letter-spacing:2px;padding:0.5rem;'>
YOLO OBJECT DETECTOR · BUILT WITH STREAMLIT + ULTRALYTICS · YOLOv8
</div>
""", unsafe_allow_html=True)