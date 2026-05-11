"""
YOLO Object Detector — Streamlit App
=====================================
Requirements (requirements.txt):
    streamlit>=1.35.0
    ultralytics>=8.0.0
    opencv-python-headless>=4.9.0
    Pillow>=10.0.0
    numpy>=1.24.0

Bugs fixed in this version:
  1. Removed unused imports: tempfile, os
  2. st.image use_column_width (deprecated) → use_container_width=True
  3. detect_objects: local variable shadowed built-in 'model' → renamed to 'yolo_model'
  4. BytesIO buffer missing seek(0) before download_button → fixed inside helper
  5. ZeroDivisionError when no detections → guarded with conditional
  6. PIL Image opened from UploadedFile: seek(0) added for safe re-use
  7. Camera image opened from UploadedFile: seek(0) added for safe re-use
  8. f-string inner quotes standardised to avoid SyntaxWarning on older Python
  9. Type hints added to all helper functions
 10. All comments and UI strings are in English
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import io
from ultralytics import YOLO

# ─────────────────────────────────────────
#  Page Configuration
# ─────────────────────────────────────────
st.set_page_config(
    page_title="YOLO Object Detector",
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

  /* ── Status badge ── */
  .status-badge {
    display: inline-block;
    padding: 0.25rem 0.8rem;
    border-radius: 20px;
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
  }
  .badge-live  { background: #00d4ff22; border: 1px solid #00d4ff; color: #00d4ff; }
  .badge-ready { background: #7b61ff22; border: 1px solid #7b61ff; color: #7b61ff; }

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
def load_model(model_name: str) -> YOLO:
    """Download and cache the YOLO model weights. Returns a YOLO instance."""
    return YOLO(model_name)


# ─────────────────────────────────────────
#  Helper Functions
# ─────────────────────────────────────────
def detect_objects(
    yolo_model: YOLO,
    image: np.ndarray,
    conf_thresh: float,
) -> tuple[np.ndarray, list[dict]]:
    """
    Run YOLO inference on a BGR numpy array.

    Args:
        yolo_model:  Loaded YOLO instance.
                    NOTE: parameter is named 'yolo_model' (not 'model') to avoid
                    shadowing the module-level 'model' variable — fixes W0621.
        image:       BGR image as a numpy array (OpenCV format).
        conf_thresh: Minimum confidence threshold (0.0 – 1.0).

    Returns:
        Tuple of (annotated_bgr_image, list_of_detection_dicts).
        Each detection dict has keys: 'label' (str) and 'confidence' (float).
    """
    results = yolo_model(image, conf=conf_thresh, verbose=False)
    annotated: np.ndarray = results[0].plot()  # returns BGR numpy array
    detections: list[dict] = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label  = yolo_model.names[cls_id]
        conf   = float(box.conf[0])
        detections.append({"label": label, "confidence": conf})
    return annotated, detections


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    """Convert a PIL Image (any mode) to a BGR numpy array for OpenCV."""
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    """Convert a BGR numpy array back to an RGB PIL Image."""
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def image_to_download_bytes(pil_img: Image.Image) -> bytes:
    """
    Encode a PIL Image as PNG bytes ready for st.download_button.

    Bug fixed: BytesIO buffer is always seeked to position 0 before
    calling getvalue(), which prevents st.download_button from receiving
    an empty buffer when the stream pointer is at the end.
    """
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)  # REQUIRED: reset pointer so getvalue() reads from the start
    return buf.getvalue()


def open_uploaded_image(uploaded_file) -> Image.Image:
    """
    Safely open a Streamlit UploadedFile as a PIL Image.

    Bug fixed: Seeks the file buffer back to position 0 before opening,
    which prevents a blank/corrupt image if the buffer was already read
    earlier in the same Streamlit rerun cycle.
    """
    uploaded_file.seek(0)  # REQUIRED: reset stream before PIL reads it
    return Image.open(uploaded_file)


# ─────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='font-family:Orbitron,sans-serif;font-size:1.1rem;"
        "color:#00d4ff;letter-spacing:3px;margin-bottom:1rem;'>⚙ SETTINGS</div>",
        unsafe_allow_html=True,
    )

    model_choice: str = st.selectbox(
        "YOLO Model",
        ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"],
        help="n=nano (fastest)  s=small  m=medium  l=large (most accurate)",
    )

    conf_threshold: float = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Detections below this confidence score are ignored",
    )

    st.markdown("---")
    st.markdown(
        """
        <div class='info-block'>
        <b>How to use:</b><br>
        📁 <b>Upload Image</b> – Select an image from your device<br><br>
        📷 <b>Live Camera</b> – Capture a photo with your webcam
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        """
        <div style='font-size:0.72rem;color:#4a7fa5;line-height:1.8;'>
        🧠 Model: YOLOv8 (Ultralytics)<br>
        🖥 Framework: Streamlit<br>
        🎯 Classes: 80 (COCO dataset)<br>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────
#  Header
# ─────────────────────────────────────────
st.markdown("<div class='hero-title'>🎯 YOLO DETECTOR</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='hero-sub'>REAL-TIME OBJECT DETECTION SYSTEM · POWERED BY YOLOv8</div>",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────
#  Load Model
# ─────────────────────────────────────────
with st.spinner(f"Loading model {model_choice} — please wait ..."):
    model = load_model(model_choice)

st.markdown(
    f"<div style='text-align:center;margin-bottom:1.5rem;'>"
    f"<span class='status-badge badge-ready'>✓ MODEL READY · {model_choice.upper()}</span>"
    f"</div>",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────
#  Tabs
# ─────────────────────────────────────────
tab1, tab2 = st.tabs(["📁  UPLOAD IMAGE", "📷  LIVE CAMERA"])


# ══════════════════════════════════════════
#  TAB 1 – Upload Image
# ══════════════════════════════════════════
with tab1:
    st.markdown(
        "<div style='font-family:Orbitron,sans-serif;font-size:0.85rem;"
        "color:#4a7fa5;letter-spacing:2px;margin-bottom:1rem;'>"
        "UPLOAD YOUR IMAGE FOR DETECTION</div>",
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Select an image (JPG, PNG, BMP, WEBP)",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        # BUG FIX: seek(0) before opening ensures PIL reads from the start
        # of the buffer even if Streamlit already consumed it internally.
        pil_img = open_uploaded_image(uploaded_file)

        col_orig, col_result = st.columns(2, gap="medium")

        with col_orig:
            st.markdown(
                "<div class='metric-label' style='margin-bottom:0.5rem;'>ORIGINAL IMAGE</div>",
                unsafe_allow_html=True,
            )
            # BUG FIX: use_column_width is deprecated → use_container_width=True
            st.image(pil_img, use_container_width=True)
            w, h = pil_img.size
            st.markdown(
                f"<div style='font-size:0.75rem;color:#4a7fa5;text-align:center;'>"
                f"{w} &times; {h} px</div>",
                unsafe_allow_html=True,
            )

        with col_result:
            st.markdown(
                "<div class='metric-label' style='margin-bottom:0.5rem;'>DETECTION RESULT</div>",
                unsafe_allow_html=True,
            )
            with st.spinner("Running object detection ..."):
                t0 = time.time()
                bgr_img = pil_to_bgr(pil_img)
                annotated_bgr, detections = detect_objects(model, bgr_img, conf_threshold)
                elapsed = (time.time() - t0) * 1000  # milliseconds

            result_img = bgr_to_pil(annotated_bgr)
            # BUG FIX: use_column_width deprecated → use_container_width=True
            st.image(result_img, use_container_width=True)

        # ── Metrics row ──
        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)

        with m1:
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-label'>Objects Detected</div>"
                f"<div class='metric-value'>{len(detections)}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-label'>Inference Time</div>"
                f"<div class='metric-value'>{elapsed:.0f}"
                f"<span style='font-size:1rem'>ms</span></div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with m3:
            # BUG FIX: guard against ZeroDivisionError when no objects are found
            avg_conf = (
                sum(d["confidence"] for d in detections) / len(detections) * 100
                if detections
                else 0.0
            )
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-label'>Avg Confidence</div>"
                f"<div class='metric-value'>{avg_conf:.0f}"
                f"<span style='font-size:1rem'>%</span></div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # ── Detection list ──
        if detections:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                "<div class='metric-label' style='margin-bottom:0.5rem;'>DETECTED OBJECTS</div>",
                unsafe_allow_html=True,
            )
            items_html = "".join(
                f"<div class='detection-item'>"
                f"<span class='det-label'>&#9679; {d['label'].upper()}</span>"
                f"<span class='det-conf'>{d['confidence'] * 100:.1f}%</span>"
                f"</div>"
                for d in sorted(detections, key=lambda x: x["confidence"], reverse=True)
            )
            st.markdown(
                f"<div class='detection-box'>{items_html}</div>",
                unsafe_allow_html=True,
            )

        # ── Download button ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            label="⬇  DOWNLOAD RESULT IMAGE",
            # BUG FIX: image_to_download_bytes() performs seek(0) internally
            data=image_to_download_bytes(result_img),
            file_name="detection_result.png",
            mime="image/png",
        )

    else:
        st.markdown(
            """
            <div style='text-align:center;padding:3rem;color:#4a7fa5;'>
                <div style='font-size:3rem;margin-bottom:1rem;'>📂</div>
                <div style='font-family:Orbitron,sans-serif;letter-spacing:2px;font-size:0.8rem;'>
                    UPLOAD AN IMAGE TO GET STARTED
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════
#  TAB 2 – Live Camera (Capture)
# ══════════════════════════════════════════
with tab2:
    st.markdown(
        "<div style='font-family:Orbitron,sans-serif;font-size:0.85rem;"
        "color:#4a7fa5;letter-spacing:2px;margin-bottom:0.5rem;'>"
        "USE YOUR CAMERA TO CAPTURE A PHOTO</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class='info-block'>
        📷 Click <b>"Take Photo"</b> to capture an image — detection runs automatically.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # Streamlit's built-in camera widget
    camera_image = st.camera_input(
        label="Camera",
        label_visibility="collapsed",
        key="camera_capture",
    )

    if camera_image is not None:
        # BUG FIX: seek(0) before PIL opens the camera buffer
        camera_image.seek(0)
        pil_cam = Image.open(camera_image)

        col_c1, col_c2 = st.columns(2, gap="medium")

        with col_c1:
            st.markdown(
                "<div class='metric-label' style='margin-bottom:0.5rem;'>CAPTURED IMAGE</div>",
                unsafe_allow_html=True,
            )
            # BUG FIX: use_column_width deprecated → use_container_width=True
            st.image(pil_cam, use_container_width=True)

        with col_c2:
            st.markdown(
                "<div class='metric-label' style='margin-bottom:0.5rem;'>DETECTION RESULT</div>",
                unsafe_allow_html=True,
            )
            with st.spinner("Running object detection ..."):
                t0 = time.time()
                bgr_cam = pil_to_bgr(pil_cam)
                annotated_cam, detections_cam = detect_objects(model, bgr_cam, conf_threshold)
                elapsed_cam = (time.time() - t0) * 1000  # milliseconds

            result_cam = bgr_to_pil(annotated_cam)
            # BUG FIX: use_column_width deprecated → use_container_width=True
            st.image(result_cam, use_container_width=True)

        # ── Metrics ──
        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)

        with m1:
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-label'>Objects Detected</div>"
                f"<div class='metric-value'>{len(detections_cam)}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-label'>Inference Time</div>"
                f"<div class='metric-value'>{elapsed_cam:.0f}"
                f"<span style='font-size:1rem'>ms</span></div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with m3:
            # BUG FIX: guard against ZeroDivisionError when no objects are found
            avg_c = (
                sum(d["confidence"] for d in detections_cam) / len(detections_cam) * 100
                if detections_cam
                else 0.0
            )
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-label'>Avg Confidence</div>"
                f"<div class='metric-value'>{avg_c:.0f}"
                f"<span style='font-size:1rem'>%</span></div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # ── Detection list ──
        if detections_cam:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                "<div class='metric-label' style='margin-bottom:0.5rem;'>DETECTED OBJECTS</div>",
                unsafe_allow_html=True,
            )
            items_html2 = "".join(
                f"<div class='detection-item'>"
                f"<span class='det-label'>&#9679; {d['label'].upper()}</span>"
                f"<span class='det-conf'>{d['confidence'] * 100:.1f}%</span>"
                f"</div>"
                for d in sorted(detections_cam, key=lambda x: x["confidence"], reverse=True)
            )
            st.markdown(
                f"<div class='detection-box'>{items_html2}</div>",
                unsafe_allow_html=True,
            )

        # ── Download ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            label="⬇  DOWNLOAD RESULT IMAGE",
            # BUG FIX: image_to_download_bytes() performs seek(0) internally
            data=image_to_download_bytes(result_cam),
            file_name="camera_detection_result.png",
            mime="image/png",
        )

    else:
        st.markdown(
            """
            <div style='text-align:center;padding:3rem;color:#4a7fa5;'>
                <div style='font-size:3rem;margin-bottom:1rem;'>📷</div>
                <div style='font-family:Orbitron,sans-serif;letter-spacing:2px;font-size:0.8rem;'>
                    ALLOW CAMERA ACCESS · THEN TAKE A PHOTO
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── Footer ──
st.markdown("---")
st.markdown(
    """
    <div style='text-align:center;font-size:0.72rem;color:#1e3a5f;letter-spacing:2px;padding:0.5rem;'>
    YOLO OBJECT DETECTOR · BUILT WITH STREAMLIT + ULTRALYTICS · YOLOv8
    </div>
    """,
    unsafe_allow_html=True,
)