"""
YOLO Object Detector — Streamlit App
=====================================
Camera Tab: Auto-starts & captures frames continuously via JavaScript WebRTC.
No START button needed — camera begins immediately when tab is active.
Includes detailed object counting per class.

Extra dependency:
    pip install streamlit-js-eval
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import io
import base64
from collections import Counter
from ultralytics import YOLO
from streamlit_js_eval import streamlit_js_eval   # pip install streamlit-js-eval

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

  html, body, [class*="css"] {
    font-family: 'Share Tech Mono', monospace;
    background-color: #0a0e1a;
    color: #e0e6f0;
  }
  .stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0f1629 50%, #0a0e1a 100%);
  }
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1526 0%, #111d35 100%);
    border-right: 1px solid #1e3a5f;
  }
  section[data-testid="stSidebar"] * { color: #a8c4e0 !important; }

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
  .stTabs [role="tablist"] { gap: 8px; border-bottom: 1px solid #1e3a5f; }
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
  [data-testid="stFileUploader"] {
    background: #111d35;
    border: 2px dashed #1e3a5f;
    border-radius: 12px;
    padding: 1rem;
  }
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
  .det-count {
    background: #00d4ff22;
    border: 1px solid #00d4ff55;
    border-radius: 20px;
    padding: 0.1rem 0.6rem;
    font-family: 'Orbitron', sans-serif;
    font-size: 0.75rem;
    color: #00d4ff;
    min-width: 32px;
    text-align: center;
  }
  .count-table {
    background: #0d1526;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1rem 1.4rem;
    margin-top: 0.8rem;
  }
  .count-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.45rem 0;
    border-bottom: 1px solid #1e3a5f33;
  }
  .count-row:last-child { border-bottom: none; }
  .count-bar-wrap {
    flex: 1;
    margin: 0 1rem;
    background: #1e3a5f44;
    border-radius: 4px;
    height: 8px;
    overflow: hidden;
  }
  .count-bar {
    height: 8px;
    border-radius: 4px;
    background: linear-gradient(90deg, #00d4ff, #7b61ff);
    transition: width 0.4s ease;
  }
  .count-name { color: #a8c4e0; font-size: 0.82rem; min-width: 90px; }
  .count-num  {
    font-family: 'Orbitron', sans-serif;
    font-size: 0.85rem;
    color: #00d4ff;
    min-width: 28px;
    text-align: right;
  }
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
  hr { border-color: #1e3a5f; }
  .info-block {
    background: #111d35;
    border-left: 3px solid #00d4ff;
    padding: 0.8rem 1.2rem;
    border-radius: 0 8px 8px 0;
    font-size: 0.82rem;
    color: #a8c4e0;
    margin: 0.5rem 0;
  }
  .auto-badge {
    display: inline-block;
    padding: 0.3rem 1rem;
    border-radius: 20px;
    font-size: 0.7rem;
    letter-spacing: 2px;
    background: #00ff8822;
    border: 1px solid #00ff88;
    color: #00ff88;
    animation: pulse-green 2s infinite;
  }
  @keyframes pulse-green {
    0%, 100% { box-shadow: 0 0 6px #00ff8866; }
    50%       { box-shadow: 0 0 16px #00ff88bb; }
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
#  Load Model (cached)
# ─────────────────────────────────────────
@st.cache_resource
def load_model(model_name: str) -> YOLO:
    return YOLO(model_name)


# ─────────────────────────────────────────
#  Helper Functions
# ─────────────────────────────────────────
def detect_objects(
    yolo_model: YOLO,
    image: np.ndarray,
    conf_thresh: float,
) -> tuple[np.ndarray, list[dict]]:
    results = yolo_model(image, conf=conf_thresh, verbose=False)
    annotated = results[0].plot()
    detections: list[dict] = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label  = yolo_model.names[cls_id]
        conf   = float(box.conf[0])
        detections.append({"label": label, "confidence": conf})
    return annotated, detections


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def image_to_download_bytes(pil_img: Image.Image) -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


def base64_to_pil(b64_string: str) -> Image.Image | None:
    try:
        if "," in b64_string:
            b64_string = b64_string.split(",", 1)[1]
        img_bytes = base64.b64decode(b64_string)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        return None


def render_detections_html(detections: list[dict]) -> str:
    """Detection list with per-item confidence."""
    return "".join(
        f"<div class='detection-item'>"
        f"<span class='det-label'>● {d['label'].upper()}</span>"
        f"<span class='det-conf'>{d['confidence'] * 100:.1f}%</span>"
        f"</div>"
        for d in sorted(detections, key=lambda x: x["confidence"], reverse=True)
    )


def render_object_count_html(detections: list[dict]) -> str:
    """
    Bar chart–style count table grouped by object class.
    Shows: class name | bar | count
    """
    if not detections:
        return ""

    counts = Counter(d["label"] for d in detections)
    max_count = max(counts.values())
    rows = ""
    for label, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        pct = int(cnt / max_count * 100)
        rows += (
            f"<div class='count-row'>"
            f"  <span class='count-name'>● {label.upper()}</span>"
            f"  <div class='count-bar-wrap'>"
            f"    <div class='count-bar' style='width:{pct}%;'></div>"
            f"  </div>"
            f"  <span class='count-num'>{cnt}</span>"
            f"</div>"
        )
    return f"<div class='count-table'>{rows}</div>"


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
        help="n=nano (fastest), s=small, m=medium, l=large (most accurate)",
    )

    conf_threshold: float = st.slider(
        "Confidence Threshold",
        min_value=0.1, max_value=1.0, value=0.5, step=0.05,
        help="Skip detections below this value",
    )

    capture_interval: int = st.slider(
        "Auto-Capture Interval (seconds)",
        min_value=1, max_value=10, value=3, step=1,
        help="How often the live camera captures and analyses a new frame automatically",
    )

    st.markdown("---")
    st.markdown(
        f"""
        <div class='info-block'>
        <b>Jinsi ya kutumia:</b><br>
        📁 <b>Upload Image</b> – Chagua picha kutoka kompyuta yako<br><br>
        📷 <b>Live Camera</b> – Camera inaanza <b>automatically</b>!
        Frames zinachukuliwa kila <b>{capture_interval}s</b>
        na detection inafanywa bila kuhitaji kubonyeza kitu.
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
with st.spinner(f"⏳ Loading model {model_choice} ..."):
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
        pil_img = Image.open(uploaded_file)
        col_orig, col_result = st.columns(2, gap="medium")

        with col_orig:
            st.markdown(
                "<div class='metric-label' style='margin-bottom:0.5rem;'>ORIGINAL IMAGE</div>",
                unsafe_allow_html=True,
            )
            st.image(pil_img, use_container_width=True)
            w, h = pil_img.size
            st.markdown(
                f"<div style='font-size:0.75rem;color:#4a7fa5;text-align:center;'>"
                f"{w} × {h} px</div>",
                unsafe_allow_html=True,
            )

        with col_result:
            st.markdown(
                "<div class='metric-label' style='margin-bottom:0.5rem;'>DETECTION RESULT</div>",
                unsafe_allow_html=True,
            )
            with st.spinner("🔍 Running detection..."):
                t0 = time.time()
                bgr_img = pil_to_bgr(pil_img)
                annotated_bgr, detections = detect_objects(model, bgr_img, conf_threshold)
                elapsed = (time.time() - t0) * 1000
            result_img = bgr_to_pil(annotated_bgr)
            st.image(result_img, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(
                f"<div class='metric-card'><div class='metric-label'>Objects Detected</div>"
                f"<div class='metric-value'>{len(detections)}</div></div>",
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f"<div class='metric-card'><div class='metric-label'>Inference Time</div>"
                f"<div class='metric-value'>{elapsed:.0f}"
                f"<span style='font-size:1rem'>ms</span></div></div>",
                unsafe_allow_html=True,
            )
        with m3:
            avg_conf = (
                sum(d["confidence"] for d in detections) / len(detections) * 100
                if detections else 0.0
            )
            st.markdown(
                f"<div class='metric-card'><div class='metric-label'>Avg Confidence</div>"
                f"<div class='metric-value'>{avg_conf:.0f}"
                f"<span style='font-size:1rem'>%</span></div></div>",
                unsafe_allow_html=True,
            )

        if detections:
            st.markdown("<br>", unsafe_allow_html=True)

            det_col, cnt_col = st.columns(2, gap="medium")

            with det_col:
                st.markdown(
                    "<div class='metric-label' style='margin-bottom:0.5rem;'>"
                    "🎯 DETECTED OBJECTS (BY CONFIDENCE)</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='detection-box'>{render_detections_html(detections)}</div>",
                    unsafe_allow_html=True,
                )

            with cnt_col:
                st.markdown(
                    "<div class='metric-label' style='margin-bottom:0.5rem;'>"
                    "🔢 OBJECT COUNT PER CLASS</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(render_object_count_html(detections), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            label="⬇  DOWNLOAD RESULT IMAGE",
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
#  TAB 2 – AUTO Live Camera (No Button)
#  Camera starts immediately — no click needed
# ══════════════════════════════════════════
with tab2:
    st.markdown(
        "<div style='font-family:Orbitron,sans-serif;font-size:0.85rem;"
        "color:#4a7fa5;letter-spacing:2px;margin-bottom:0.5rem;'>"
        "AUTO LIVE CAMERA — INAANZA AUTOMATICALLY</div>",
        unsafe_allow_html=True,
    )

    # Status badge — always LIVE since no manual toggle
    frame_n = st.session_state.get("cam_frame_count", 0)
    st.markdown(
        f"<span class='auto-badge'>"
        f"🟢 AUTO-RUNNING · FRAME #{frame_n} · KILA {capture_interval}s</span>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # ── JavaScript: access webcam → draw one frame → return base64 ──
    JS_CAPTURE = f"""
    (async () => {{
        let stream;
        try {{
            stream = await navigator.mediaDevices.getUserMedia({{
                video: {{ width: 640, height: 480, facingMode: "environment" }}
            }});
        }} catch (err) {{
            return "CAMERA_ERROR:" + err.message;
        }}

        const video = document.createElement("video");
        video.srcObject   = stream;
        video.autoplay    = true;
        video.playsInline = true;
        video.muted       = true;
        video.style.display = "none";
        document.body.appendChild(video);

        await new Promise(resolve => {{ video.onloadedmetadata = resolve; }});
        await video.play();

        // Warm-up delay so exposure settles
        await new Promise(r => setTimeout(r, 700));

        const canvas = document.createElement("canvas");
        canvas.width  = video.videoWidth  || 640;
        canvas.height = video.videoHeight || 480;
        canvas.getContext("2d").drawImage(video, 0, 0);

        stream.getTracks().forEach(t => t.stop());
        video.remove();

        return canvas.toDataURL("image/jpeg", 0.85);
    }})()
    """

    # ── Session state init ──
    if "cam_frame_b64"    not in st.session_state:
        st.session_state["cam_frame_b64"]    = None
    if "cam_last_capture" not in st.session_state:
        st.session_state["cam_last_capture"] = 0.0
    if "cam_frame_count"  not in st.session_state:
        st.session_state["cam_frame_count"]  = 0

    # ── Auto-capture: always running, no button ──
    now             = time.time()
    time_since_last = now - st.session_state["cam_last_capture"]

    if time_since_last >= capture_interval:
        js_key  = f"webcam_{st.session_state['cam_frame_count']}"
        raw_b64 = streamlit_js_eval(js_code=JS_CAPTURE, key=js_key)

        if raw_b64 and isinstance(raw_b64, str):
            if raw_b64.startswith("CAMERA_ERROR"):
                error_msg = raw_b64.replace("CAMERA_ERROR:", "")
                st.error(
                    f"🚫 Ruhusa ya camera imezuiwa au camera haipatikani:\n{error_msg}\n\n"
                    "Hakikisha browser yako inaruhusu camera, kisha ufungue upya ukurasa."
                )
                st.stop()
            else:
                st.session_state["cam_frame_b64"]    = raw_b64
                st.session_state["cam_last_capture"] = time.time()
                st.session_state["cam_frame_count"] += 1

    # ── Display latest frame + detection ──
    if st.session_state["cam_frame_b64"]:
        pil_cam = base64_to_pil(st.session_state["cam_frame_b64"])

        if pil_cam is None:
            st.warning("⚠️ Frame haikuweza kusomwa. Ikisubiri frame inayofuata...")
        else:
            col_c1, col_c2 = st.columns(2, gap="medium")

            with col_c1:
                st.markdown(
                    "<div class='metric-label' style='margin-bottom:0.5rem;'>"
                    "📷 LIVE FRAME</div>",
                    unsafe_allow_html=True,
                )
                st.image(pil_cam, use_container_width=True)

            with col_c2:
                st.markdown(
                    "<div class='metric-label' style='margin-bottom:0.5rem;'>"
                    "🎯 DETECTION RESULT</div>",
                    unsafe_allow_html=True,
                )
                with st.spinner("🔍 Inachunguza vitu..."):
                    t0 = time.time()
                    bgr_cam = pil_to_bgr(pil_cam)
                    annotated_cam, detections_cam = detect_objects(
                        model, bgr_cam, conf_threshold
                    )
                    elapsed_cam = (time.time() - t0) * 1000

                result_cam = bgr_to_pil(annotated_cam)
                st.image(result_cam, use_container_width=True)

            # ── Metrics row ──
            st.markdown("<br>", unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<div class='metric-label'>Jumla ya Vitu</div>"
                    f"<div class='metric-value'>{len(detections_cam)}</div></div>",
                    unsafe_allow_html=True,
                )
            with m2:
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<div class='metric-label'>Muda wa Detection</div>"
                    f"<div class='metric-value'>{elapsed_cam:.0f}"
                    f"<span style='font-size:1rem'>ms</span></div></div>",
                    unsafe_allow_html=True,
                )
            with m3:
                avg_c = (
                    sum(d["confidence"] for d in detections_cam) / len(detections_cam) * 100
                    if detections_cam else 0.0
                )
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<div class='metric-label'>Avg Confidence</div>"
                    f"<div class='metric-value'>{avg_c:.0f}"
                    f"<span style='font-size:1rem'>%</span></div></div>",
                    unsafe_allow_html=True,
                )

            # ── Detection list + Count table ──
            if detections_cam:
                st.markdown("<br>", unsafe_allow_html=True)
                det_col, cnt_col = st.columns(2, gap="medium")

                with det_col:
                    st.markdown(
                        "<div class='metric-label' style='margin-bottom:0.5rem;'>"
                        "🎯 VITU VILIVYOGUNDULIWA (KWA UHAKIKA)</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<div class='detection-box'>"
                        f"{render_detections_html(detections_cam)}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                with cnt_col:
                    st.markdown(
                        "<div class='metric-label' style='margin-bottom:0.5rem;'>"
                        "🔢 IDADI KWA KILA AINA YA KITU</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        render_object_count_html(detections_cam),
                        unsafe_allow_html=True,
                    )

            # ── Download ──
            st.markdown("<br>", unsafe_allow_html=True)
            st.download_button(
                label="⬇  DOWNLOAD RESULT IMAGE",
                data=image_to_download_bytes(result_cam),
                file_name="camera_detection_result.png",
                mime="image/png",
                key=f"dl_cam_{st.session_state['cam_frame_count']}",
            )

    else:
        # First load — waiting for first frame
        st.markdown(
            """
            <div style='text-align:center;padding:3rem;color:#4a7fa5;'>
                <div style='font-size:3rem;margin-bottom:1rem;'>📷</div>
                <div style='font-family:Orbitron,sans-serif;letter-spacing:2px;font-size:0.8rem;'>
                    ⏳ IKISUBIRI RUHUSA YA CAMERA NA FRAME YA KWANZA...
                </div>
                <div style='font-size:0.75rem;margin-top:1rem;color:#3a5f7a;'>
                    Kama browser inakuuliza ruhusa ya camera — bonyeza ALLOW/RUHUSU
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Auto-rerun to keep capturing ──
    wait = max(0.0, capture_interval - (time.time() - st.session_state["cam_last_capture"]))
    time.sleep(wait)
    st.rerun()


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