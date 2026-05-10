from ultralytics import YOLO
import cv2
import streamlit as st
from PIL import Image

st.set_page_config(page_title="OBJECT DETECTOR", page_icon="🤖", layout="wide")
st.markdown("""
<div style="
            border:4px solid red;
            background-color:green;
            border-radius:30%;
            text-align:center;
            ">
            <h2>🔎 OBJECT DETECTOR</h2>
            <p>Upload image of an object and let AI to find the object for you</p>
            </div>
            """,unsafe_allow_html=True)

# SIDE BAR #
st.sidebar.header("⚙️ Control Panel")
model = st.sidebar.selectbox("Choose Model", ["yolov8n", "yolov8s", "yolov8m"])
file_uploaded = st.sidebar.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
confidence = st.sidebar.slider("Confidence Threshold", 0, 100)
max_det = st.sidebar.selectbox("Max Detections", [5, 10, 20])
run = st.sidebar.button("Run Detection")
st.markdown("""
<style>
        div.stButton>button:first-child{
            background-color:blue;
            color:white;
        }
        div.stSelectbox>div{
            border:2px solid red;
            border-radius:20px;
            background-color:blue;
            size:20px;
            color:white;
        }
</style>
""", unsafe_allow_html=True)

# Main #
col1, col2, col3 = st.columns(3, gap="large")

# For first column
with col1:
    st.subheader("Input Preview")
    if file_uploaded:
        st.info("File uploaded successfully")
        st.write(f"File Name: {file_uploaded.name}")
        st.write(f"File Type: {file_uploaded.type}")
        st.write(f"File size: {file_uploaded.size}")
    else:
        st.info("Upload The Image from the side bar")
        st.markdown("""
Instruction
#### 1. Upload an image
#### 2. Choose a model
#### 3. Set Confidence level
#### 4. Click Run Detection
""")

# Footer
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
    }
</style>

<div class="footer">
    Developed by BLECA<sup style="color:red">TM</sup>
</div>
""", unsafe_allow_html=True)

# for second column
with col2:
    st.markdown("Image Display Window")
    if file_uploaded:
        if run:
            # image=Image.open(file_uploaded)
            # st.image(image, caption="Uploaded image", use_container_width=True)
            model = YOLO(f"{model}.pt")
            cap = cv2.VideoCapture(1)  # open the camera
            if not cap.isOpened():
                print("Camera not opened successful")
                exit()
            while True:
                ret, frame = cap.read()
                result = model(frame)
                annoted_frame = result[0].plot()
                cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)
                cv2.imshow("Object Detection ", annoted_frame)
                cv2.waitKey(0)
                cap.release()
                cv2.destroyAllWindows()
                if not ret:
                    print("Failed to open poperly")

# for third column
with col3:
    st.markdown("For Analysis and Output")