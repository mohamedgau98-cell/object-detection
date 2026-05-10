import streamlit as st
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
st.sidebar.header("Settings")
model=st.sidebar.selectbox("select model",["yolov8n","yolov8m","yolov8s"])
uploaded_file=st.sidebar.file_uploader("upload image",type=["jpg","png"])
confidence=st.sidebar.slider("Confidence threshold",1,100)
confidence2=st.sidebar.text_input("Confidence thresholds: ")
max_detection=st.sidebar.selectbox("Select maximum Detection ",[10,20,30,40,50])
run_button=st.sidebar.button("Run Detection")
st.markdown("""
            <style>
            div.stButton>button:first-child{
            color:red;
            background-color:blue;
            }
            </style>
            """,unsafe_allow_html=True)
col1,col2=st.columns(2)
with col1:
    if uploaded_file:
        st.markdown("Information")
        st.write(f"NAME: {uploaded_file.name}")
        st.write(f"type: {uploaded_file.type}")
        st.write(f"size: {uploaded_file.size}")
    else:
        print("Upload your image")
