# app.py
# Upgraded Streamlit app for Aerial Bird vs Drone
# - Classifier (EfficientNet transfer + Grad-CAM)
# - YOLOv8 Detector (ultralytics) with annotated output
# - Camera / Upload options, improved UI, links & footer
# Paste this file into your project root and run `streamlit run app.py`

import streamlit as st
from PIL import Image
import numpy as np
import os, sys, io
import tempfile
import subprocess

# --- Path setup (adjust to your repo layout) ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)
    sys.path.append(PROJECT_ROOT)

# --- Try to import local helpers (predict, model_utils) ---
try:
    from predict import predict_image, make_gradcam_heatmap, save_and_display_gradcam
except Exception as e:
    predict_image = None
    make_gradcam_heatmap = None
    save_and_display_gradcam = None
    _predict_import_error = e

try:
    from model_utils import make_transfer_model
except Exception as e:
    make_transfer_model = None
    _mu_import_error = e

# --- Try to import ultralytics YOLO ---
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except Exception:
    YOLO = None
    ULTRALYTICS_AVAILABLE = False

# --- Streamlit page config ---
st.set_page_config(page_title="Aerial Intelligence Suite", page_icon="🚁", layout="wide")

# --- Styling (dark friendly) ---
st.markdown(
    """
    <style>
    body {background-color: #0b1020;}
    .header {background: linear-gradient(90deg,#0ea5a4,#6366f1); color: white; padding:18px; border-radius:12px;}
    .card {background:#101420; color:#e6eef8; padding:16px; border-radius:10px; box-shadow:0 6px 24px rgba(0,0,0,0.5);}
    .small {color:#b7c0d6}
    .link {color:#ffd166}
    </style>
    """, unsafe_allow_html=True
)

# --- Helper: save uploaded / captured image to disk and return path ---
def save_temp_image(uploaded_file_or_pil, suffix=".png"):
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    if isinstance(uploaded_file_or_pil, Image.Image):
        uploaded_file_or_pil.save(tf.name)
    else:
        # file-like
        data = uploaded_file_or_pil.read()
        tf.write(data)
        tf.flush()
    return tf.name

# --- Load classifier model weights path (weights file expected) ---
WEIGHTS_PATH = os.path.join("models", "final_model.weights.h5")
CLASSIFIER_WEIGHTS_EXIST = os.path.exists(WEIGHTS_PATH)

# --- Try to create model object if helper available ---
classifier_model = None
if make_transfer_model is not None and CLASSIFIER_WEIGHTS_EXIST:
    try:
        classifier_model = make_transfer_model()
        classifier_model.load_weights(WEIGHTS_PATH)
    except Exception as e:
        classifier_model = None
        _load_weights_error = e

# --- Sidebar navigation ---
st.sidebar.title("Aerial Intelligence Suite")
choice = st.sidebar.radio("Choose section", ["Home", "Classifier", "YOLO Detector", "About"])

# --------------------- HOME ---------------------
if choice == "Home":
    st.markdown('<div class="header"><h1>🚁 Aerial Intelligence Suite — Bird vs Drone</h1></div>', unsafe_allow_html=True)
    st.markdown("")
    st.markdown(
        """
        <div class="card">
        <h3>What this app does</h3>
        <ul>
          <li>Classify aerial images as <b>Bird</b> or <b>Drone</b> (EfficientNet transfer learning).</li>
          <li>Explain classifier decisions using <b>Grad-CAM</b>.</li>
          <li>Detect objects with <b>YOLOv8</b> (optional).</li>
        </ul>
        <p class="small">Use the sidebar to navigate. If a model or weights are missing, the app will tell you exactly what to add.</p>
        </div>
        """, unsafe_allow_html=True
    )

# --------------------- CLASSIFIER ---------------------
elif choice == "Classifier":
    st.markdown('<div class="header"><h2>🔍 Classifier + Grad-CAM</h2></div>', unsafe_allow_html=True)
    st.write("Upload an image or take a picture. The app will run the classifier then show Grad-CAM heatmap.")

    col1, col2 = st.columns([2,1])

    with col1:
        uploaded = st.file_uploader("Upload image (jpg, png)", type=["jpg","jpeg","png"])
        use_camera = st.button("Use camera (webcam)")  # streamlit's camera_input exists but many environments don't support it
        img_path = None

        if use_camera:
            # use streamlit's camera_input if available
            try:
                cam_img = st.camera_input("Take a photo")
                if cam_img:
                    img_path = save_temp_image(cam_img, suffix=".png")
            except Exception:
                st.warning("Camera input is not available in this environment.")
        elif uploaded:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, caption="Uploaded image", use_column_width=True)
            img_path = save_temp_image(img, suffix=".png")

        if img_path:
            st.markdown("---")
            # Predict button
            if st.button("Predict & Explain (Grad-CAM)"):
                # Check prerequisites
                if predict_image is None or make_gradcam_heatmap is None:
                    st.error("Local prediction utilities (`src/predict.py`) are missing or failed to import.")
                    if '_predict_import_error' in globals():
                        st.info(f"Import error: {_predict_import_error}")
                elif classifier_model is None:
                    if not CLASSIFIER_WEIGHTS_EXIST:
                        st.error(f"Classifier weights not found at `{WEIGHTS_PATH}`. Save your trained weights as that file.")
                    else:
                        st.error("Model creation or weight loading failed. See terminal for details.")
                        if '_load_weights_error' in globals():
                            st.info(f"Load error: {_load_weights_error}")
                else:
                    # run classifier
                    try:
                        label, conf = predict_image(WEIGHTS_PATH, img_path)  # your predict expects (model_path, img_path)
                        st.success(f"Prediction: **{label.upper()}** — Confidence: **{conf*100:.2f}%**")
                    except Exception as exc:
                        st.error("Prediction failed: " + str(exc))

                    # grad-cam
                    try:
                        # prepare image arr for grad-cam (224x224)
                        from tensorflow.keras.preprocessing import image as kimage
                        arr_img = kimage.load_img(img_path, target_size=(224,224))
                        arr = kimage.img_to_array(arr_img)/255.0
                        arr = np.expand_dims(arr, 0)

                        # find last conv layer name
                        last_conv = None
                        for layer in reversed(classifier_model.layers):
                            if 'conv' in layer.name:
                                last_conv = layer.name
                                break

                        if last_conv is None:
                            st.warning("No conv layer found on the classifier — Grad-CAM not available.")
                        else:
                            heatmap = make_gradcam_heatmap(arr, classifier_model, last_conv)
                            out_cam = os.path.join("demos", "gradcam_demo.jpg")
                            os.makedirs("demos", exist_ok=True)
                            save_and_display_gradcam(img_path, heatmap, cam_path=out_cam)
                            st.image(out_cam, caption="Grad-CAM", use_column_width=True)
                    except Exception as exc:
                        st.error("Grad-CAM failed: " + str(exc))

# --------------------- YOLO DETECTOR ---------------------
elif choice == "YOLO Detector":
    st.markdown('<div class="header"><h2>🎯 YOLOv8 Detector</h2></div>', unsafe_allow_html=True)
    st.write("Use YOLOv8 to detect objects. This requires `ultralytics` installed (it is) and will download the model weights if necessary.")

    col1, col2 = st.columns([2,1])
    with col1:
        y_upload = st.file_uploader("Upload image for detection", type=["jpg","jpeg","png"])
        run_yolo = st.button("Run YOLOv8 Detection")

        if y_upload:
            st.image(Image.open(y_upload).convert("RGB"), caption="Image to detect", use_column_width=True)
            y_path = save_temp_image(y_upload, suffix=".jpg")
        else:
            y_path = None

        if run_yolo:
            if not ULTRALYTICS_AVAILABLE:
                st.error("`ultralytics` not available in this environment. Install via `pip install ultralytics`.")
            elif y_path is None:
                st.warning("Upload an image first.")
            else:
                with st.spinner("Running YOLOv8..."):
                    try:
                        # Use a small pretrained weight by default (yolov8n)
                        # If you have a custom model, set model_path variable to it.
                        model_path = "yolov8n.pt"
                        ymodel = YOLO(model_path)
                        results = ymodel.predict(source=y_path, imgsz=640, conf=0.25, save=False)
                        # results is a list-like; take first
                        r = results[0]
                        # create annotated image
                        annotated = r.plot()  # returns np array with annotations
                        # save annotated
                        out_annot = os.path.join("demos", "yolo_annotated.jpg")
                        os.makedirs("demos", exist_ok=True)
                        Image.fromarray(annotated).save(out_annot)
                        st.image(out_annot, caption="YOLOv8 Detection", use_column_width=True)
                        # show bounding boxes + classes in a small table
                        boxes = []
                        for box in r.boxes:
                            cls = int(box.cls.cpu().numpy()) if hasattr(box, "cls") else None
                            conf = float(box.conf.cpu().numpy()) if hasattr(box, "conf") else None
                            xyxy = box.xyxy.cpu().numpy().tolist() if hasattr(box, "xyxy") else None
                            boxes.append({"class": cls, "conf": conf, "xyxy": xyxy})
                        if boxes:
                            st.markdown("**Detections (raw):**")
                            st.write(boxes)
                    except Exception as exc:
                        st.error("YOLO inference failed: " + str(exc))

    with col2:
        st.markdown("<div class='card'><h4>Notes</h4><ul>"
                    "<li>If `yolov8n.pt` is not present it will be downloaded automatically by ultralytics.</li>"
                    "<li>For custom detector, put your `.pt` model in project root and change <code>model_path</code>.</li>"
                    "</ul></div>", unsafe_allow_html=True)

# --------------------- ABOUT ---------------------
elif choice == "About":
    st.markdown('<div class="header"><h2>About & Links</h2></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
      <h3>Pruthviraj Tarode</h3>
      <p>Machine Learning • Deep Learning • Computer Vision</p>
      <p>
        🔗 <a class="link" href="https://www.linkedin.com/in/pruthviraj-tarode-616ab1258/" target="_blank">LinkedIn</a> &nbsp;|
        💻 <a class="link" href="https://github.com/pruthvirajtarode" target="_blank">GitHub</a> &nbsp;|
        🌐 <a class="link" href="https://pruthvirajtarode.github.io/" target="_blank">Portfolio</a>
      </p>
      <h4>Helpful tips</h4>
      <ul>
        <li>Classifier expects weights: <code>models/final_model.weights.h5</code> — save model.save_weights(...) to that path after training.</li>
        <li>If you want the app to load a full Keras .h5 (model + weights) instead, replace the classifier load & predict calls with <code>tf.keras.models.load_model</code>.</li>
        <li>YOLO: put your custom <code>best.pt</code> in project root and change <code>model_path</code> variable inside the YOLO block.</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)

# --- Footer ---
st.markdown(
    """
    <div style="margin-top:30px; padding:12px; text-align:center; color:#dbeafe;">
      🚀 Designed & developed by <b>Pruthviraj Tarode</b> — Turning aerial images into actionable intelligence.
    </div>
    """, unsafe_allow_html=True
)
