import streamlit as st
from PIL import Image
import numpy as np
import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
src_path = os.path.join(project_root, 'src')
sys.path.append(project_root)
sys.path.append(src_path)

from src.predict import predict_image, make_gradcam_heatmap, save_and_display_gradcam

st.set_page_config(page_title='Aerial Bird vs Drone', page_icon='🛰️', layout='wide')

st.markdown("""<style>
.header { background: linear-gradient(90deg,#0ea5a4,#6366f1); color: white; padding:20px; border-radius:12px; box-shadow:0 12px 40px rgba(0,0,0,0.06); }
.card { background: white; border-radius:12px; padding:16px; box-shadow:0 8px 24px rgba(0,0,0,0.06); }
.small { color: #455A64; }
.btn { background:#6366f1; color:white; padding:8px 12px; border-radius:8px }
</style>""", unsafe_allow_html=True)

st.markdown("""<div class='header'><h1> Aerial Object Classifier - Bird vs Drone</h1><p class='small'>Interactive demo with Grad-CAM and optional YOLOv8 detection.</p></div>""", unsafe_allow_html=True)

col1, col2 = st.columns([2,1])
with col1:
    uploaded = st.file_uploader('Upload an aerial image', type=['jpg','jpeg','png'])
    if uploaded:
        img = Image.open(uploaded).convert('RGB')
        st.image(img, caption='Uploaded image', use_column_width=True)
        tmp = 'tmp_upload.png'
        img.save(tmp)
        st.info('Model loaded from models/final_model.h5 (create demo model with src/create_dummy_model.py)')
        if st.button('Predict & Explain (Grad-CAM)'):
            if not os.path.exists('models/final_model.h5'):
                st.error('models/final_model.h5 not found. Run src/create_dummy_model.py to create it.')
            else:
                try:
                    label, conf = predict_image('models/final_model.h5', tmp)
                    st.success(f'Prediction: **{label.upper()}** - Confidence: **{conf*100:.2f}%**')
                except Exception as e:
                    st.error('Prediction failed: ' + str(e))
                # Grad-CAM
                try:
                    import tensorflow as tf
                    model = tf.keras.models.load_model('models/final_model.h5')
                    last_conv = None
                    for layer in reversed(model.layers):
                        if 'conv' in layer.name:
                            last_conv = layer.name
                            break
                    if last_conv:
                        from tensorflow.keras.preprocessing import image as keras_image
                        imgk = keras_image.load_img(tmp, target_size=(224,224))
                        arr = keras_image.img_to_array(imgk)/255.0
                        arr = np.expand_dims(arr,0)
                        heatmap = make_gradcam_heatmap(arr, model, last_conv, [])
                        cam = save_and_display_gradcam(tmp, heatmap, cam_path='demos/gradcam_demo.jpg')
                        st.image(cam, caption='Grad-CAM', use_column_width=True)
                    else:
                        st.warning('No conv layer found for Grad-CAM.')
                except Exception as e:
                    st.warning('Grad-CAM failed: ' + str(e))

with col2:
    st.markdown('<div class="card"><h3>Quick actions & tips</h3><ul><li>Replace <code>models/final_model.h5</code> with your trained model.</li><li>Train using <code>src/train_classifier.py</code>.</li><li>See <code>yolov8_instructions.md</code> for detection.</li></ul></div>', unsafe_allow_html=True)
    if st.button('Show demo script (2-3 min)'):
        st.code('1. Quick pitch (15s)\\n2. Show dataset samples (15s)\\n3. Model & training highlights (20s)\\n4. Live demo: upload -> predict -> Grad-CAM (60s)\\n5. Results & closing (20s)')

st.markdown('---')
st.markdown('Prepared by **Pruthviraj Tarode** - demo-ready for internship interviews.')
