import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import cv2, os

def load_img(path, target=(224,224)):
    img = image.load_img(path, target_size=target)
    arr = image.img_to_array(img) / 255.0
    return np.expand_dims(arr,0), img

def predict_image(model_path, img_path):
    model = load_model(model_path)
    arr, pil = load_img(img_path)
    p = model.predict(arr)[0][0]
    label = 'drone' if p > 0.5 else 'bird'
    conf = p if p>0.5 else 1-p
    return label, float(conf)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names):
    import tensorflow as tf
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-9)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path='demos/gradcam.jpg', alpha=0.4):
    import numpy as np
    from PIL import Image
    import cv2, os
    img = Image.open(img_path).convert('RGB')
    heatmap = np.uint8(255 * heatmap)
    heatmap = Image.fromarray(heatmap).resize(img.size, resample=Image.BILINEAR)
    heatmap = np.array(heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(np.array(img), 1-alpha, heatmap, alpha, 0)
    os.makedirs(os.path.dirname(cam_path), exist_ok=True)
    cv2.imwrite(cam_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))
    return cam_path
