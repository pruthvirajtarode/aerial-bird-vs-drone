import tensorflow as tf
from model_utils import make_transfer_model

# 1️⃣ Recreate the model architecture
model = make_transfer_model(input_shape=(224,224,3))

# 2️⃣ Load weights into the model
model.load_weights("models/final_model.h5")

# 3️⃣ Save new clean weights file
model.save_weights("models/final_model.weights.h5")

print("✔ Converted successfully: models/final_model.weights.h5 created!")
