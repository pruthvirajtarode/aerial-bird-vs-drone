import tensorflow as tf

model = tf.keras.models.load_model("models/final_model.h5")
model.save_weights("models/final_model.weights.h5")

print("✔ Weights saved to models/final_model.weights.h5")
