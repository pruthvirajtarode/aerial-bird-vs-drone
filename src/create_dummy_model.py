# Creates a tiny demo Keras model and saves it to models/final_model.h5
# Requires TensorFlow to be installed in the environment.

import numpy as np
from tensorflow.keras import layers, models
import os

def create_and_save_dummy_model(path='models/final_model.h5'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    inp = layers.Input(shape=(224,224,3))
    x = layers.Rescaling(1./255)(inp)
    x = layers.Conv2D(8,3,activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(16,3,activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(16, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inp, out)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    dummy = np.random.rand(1,224,224,3).astype('float32')
    model.predict(dummy)
    model.save(path)
    print('Saved demo model to', path)

if __name__ == '__main__':
    create_and_save_dummy_model()
