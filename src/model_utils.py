import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import EfficientNetB0

def make_transfer_model(input_shape=(224,224,3), base_arch='efficientnet'):
    base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    preprocess = tf.keras.applications.efficientnet.preprocess_input
    base.trainable = False
    inputs = layers.Input(shape=input_shape)
    x = preprocess(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def get_callbacks(checkpoint_path='models/best_model.h5', patience=6):
    es = callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    mc = callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True)
    rl = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    return [es, mc, rl]
