import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


def make_transfer_model(input_shape=(224, 224, 3)):

    # Load EfficientNetB0 (feature extractor)
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )

    base.trainable = False
    preprocess = tf.keras.applications.efficientnet.preprocess_input

    # Build full model
    inputs = layers.Input(shape=input_shape)
    x = preprocess(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)

    # No metrics → No tensor JSON serialization error
    model.compile(
        optimizer=optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[]   # <--- IMPORTANT FIX
    )

    return model
