import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model_utils import make_transfer_model


def make_generators(data_dir, img_size=(224, 224), batch_size=32):

    train_dir = os.path.join(data_dir, "train")
    valid_dir = os.path.join(data_dir, "val")   # <-- using 'val' folder

    print(f"\n📁 Training directory: {train_dir}")
    print(f"📁 Validation directory: {valid_dir}")

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.08,
        height_shift_range=0.08,
        shear_range=0.05,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary"
    )

    val_gen = val_datagen.flow_from_directory(
        valid_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False
    )

    return train_gen, val_gen


def train(data_dir, epochs=20, batch_size=32):
    print("\n📂 Loading dataset...")
    train_gen, val_gen = make_generators(data_dir, batch_size=batch_size)

    print("\n🧠 Building model...")
    model = make_transfer_model()

    print("\n🚀 Training started...\n")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs
    )

    print("\n💾 Saving weights ONLY (safe method)...")
    os.makedirs("models", exist_ok=True)

    weights_path = "models/final_model.weights.h5"
    model.save_weights(weights_path)

    print("\n=====================================")
    print("   🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print("=====================================\n")
    print(f"✔ Weights saved at: {weights_path}")
    print("✔ Use this to load weights in Streamlit:\n")
    print("    from model_utils import make_transfer_model")
    print("    model = make_transfer_model()")
    print("    model.load_weights('models/final_model.weights.h5')")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="classification_dataset")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()

    train(args.data, args.epochs, args.batch)
