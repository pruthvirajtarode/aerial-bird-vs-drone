import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model_utils import make_transfer_model, get_callbacks

def make_generators(data_dir, img_size=(224,224), batch_size=32):
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')

    train_aug = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.08,
        height_shift_range=0.08,
        shear_range=0.05,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    val_aug = ImageDataGenerator(rescale=1./255)

    train_gen = train_aug.flow_from_directory(train_dir, target_size=img_size, batch_size=batch_size, class_mode='binary')
    val_gen = val_aug.flow_from_directory(valid_dir, target_size=img_size, batch_size=batch_size, class_mode='binary', shuffle=False)
    return train_gen, val_gen

def train(data_dir='classification_dataset', epochs=20, batch_size=32):
    train_gen, val_gen = make_generators(data_dir, batch_size=batch_size)
    model = make_transfer_model(input_shape=(224,224,3), base_arch='efficientnet')
    cbs = get_callbacks(checkpoint_path='models/best_model.h5')
    history = model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=cbs)
    model.save('models/final_model.h5')

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='classification_dataset')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch', type=int, default=32)
    args = p.parse_args()
    train(data_dir=args.data, epochs=args.epochs, batch_size=args.batch)
