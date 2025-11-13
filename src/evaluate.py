import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

def evaluate_model(model_path='models/final_model.h5', test_dir='classification_dataset/test', batch_size=32):
    model = tf.keras.models.load_model(model_path)
    datagen = ImageDataGenerator(rescale=1./255)
    test_gen = datagen.flow_from_directory(test_dir, target_size=(224,224), batch_size=batch_size, class_mode='binary', shuffle=False)
    preds = model.predict(test_gen)
    y_pred = (preds.ravel() > 0.5).astype(int)
    y_true = test_gen.classes
    print(classification_report(y_true, y_pred, target_names=list(test_gen.class_indices.keys())))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=list(test_gen.class_indices.keys()), yticklabels=list(test_gen.class_indices.keys()))
    plt.savefig('demos/confusion_matrix.png')

if __name__ == '__main__':
    evaluate_model()
