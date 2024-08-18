import tensorflow as tf
import json
from tensorflow.keras.preprocessing import image
import numpy as np

model = tf.keras.models.load_model('cat_breed_classifier.keras')

with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)

class_indices = {v: k for k, v in class_indices.items()}



def prepare_image(img_path):
    # Измените размер, если требуется
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Добавление оси для пакета
    img_array /= 255.0  # Нормализация
    return img_array



def predict_breed(img_path):
    img = prepare_image(img_path)
    predictions = model.predict(img)
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_indices[predicted_class_index]
    return predicted_class


img_path = input('path:' ) 
breed = predict_breed(img_path)
print(f'The predicted breed is): {breed}')
