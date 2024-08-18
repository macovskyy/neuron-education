import tensorflow as tf
import json
from tensorflow.keras.preprocessing import image
import numpy as np

# Завантаження збереженої моделі
model = tf.keras.models.load_model('cat_breed_classifier.keras')

# Завантаження мапи класів з файлу JSON
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)

# Перетворення мапи класів у зворотну форму (індекс до назви)
class_indices = {v: k for k, v in class_indices.items()}


def prepare_image(img_path):
    # Завантаження зображення та зміна його розміру
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)  # Перетворення зображення у масив
    img_array = np.expand_dims(img_array, axis=0)  # Додавання осі для пакета
    img_array /= 255.0  # Нормалізація значень пікселів
    return img_array


def predict_breed(img_path):
    img = prepare_image(img_path)  # Підготовка зображення
    predictions = model.predict(img)  # Прогнозування класу
    # Отримання індексу класу з найбільшою ймовірністю
    predicted_class_index = np.argmax(predictions)
    # Отримання назви класу за індексом
    predicted_class = class_indices[predicted_class_index]
    return predicted_class


# Введення шляху до зображення користувачем
img_path = input('path: ')
breed = predict_breed(img_path)  # Прогнозування породи
print(f'The predicted breed is: {breed}')  # Виведення результату
