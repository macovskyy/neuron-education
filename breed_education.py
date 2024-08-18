import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# Шлях до даних
base_dir = 'data'

# Генератори даних
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Генератор для навчальної вибірки
train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Генератор для валідаційної вибірки
validation_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Модель на основі MobileNetV2
base_model = MobileNetV2(
    weights='imagenet', include_top=False, input_shape=(128, 128, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Глобальне середнє пулінгування
x = Dense(1024, activation='relu')(x)  # Повнозв'язковий шар з активацією ReLU
predictions = Dense(len(train_generator.class_indices),
                    # Повнозв'язковий шар для класифікації
                    activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Замороження ваг базової моделі (MobileNetV2) для навчання тільки нових шарів
for layer in base_model.layers:
    layer.trainable = False

# Компіляція моделі
model.compile(optimizer=Adam(), loss='categorical_crossentropy',
              metrics=['accuracy'])

# Навчання моделі
model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# Збереження моделі
model.save('cat_breed_classifier.keras')

# Збереження класів у файл JSON
with open('class_indices.json', 'w') as f:
    json.dump(train_generator.class_indices, f)
