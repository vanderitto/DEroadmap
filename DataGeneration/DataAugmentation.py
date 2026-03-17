import os, warnings
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers
import kagglehub

# Pobieranie danych
path = kagglehub.dataset_download("ryanholbrook/car-or-truck")
print("Path to dataset files:", path)

# Ustawienia ziarna (reproducibility)
def set_seed(seed=31415):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
set_seed()

# Przygotowanie ścieżek (Naprawione pod Windows)
train_dir = os.path.join(path, 'train')
valid_dir = os.path.join(path, 'valid')

# Ładowanie surowych zbiorów
ds_train_ = image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='binary',
    image_size=[128, 128],
    interpolation='nearest',
    batch_size=64,
    shuffle=True,
)

ds_valid_ = image_dataset_from_directory(
    valid_dir,
    labels='inferred',
    label_mode='binary',
    image_size=[128, 128],
    interpolation='nearest',
    batch_size=64,
    shuffle=False,
)

# Definicje pomocnicze
AUTOTUNE = tf.data.experimental.AUTOTUNE

def convert_to_float(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image, label

# DATA AUGMENTATION (Tworzenie wirtualnych danych)
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),    # Odbicie lewo-prawo
    layers.RandomRotation(0.1),         # Obrót o max 10%
    layers.RandomContrast(0.9),         # Zmiana jasności/kontrastu
])

# Budowa finalnych potoków (Pipeline)
ds_train = (
    ds_train_
    .map(lambda x, y: (data_augmentation(x, training=True), y)) 
    .map(convert_to_float)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)

ds_valid = (
    ds_valid_
    .map(convert_to_float)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)

# Zobaczmy, co "widzi" sieć po augmentacji
plt.figure(figsize=(10, 10))
for images, _ in ds_train.take(1): # Pobierz jedną partię (batch)
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy())
        plt.axis("off")
plt.suptitle("Przykłady po Data Augmentation", fontsize=16)
plt.show()