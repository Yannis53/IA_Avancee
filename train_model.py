import os
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np


train_dataset = tf.keras.utils.image_dataset_from_directory(
# chemin vers le dossier parent contenant les sous-dossiers
'data',
# fraction des données à utiliser pour la validation
validation_split=0.2,
subset="training",
seed=123, # pour reproductibilité
# taille des images
image_size=(128, 128),
batch_size=16)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
'data',
validation_split=0.2,
subset="validation",
seed=123,
image_size=(128, 128),
batch_size=16)


normalization_layer = layers.Rescaling(1./255)
normalized_train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
normalized_validation_dataset = validation_dataset.map(lambda x, y: (normalization_layer(x), y))


model = models.Sequential([
layers.InputLayer(shape=(128, 128, 3)),
layers.Conv2D(32, (3, 3), activation='relu'),
layers.MaxPooling2D((2, 2)),
layers.Conv2D(64, (3, 3), activation='relu'),
layers.MaxPooling2D((2, 2)),
layers.Conv2D(128, (3, 3), activation='relu'),
layers.MaxPooling2D((2, 2)),
layers.Flatten(),
layers.Dense(128, activation='relu'),
layers.Dense(7)
])


model.compile(optimizer='adam',
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
metrics=['accuracy'])


history = model.fit(
normalized_train_dataset,
validation_data=normalized_validation_dataset,
# ajustez en fonction de vos besoins
epochs=50
)


loss, accuracy = model.evaluate(normalized_validation_dataset); print(f'Accuracy: {accuracy:.2f}')


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
# Assurez-vous que les dimensions correspondent
epochs_range = range(len(acc))
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()