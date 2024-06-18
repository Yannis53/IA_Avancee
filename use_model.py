import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os


def load_and_preprocess_image(img_path, img_size=(128, 128)):
	img = image.load_img(img_path, target_size=img_size)
	img_array = image.img_to_array(img)
	img_array = tf.expand_dims(img_array, 0) # Ajouter une dimension pour le batch
	img_array = img_array / 255.0 # Normalisation des pixels entre 0 et 1
	return img_array
	

def predict_image(model, img_array):
	predictions = model.predict(img_array)
	score = tf.nn.softmax(predictions[0])
	return score


new_images_dir = 'new_images'


class_names = train_dataset.class_names


for img_name in os.listdir(new_images_dir):
	img_path = os.path.join(new_images_dir, img_name)
	if os.path.isfile(img_path): # Vérifier si c'est un fichier
		img_array = load_and_preprocess_image(img_path)

		# Faire une prédiction
		score = predict_image(model, img_array)

		# Afficher les résultats
		predicted_class = class_names[np.argmax(score)]
		confidence = 100 * np.max(score)

		print(f"Image: {img_name}")
		print(f"Classée comme : {predicted_class} avec une confiance de {confidence:.2f}%\n")