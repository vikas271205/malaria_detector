from tensorflow.keras.preprocessing import image
import numpy as np

def load_and_prepare(img_path, target_size=(64, 64)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)  # Shape: (1, 64, 64, 3)
