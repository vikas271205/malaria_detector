import numpy as np
from tensorflow.keras.preprocessing import image
from backend.config import IMAGE_SIZE

def load_and_prepare(img_path):
    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    arr = image.img_to_array(img)
    arr = arr.astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)
