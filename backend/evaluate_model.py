import os, numpy as np, tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from keras.preprocessing.image import ImageDataGenerator

MODEL_PATH   = "malaria_model.h5"
TEST_DIR     = os.path.join("data", "cell_images")
IMAGE_SIZE   = (64, 64)
BATCH_SIZE   = 32

model = tf.keras.models.load_model(MODEL_PATH)

test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)

test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",  # same split we used before
    shuffle=False)

y_true = test_gen.classes
y_pred = model.predict(test_gen).ravel()
y_hat  = (y_pred >= 0.5).astype(int)

print(classification_report(y_true, y_hat, target_names=["Parasitized","Uninfected"]))
print("Confusion matrix:\n", confusion_matrix(y_true, y_hat))
