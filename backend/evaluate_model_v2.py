import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# ================= CONFIG =================
DATA_ROOT = "data"
TEST_DIR = os.path.join(DATA_ROOT, "test")

MODEL_PATH = "malaria_model_v2.keras"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
# =========================================

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Test data generator (NO augmentation)
test_datagen = ImageDataGenerator(rescale=1./255)

test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

print("Class indices:", test_gen.class_indices)

# Ground truth
y_true = test_gen.classes

# Predictions
y_prob = model.predict(test_gen).ravel()
y_pred = (y_prob >= 0.5).astype(int)

# Metrics
print("\nClassification Report (TEST SET):")
print(classification_report(
    y_true,
    y_pred,
    target_names=list(test_gen.class_indices.keys())
))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

auc = roc_auc_score(y_true, y_prob)
print(f"\nROC-AUC (test set): {auc:.4f}")
