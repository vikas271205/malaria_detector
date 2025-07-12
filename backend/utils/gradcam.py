import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import os

def get_gradcam_heatmap(model, img_array, last_conv_layer_name="conv2d_2"):
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]  # Adjust index for your binary class

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_overlay_gradcam(original_img_path, heatmap, output_path="gradcam.jpg", alpha=0.5):
    img = cv2.imread(original_img_path)
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    overlayed_img = cv2.addWeighted(heatmap_colored, alpha, img, 1 - alpha, 0)

    cv2.imwrite(output_path, overlayed_img)
    return output_path
