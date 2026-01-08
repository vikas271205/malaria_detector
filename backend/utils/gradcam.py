import numpy as np
import tensorflow as tf
import cv2

def _find_last_conv(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found.")

def get_gradcam_heatmap(model, img_array, target_class=None):
    last_conv = _find_last_conv(model)
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array, training=False)
        if target_class is None:
            target_class = tf.cast(preds >= 0.5, tf.int32)
        loss = preds[:, 0] if target_class == 1 else (1.0 - preds[:, 0])

    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heatmap = tf.reduce_sum(conv_out * pooled, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    denom = tf.reduce_max(heatmap)
    heatmap = heatmap / tf.maximum(denom, tf.keras.backend.epsilon())
    return heatmap.numpy()

def save_and_overlay_gradcam(original_img_path, heatmap, out_path, alpha=0.5):
    img = cv2.imread(original_img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heatmap, alpha, img, 1 - alpha, 0)
    cv2.imwrite(out_path, overlay)
    return out_path
