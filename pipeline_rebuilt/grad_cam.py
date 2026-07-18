import argparse
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

from common import ensure_dirs
from config import DEFAULT_GRADCAM_IMAGE, GRADCAM_DIR, IMG_SIZE
from model_registry import get_model_spec


def parse_args():
    parser = argparse.ArgumentParser(description="Grad-CAM pentru modelele rebuilt.")
    parser.add_argument("--model", choices=["efficientnet", "resnet"], default="resnet")
    parser.add_argument("--image", default=str(DEFAULT_GRADCAM_IMAGE))
    parser.add_argument("--last-conv-layer", default=None)
    return parser.parse_args()


def load_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    return img, img_array


def find_last_conv_layer(model, preferred_name=None):
    if preferred_name:
        try:
            model.get_layer(preferred_name)
            return preferred_name
        except ValueError:
            pass

    for layer in reversed(model.layers):
        output_shape = getattr(layer, "output_shape", None)
        if output_shape is None and hasattr(layer, "output"):
            shape = getattr(layer.output, "shape", None)
            if shape is not None:
                output_shape = tuple(shape)
        if output_shape is not None and len(output_shape) == 4:
            return layer.name

    raise ValueError("Nu am gasit un strat convolutional 4D pentru Grad-CAM.")


def make_gradcam_heatmap(img_array, model, preprocess_input, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )

    input_tensor = tf.convert_to_tensor(np.expand_dims(img_array, axis=0))
    input_tensor = preprocess_input(input_tensor)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_tensor, training=False)
        class_channel = predictions[:, 0]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if float(max_val) > 0:
        heatmap /= max_val

    return heatmap.numpy(), float(tf.squeeze(predictions).numpy())


def save_gradcam(original_img, heatmap, output_path):
    original_img = original_img.resize(IMG_SIZE)
    original_array = np.array(original_img).astype("float32") / 255.0
    heatmap_resized = tf.image.resize(heatmap[..., np.newaxis], IMG_SIZE).numpy().squeeze()
    heatmap_colored = plt.get_cmap("jet")(heatmap_resized)[:, :, :3]
    overlay = np.clip((1 - 0.4) * original_array + 0.4 * heatmap_colored, 0, 1)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(original_array)
    plt.title("Imagine originala")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap_resized, cmap="jet")
    plt.title("Heatmap")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title("Grad-CAM")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    ensure_dirs(GRADCAM_DIR)

    spec = get_model_spec(args.model)
    last_conv = args.last_conv_layer or spec.last_conv_layer
    image_path = Path(args.image)

    model = keras.models.load_model(spec.path, compile=False)
    last_conv = find_last_conv_layer(model, last_conv)
    original_img, img_array = load_image(image_path)
    heatmap, prob = make_gradcam_heatmap(img_array, model, spec.preprocess_function, last_conv)

    label = "malignant" if prob >= spec.threshold else "benign"
    output_path = GRADCAM_DIR / f"{args.model}_{image_path.stem}_gradcam.png"
    save_gradcam(original_img, heatmap, output_path)

    print(f"Imagine: {image_path}")
    print(f"Layer Grad-CAM folosit: {last_conv}")
    print(f"Probabilitate malignant: {prob:.4f}")
    print(f"Prag folosit: {spec.threshold:.2f}")
    print(f"Predictie finala: {label}")
    print(f"Vizualizare salvata la: {output_path}")


if __name__ == "__main__":
    main()
