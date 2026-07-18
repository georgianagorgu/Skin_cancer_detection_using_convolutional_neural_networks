import argparse
from pathlib import Path

import keras
import matplotlib
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import ensure_dirs
from config import DEFAULT_GRADCAM_IMAGE, IMG_SIZE, XAI_DIR
from model_registry import get_model_spec


def load_image_array(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    return img, img_array


def find_last_conv_layer(model, preferred_name):
    try:
        model.get_layer(preferred_name)
        return preferred_name
    except ValueError:
        pass

    for layer in reversed(model.layers):
        shape = getattr(getattr(layer, "output", None), "shape", None)
        if shape is not None and len(shape) == 4:
            return layer.name
    raise ValueError("Nu am gasit un strat convolutional 4D pentru XAI.")


def normalize_heatmap(heatmap):
    heatmap = np.maximum(heatmap, 0)
    max_value = np.max(heatmap)
    if max_value > 0:
        heatmap = heatmap / max_value
    return heatmap


def grad_model_for(model, last_conv_layer):
    return tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer).output, model.output],
    )


def grad_cam(img_array, model, preprocess_input, last_conv_layer):
    g_model = grad_model_for(model, last_conv_layer)
    input_tensor = preprocess_input(tf.convert_to_tensor(np.expand_dims(img_array, axis=0)))

    with tf.GradientTape() as tape:
        conv_outputs, predictions = g_model(input_tensor, training=False)
        score = predictions[:, 0]

    grads = tape.gradient(score, conv_outputs)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(conv_outputs[0] * weights, axis=-1)
    return normalize_heatmap(heatmap.numpy()), float(tf.squeeze(predictions).numpy())


def grad_cam_plus_plus(img_array, model, preprocess_input, last_conv_layer):
    g_model = grad_model_for(model, last_conv_layer)
    input_tensor = preprocess_input(tf.convert_to_tensor(np.expand_dims(img_array, axis=0)))

    with tf.GradientTape() as tape:
        conv_outputs, predictions = g_model(input_tensor, training=False)
        score = predictions[:, 0]

    grads = tape.gradient(score, conv_outputs)[0]
    conv_outputs = conv_outputs[0]
    first = tf.exp(score)[0] * grads
    second = tf.exp(score)[0] * tf.square(grads)
    third = tf.exp(score)[0] * tf.pow(grads, 3)

    denominator = 2.0 * second + tf.reduce_sum(conv_outputs * third, axis=(0, 1), keepdims=True)
    denominator = tf.where(denominator != 0.0, denominator, tf.ones_like(denominator))
    alpha = second / denominator
    weights = tf.reduce_sum(alpha * tf.maximum(first, 0.0), axis=(0, 1))
    heatmap = tf.reduce_sum(weights * conv_outputs, axis=-1)
    return normalize_heatmap(heatmap.numpy()), float(tf.squeeze(predictions).numpy())


def score_cam(img_array, model, preprocess_input, last_conv_layer, max_maps=32):
    g_model = grad_model_for(model, last_conv_layer)
    input_batch = preprocess_input(np.expand_dims(img_array.copy(), axis=0))
    conv_outputs, predictions = g_model(input_batch, training=False)
    activations = conv_outputs[0].numpy()
    base_score = float(tf.squeeze(predictions).numpy())

    channel_scores = activations.mean(axis=(0, 1))
    selected_channels = np.argsort(channel_scores)[-max_maps:]
    weights = []
    maps = []

    for channel in selected_channels:
        activation_map = activations[:, :, channel]
        resized = tf.image.resize(activation_map[..., np.newaxis], IMG_SIZE).numpy().squeeze()
        resized = normalize_heatmap(resized)
        if np.max(resized) <= 0:
            continue
        masked_img = img_array * resized[..., np.newaxis]
        masked_batch = preprocess_input(np.expand_dims(masked_img.copy(), axis=0))
        score = float(np.squeeze(model.predict(masked_batch, verbose=0)))
        weights.append(score)
        maps.append(resized)

    if not maps:
        return np.zeros(IMG_SIZE, dtype="float32"), base_score

    weights = np.maximum(np.asarray(weights), 0)
    if np.sum(weights) > 0:
        weights = weights / np.sum(weights)
    heatmap = np.sum(np.asarray(maps) * weights[:, np.newaxis, np.newaxis], axis=0)
    return normalize_heatmap(heatmap), base_score


def overlay_heatmap(original_img, heatmap):
    original_array = np.array(original_img.resize(IMG_SIZE)).astype("float32") / 255.0
    heatmap_resized = tf.image.resize(heatmap[..., np.newaxis], IMG_SIZE).numpy().squeeze()
    heatmap_colored = plt.get_cmap("jet")(heatmap_resized)[:, :, :3]
    return np.clip(0.6 * original_array + 0.4 * heatmap_colored, 0, 1)


def save_xai_figure(original_img, heatmaps, output_path, prob, threshold):
    fig, axes = plt.subplots(1, len(heatmaps) + 1, figsize=(4 * (len(heatmaps) + 1), 4))
    original_array = np.array(original_img.resize(IMG_SIZE)).astype("float32") / 255.0
    label = "malignant" if prob >= threshold else "benign"

    axes[0].imshow(original_array)
    axes[0].set_title(f"Original\npred: {label}, P(mal)={prob:.3f}")
    axes[0].axis("off")

    for ax, (name, heatmap) in zip(axes[1:], heatmaps.items()):
        ax.imshow(overlay_heatmap(original_img, heatmap))
        ax.set_title(name)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="XAI: Grad-CAM, Grad-CAM++ si Score-CAM.")
    parser.add_argument("--model", choices=["efficientnet", "resnet"], default="resnet")
    parser.add_argument("--image", default=str(DEFAULT_GRADCAM_IMAGE))
    parser.add_argument("--last-conv-layer", default=None)
    parser.add_argument("--scorecam-maps", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dirs(XAI_DIR)

    spec = get_model_spec(args.model)
    model = keras.models.load_model(spec.path, compile=False)
    last_conv = find_last_conv_layer(model, args.last_conv_layer or spec.last_conv_layer)
    image_path = Path(args.image)
    original_img, img_array = load_image_array(image_path)

    heatmap_gradcam, prob = grad_cam(img_array, model, spec.preprocess_function, last_conv)
    heatmap_gradcam_pp, _ = grad_cam_plus_plus(img_array, model, spec.preprocess_function, last_conv)
    heatmap_scorecam, _ = score_cam(
        img_array,
        model,
        spec.preprocess_function,
        last_conv,
        max_maps=args.scorecam_maps,
    )

    output_path = XAI_DIR / f"{args.model}_{image_path.stem}_xai_methods.png"
    save_xai_figure(
        original_img,
        {
            "Grad-CAM": heatmap_gradcam,
            "Grad-CAM++": heatmap_gradcam_pp,
            "Score-CAM": heatmap_scorecam,
        },
        output_path,
        prob,
        spec.threshold,
    )

    print(f"Imagine: {image_path}")
    print(f"Model: {spec.display_name}")
    print(f"Layer XAI folosit: {last_conv}")
    print(f"Probabilitate malignant: {prob:.4f}")
    print(f"Vizualizare XAI salvata la: {output_path}")


if __name__ == "__main__":
    main()
