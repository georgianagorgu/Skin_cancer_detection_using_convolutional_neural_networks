import argparse
import os
from pathlib import Path

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import matplotlib.pyplot as plt
import numpy as np

try:
    import keras
    import tensorflow as tf
except ImportError as exc:
    raise SystemExit(
        "TensorFlow nu poate fi importat in acest mediu. "
        "Pe Windows, cauza uzuala este runtime-ul nativ TensorFlow "
        "(Visual C++ Redistributable lipsa/invechita sau build incompatibil cu CPU-ul). "
        "Detalii originale:\n"
        f"{exc}"
    ) from exc

from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_effnet
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet
from tensorflow.keras.preprocessing import image

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "model_2_resnet50.keras"
IMAGE_PATH = BASE_DIR / "dataset" / "new_dataset" / "test" / "benign" / "1.jpg"
OUTPUT_PATH = BASE_DIR / "results" / "gradcam_model_2_resnet50.png"

IMG_SIZE = (224, 224)
THRESHOLD = 0.60
LAST_CONV_LAYER_NAME = "conv5_block3_out"


def get_preprocess_function(model_path: Path):
    model_name = model_path.name.lower()
    if "efficientnet" in model_name:
        return preprocess_effnet
    return preprocess_resnet


def parse_args():
    parser = argparse.ArgumentParser(description="Genereaza vizualizarea Grad-CAM pentru o imagine.")
    parser.add_argument("--model", default=str(MODEL_PATH), help="Calea catre modelul .keras")
    parser.add_argument("--image", default=str(IMAGE_PATH), help="Calea catre imagine")
    parser.add_argument("--threshold", type=float, default=THRESHOLD, help="Pragul de clasificare")
    parser.add_argument(
        "--last-conv-layer",
        default=LAST_CONV_LAYER_NAME,
        help="Numele stratului convolutional folosit pentru Grad-CAM",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_PATH),
        help="Fisierul in care se salveaza vizualizarea Grad-CAM",
    )
    return parser.parse_args()


def load_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
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

    raise ValueError("Nu am gasit un strat convolutional 4D potrivit pentru Grad-CAM.")


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


def display_gradcam(original_img, heatmap, output_path=None, alpha=0.4):
    original_img = original_img.resize(IMG_SIZE)
    original_array = np.array(original_img).astype("float32") / 255.0

    heatmap_resized = tf.image.resize(heatmap[..., np.newaxis], IMG_SIZE).numpy().squeeze()

    cmap = plt.get_cmap("jet")
    heatmap_colored = cmap(heatmap_resized)[:, :, :3]

    overlay = (1 - alpha) * original_array + alpha * heatmap_colored
    overlay = np.clip(overlay, 0, 1)

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

    backend = plt.get_backend().lower()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            output_path = output_path.with_name(f"{output_path.stem}_new{output_path.suffix}")
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Vizualizare salvata la: {output_path}")

    if "agg" in backend:
        plt.close()
    else:
        plt.show()


def main():
    args = parse_args()
    model_path = Path(args.model)
    image_path = Path(args.image)
    output_path = Path(args.output) if args.output else None
    preprocess_input = get_preprocess_function(model_path)

    if not model_path.exists():
        print(f"Modelul nu exista: {model_path}")
        return

    if not image_path.exists():
        print(f"Imaginea nu exista: {image_path}")
        return

    model = keras.models.load_model(model_path, compile=False)
    last_conv_layer_name = find_last_conv_layer(model, args.last_conv_layer)

    original_img, img_array = load_image(image_path, IMG_SIZE)
    heatmap, prob = make_gradcam_heatmap(
        img_array,
        model,
        preprocess_input,
        last_conv_layer_name,
    )

    label = "malignant" if prob >= args.threshold else "benign"

    print(f"Imagine: {image_path}")
    print(f"Layer Grad-CAM folosit: {last_conv_layer_name}")
    print(f"Probabilitate malignant: {prob:.4f}")
    print(f"Prag folosit: {args.threshold:.2f}")
    print(f"Predictie finala: {label}")

    display_gradcam(original_img, heatmap, output_path=output_path, alpha=0.4)


if __name__ == "__main__":
    main()
