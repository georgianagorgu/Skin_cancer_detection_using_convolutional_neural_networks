import argparse
import os
from pathlib import Path

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

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
IMAGE_PATH = BASE_DIR / "dataset" / "new_dataset" / "test" / "benign" / "31.jpg"

IMG_SIZE = (224, 224)
THRESHOLD = 0.60


def get_preprocess_function(model_path: Path):
    model_name = model_path.name.lower()
    if "efficientnet" in model_name:
        return preprocess_effnet
    return preprocess_resnet


def parse_args():
    parser = argparse.ArgumentParser(description="Ruleaza predictia pentru o singura imagine.")
    parser.add_argument("--model", default=str(MODEL_PATH), help="Calea catre modelul .keras")
    parser.add_argument("--image", default=str(IMAGE_PATH), help="Calea catre imagine")
    parser.add_argument("--threshold", type=float, default=THRESHOLD, help="Pragul de clasificare")
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model)
    image_path = Path(args.image)
    preprocess_input = get_preprocess_function(model_path)

    print("1. Script pornit")

    print("2. Verific model...")
    print("   Exista model:", model_path.exists())
    if not model_path.exists():
        print(f"Modelul nu exista: {model_path}")
        return

    print("3. Verific imagine...")
    print("   Exista imagine:", image_path.exists())
    if not image_path.exists():
        print(f"Imaginea nu exista: {image_path}")
        return

    print("4. Incarc modelul...")
    model = keras.models.load_model(model_path, compile=False)
    print("   Model incarcat cu succes")

    print("5. Incarc imaginea...")
    img = image.load_img(image_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    print("   Imagine pregatita")

    print("6. Fac predictia...")
    prediction = model.predict(img_array, verbose=0)
    prob = float(np.squeeze(prediction))
    print("   Predictia a fost calculata")

    label = "malignant" if prob >= args.threshold else "benign"

    print(f"Imagine: {image_path}")
    print(f"Probabilitate malignant: {prob:.4f}")
    print(f"Prag folosit: {args.threshold:.2f}")
    print(f"Predictie finala: {label}")


if __name__ == "__main__":
    main()
