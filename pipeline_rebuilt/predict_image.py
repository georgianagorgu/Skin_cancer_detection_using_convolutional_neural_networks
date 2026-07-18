import argparse
from pathlib import Path

import keras
import numpy as np
from tensorflow.keras.preprocessing import image

from config import DEFAULT_PREDICT_IMAGE, IMG_SIZE, PREDICTIONS_DIR
from common import ensure_dirs
from model_registry import get_model_spec


def parse_args():
    parser = argparse.ArgumentParser(description="Predictie pentru o singura imagine, din pipeline-ul rebuilt.")
    parser.add_argument("--model", choices=["efficientnet", "resnet"], default="resnet")
    parser.add_argument("--image", default=str(DEFAULT_PREDICT_IMAGE))
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dirs(PREDICTIONS_DIR)

    spec = get_model_spec(args.model)

    image_path = Path(args.image)
    model = keras.models.load_model(spec.path, compile=False)

    img = image.load_img(image_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = spec.preprocess_function(img_array)

    prob = float(np.squeeze(model.predict(img_array, verbose=0)))
    label = "malignant" if prob >= spec.threshold else "benign"

    report_path = PREDICTIONS_DIR / f"{args.model}_{image_path.stem}_prediction.txt"
    report = (
        f"Imagine: {image_path}\n"
        f"Model: {spec.path.name}\n"
        f"Probabilitate benign: {1.0 - prob:.4f}\n"
        f"Probabilitate malignant: {prob:.4f}\n"
        f"Prag folosit: {spec.threshold:.2f}\n"
        f"Predictie finala: {label}\n"
    )
    report_path.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"Raport salvat la: {report_path}")


if __name__ == "__main__":
    main()
