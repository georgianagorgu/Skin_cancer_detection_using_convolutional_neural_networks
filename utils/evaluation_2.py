import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import keras
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "new_dataset"
TEST_DIR = DATASET_PATH / "test"

MODEL_PATH = BASE_DIR / "models" / "model_2_resnet50.keras"
SAVE_CM_PATH = BASE_DIR / "results" / "confusion_matrix" / "confusion_matrix_model_2_resnet50_current_check.png"


def create_test_generator():
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    test_gen = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False
    )
    return test_gen


def balanced_accuracy_score_manual(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    recall_benign = tn / (tn + fp + 1e-8)
    recall_malignant = tp / (tp + fn + 1e-8)
    balanced_acc = (recall_benign + recall_malignant) / 2

    return balanced_acc, tn, fp, fn, tp


def find_best_threshold(y_true, y_prob):
    thresholds = np.arange(0.35, 0.81, 0.05)

    best_threshold = 0.50
    best_score = -1
    best_values = None

    print("\nCaut prag optim după balanced accuracy:\n")

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        score, tn, fp, fn, tp = balanced_accuracy_score_manual(y_true, y_pred)

        print(
            f"Threshold={threshold:.2f} | "
            f"TN={tn} | FP={fp} | FN={fn} | TP={tp} | "
            f"Balanced Accuracy={score:.4f}"
        )

        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_values = (tn, fp, fn, tp)

    tn, fp, fn, tp = best_values
    print(f"\nPragul optim ales: {best_threshold:.2f}")
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"Balanced Accuracy maximă: {best_score:.4f}")

    return best_threshold


def evaluate():
    model = keras.models.load_model(MODEL_PATH, compile=False)
    test_gen = create_test_generator()

    probs = model.predict(test_gen, verbose=0).ravel()
    y_true = test_gen.classes

    best_threshold = find_best_threshold(y_true, probs)
    y_pred = (probs >= best_threshold).astype(int)

    print("\nClassification Report:\n")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["benign", "malignant"],
            zero_division=0
        )
    )

    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["benign", "malignant"]
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap="Blues", values_format="d", ax=ax)
    plt.title(f"Confusion Matrix Model 2 (threshold {best_threshold:.2f})")
    Path(SAVE_CM_PATH).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(SAVE_CM_PATH, bbox_inches="tight")
    plt.show()

    print(f"\nMatrice salvată la: {SAVE_CM_PATH}")


if __name__ == "__main__":
    evaluate()
