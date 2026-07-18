import argparse

import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from common import create_test_generator, ensure_dirs, set_global_seed
from config import EFFICIENTNET_CM_PATH, RESNET_CM_PATH, TEST_DIR
from model_registry import get_model_spec


def balanced_accuracy_score_manual(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    recall_benign = tn / (tn + fp + 1e-8)
    recall_malignant = tp / (tp + fn + 1e-8)
    balanced_acc = (recall_benign + recall_malignant) / 2
    return balanced_acc, tn, fp, fn, tp


def find_best_threshold(y_true, y_prob):
    thresholds = np.arange(0.35, 0.81, 0.05)
    best_threshold = 0.50
    best_score = -1
    best_values = None

    print("\nCaut prag optim dupa balanced accuracy:\n")
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
    print(f"Balanced Accuracy maxima: {best_score:.4f}")
    return best_threshold


def evaluate(model_path, preprocess_function, output_path, forced_threshold=None, title="Confusion Matrix"):
    set_global_seed()
    ensure_dirs(output_path.parent)

    model = keras.models.load_model(model_path, compile=False)
    test_gen = create_test_generator(TEST_DIR, preprocess_function)

    probs = model.predict(test_gen, verbose=0).ravel()
    y_true = test_gen.classes
    threshold = forced_threshold if forced_threshold is not None else find_best_threshold(y_true, probs)
    y_pred = (probs >= threshold).astype(int)

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=["benign", "malignant"], zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["benign", "malignant"])

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap="Blues", values_format="d", ax=ax)
    plt.title(f"{title} (threshold {threshold:.2f})")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    print(f"\nMatrice salvata la: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evalueaza un model si salveaza confusion matrix.")
    parser.add_argument("--model", choices=["efficientnet", "resnet"], required=True)
    parser.add_argument("--use-fixed-threshold", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    spec = get_model_spec(args.model)
    output_paths = {
        "efficientnet": EFFICIENTNET_CM_PATH,
        "resnet": RESNET_CM_PATH,
    }
    evaluate(
        spec.path,
        spec.preprocess_function,
        output_paths[args.model],
        forced_threshold=spec.threshold if args.use_fixed_threshold else None,
        title=f"Confusion Matrix {spec.display_name} Rebuilt",
    )


if __name__ == "__main__":
    main()
