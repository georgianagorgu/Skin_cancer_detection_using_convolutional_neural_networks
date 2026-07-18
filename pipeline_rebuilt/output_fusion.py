import argparse

import keras
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import binary_probs_to_two_columns, create_eval_generator, ensure_dirs, set_global_seed
from config import (
    CONFUSION_DIR,
    OUTPUT_FUSION_CM_PATH,
    OUTPUT_FUSION_PROBABILITIES_PATH,
    OUTPUT_FUSION_REPORT_PATH,
    OUTPUT_FUSION_THRESHOLD,
    REPORTS_DIR,
    TEST_DIR,
    TRAIN_DIR,
)
from model_registry import available_model_specs


def predict_probabilities(spec, data_dir):
    gen = create_eval_generator(data_dir, spec.preprocess_function)
    model = keras.models.load_model(spec.path, compile=False)
    malignant_prob = model.predict(gen, verbose=0).ravel()
    two_class_probs = binary_probs_to_two_columns(malignant_prob)
    return gen, malignant_prob, two_class_probs


def model_train_accuracy(spec):
    gen, malignant_prob, _ = predict_probabilities(spec, TRAIN_DIR)
    y_pred = (malignant_prob >= spec.threshold).astype(int)
    return accuracy_score(gen.classes, y_pred)


def compute_weights(specs):
    train_accuracies = np.array([model_train_accuracy(spec) for spec in specs], dtype="float32")
    total = float(np.sum(train_accuracies))
    if total <= 0:
        return np.ones(len(specs), dtype="float32") / len(specs), train_accuracies
    return train_accuracies / total, train_accuracies


def save_confusion_matrix(y_true, y_pred, output_path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["benign", "malignant"])
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap="Blues", values_format="d", ax=ax)
    plt.title("Confusion Matrix - Output Fusion")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def run_output_fusion(data_dir, selected_models=None, threshold=OUTPUT_FUSION_THRESHOLD):
    set_global_seed()
    ensure_dirs(REPORTS_DIR, CONFUSION_DIR)

    specs = available_model_specs(selected_models)
    if len(specs) < 2:
        raise FileNotFoundError("Output fusion necesita cel putin doua modele antrenate in pipeline_rebuilt_outputs/models.")

    weights, train_accuracies = compute_weights(specs)
    weighted_probs = None
    reference_gen = None
    probability_table = None

    for spec, weight, train_acc in zip(specs, weights, train_accuracies):
        gen, malignant_prob, two_class_probs = predict_probabilities(spec, data_dir)
        if reference_gen is None:
            reference_gen = gen
            weighted_probs = np.zeros_like(two_class_probs, dtype="float32")
            probability_table = pd.DataFrame(
                {
                    "filename": gen.filenames,
                    "true_label": gen.classes,
                    "true_class": ["malignant" if value == 1 else "benign" for value in gen.classes],
                }
            )
        elif list(reference_gen.filenames) != list(gen.filenames):
            raise ValueError("Generatorii nu au aceeasi ordine a fisierelor. Verifica structura setului de date.")

        weighted_probs += weight * two_class_probs
        probability_table[f"{spec.key}_p_benign"] = two_class_probs[:, 0]
        probability_table[f"{spec.key}_p_malignant"] = two_class_probs[:, 1]
        probability_table[f"{spec.key}_train_accuracy"] = train_acc
        probability_table[f"{spec.key}_weight"] = weight

    y_true = reference_gen.classes
    y_pred = (weighted_probs[:, 1] >= threshold).astype(int)
    probability_table["fusion_p_benign"] = weighted_probs[:, 0]
    probability_table["fusion_p_malignant"] = weighted_probs[:, 1]
    probability_table["fusion_pred_label"] = y_pred
    probability_table["fusion_pred_class"] = ["malignant" if value == 1 else "benign" for value in y_pred]
    probability_table["is_correct"] = y_pred == y_true

    report = classification_report(
        y_true,
        y_pred,
        target_names=["benign", "malignant"],
        zero_division=0,
        output_dict=True,
    )
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(OUTPUT_FUSION_REPORT_PATH)
    probability_table.to_csv(OUTPUT_FUSION_PROBABILITIES_PATH, index=False)
    save_confusion_matrix(y_true, y_pred, OUTPUT_FUSION_CM_PATH)

    print("\nCoeficienti output fusion:")
    for spec, weight, train_acc in zip(specs, weights, train_accuracies):
        print(f"{spec.display_name}: train_accuracy={train_acc:.4f}, weight={weight:.4f}")
    print("\nRaport output fusion:")
    print(classification_report(y_true, y_pred, target_names=["benign", "malignant"], zero_division=0))
    print(f"Probabilitati salvate la: {OUTPUT_FUSION_PROBABILITIES_PATH}")
    print(f"Raport salvat la: {OUTPUT_FUSION_REPORT_PATH}")
    print(f"Matrice salvata la: {OUTPUT_FUSION_CM_PATH}")


def parse_args():
    parser = argparse.ArgumentParser(description="Output fusion cu media ponderata a probabilitatilor CNN.")
    parser.add_argument("--data-dir", default=str(TEST_DIR))
    parser.add_argument("--models", nargs="+", choices=["efficientnet", "resnet"], default=None)
    parser.add_argument("--threshold", type=float, default=OUTPUT_FUSION_THRESHOLD)
    return parser.parse_args()


def main():
    args = parse_args()
    run_output_fusion(args.data_dir, selected_models=args.models, threshold=args.threshold)


if __name__ == "__main__":
    main()
