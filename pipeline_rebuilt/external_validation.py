import argparse
from pathlib import Path

import keras
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

from common import create_eval_generator, ensure_dirs, set_global_seed
from config import EXTERNAL_TEST_DIR, EXTERNAL_VALIDATION_REPORT_PATH, REPORTS_DIR
from model_registry import available_model_specs


def evaluate_on_directory(spec, data_dir):
    gen = create_eval_generator(data_dir, spec.preprocess_function)
    model = keras.models.load_model(spec.path, compile=False)
    y_prob = model.predict(gen, verbose=0).ravel()
    y_pred = (y_prob >= spec.threshold).astype(int)
    y_true = gen.classes
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    balanced_acc = ((tp / (tp + fn + 1e-8)) + (tn / (tn + fp + 1e-8))) / 2

    print(f"\n=== Validare externa: {spec.display_name} ===")
    print(classification_report(y_true, y_pred, target_names=["benign", "malignant"], zero_division=0))

    return {
        "Model": spec.display_name,
        "External dataset": str(data_dir),
        "Threshold": spec.threshold,
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1-score": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "Balanced Accuracy": round(balanced_acc, 4),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evalueaza modelele pe un set extern. Structura asteptata: "
            "external_test/benign si external_test/malignant."
        )
    )
    parser.add_argument("--data-dir", default=str(EXTERNAL_TEST_DIR))
    parser.add_argument("--models", nargs="+", choices=["efficientnet", "resnet"], default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    set_global_seed()
    ensure_dirs(REPORTS_DIR)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Nu exista setul extern: {data_dir}. Creeaza folderele benign/malignant sau ruleaza cu --data-dir."
        )

    specs = available_model_specs(args.models)
    if not specs:
        raise FileNotFoundError("Nu am gasit modele .keras antrenate in pipeline_rebuilt_outputs/models.")

    rows = [evaluate_on_directory(spec, data_dir) for spec in specs]
    df = pd.DataFrame(rows)
    df.to_csv(EXTERNAL_VALIDATION_REPORT_PATH, index=False)
    print("\nRezumat validare externa:")
    print(df)
    print(f"\nRaport salvat la: {EXTERNAL_VALIDATION_REPORT_PATH}")


if __name__ == "__main__":
    main()
