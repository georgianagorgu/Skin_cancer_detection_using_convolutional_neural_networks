import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import keras

from common import create_test_generator, ensure_dirs, set_global_seed
from config import MODEL_COMPARISON_PATH, REPORTS_DIR, TEST_DIR
from model_registry import available_model_specs


def evaluate_model(model_path, preprocess_function, threshold, model_name):
    gen = create_test_generator(TEST_DIR, preprocess_function)
    y_true = gen.classes
    model = keras.models.load_model(model_path, compile=False)
    y_prob = model.predict(gen, verbose=0).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    balanced_acc = ((tp / (tp + fn + 1e-8)) + (tn / (tn + fp + 1e-8))) / 2

    return {
        "Model": model_name,
        "Threshold": threshold,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-score": round(f1, 4),
        "Balanced Accuracy": round(balanced_acc, 4),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def main():
    set_global_seed()
    ensure_dirs(REPORTS_DIR)

    specs = available_model_specs()
    if not specs:
        raise FileNotFoundError("Nu am gasit modele .keras antrenate in pipeline_rebuilt_outputs/models.")

    results = [
        evaluate_model(
            spec.path,
            spec.preprocess_function,
            threshold=spec.threshold,
            model_name=f"{spec.display_name} Rebuilt",
        )
        for spec in specs
    ]

    df = pd.DataFrame(results)
    print("\nComparatie modele rebuilt:\n")
    print(df)
    df.to_csv(MODEL_COMPARISON_PATH, index=False)
    print(f"\nTabel salvat la: {MODEL_COMPARISON_PATH}")


if __name__ == "__main__":
    main()
