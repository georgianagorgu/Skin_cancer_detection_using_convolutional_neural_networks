import keras
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve

from common import create_test_generator, ensure_dirs, set_global_seed
from config import PLOTS_DIR, ROC_CURVE_PATH, TEST_DIR
from model_registry import available_model_specs


def get_probs(model_path, preprocess_function):
    gen = create_test_generator(TEST_DIR, preprocess_function)
    y_true = gen.classes
    model = keras.models.load_model(model_path, compile=False)
    y_prob = model.predict(gen, verbose=0).ravel()
    return y_true, y_prob


def main():
    set_global_seed()
    ensure_dirs(PLOTS_DIR)

    specs = available_model_specs()
    if not specs:
        raise FileNotFoundError("Nu am gasit modele .keras antrenate in pipeline_rebuilt_outputs/models.")

    plt.figure(figsize=(8, 6))
    auc_values = []
    for spec in specs:
        y_true, y_prob = get_probs(spec.path, spec.preprocess_function)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        auc_values.append((spec.display_name, roc_auc))
        plt.plot(fpr, tpr, label=f"{spec.display_name} Rebuilt (AUC = {roc_auc:.4f})")

    plt.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Comparatie modele rebuilt")
    plt.legend()
    plt.grid(True)
    plt.savefig(ROC_CURVE_PATH, bbox_inches="tight")
    plt.close()

    print(f"Grafic ROC salvat la: {ROC_CURVE_PATH}")
    for name, roc_auc in auc_values:
        print(f"AUC {name}: {roc_auc:.4f}")


if __name__ == "__main__":
    main()
