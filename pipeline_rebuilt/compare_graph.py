import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from common import ensure_dirs
from config import MODEL_COMPARISON_PATH, MODEL_COMPARISON_PLOT_PATH, PLOTS_DIR

METRIC_COLUMNS = ["Accuracy", "Precision", "Recall", "F1-score"]


def main():
    ensure_dirs(PLOTS_DIR)

    if not MODEL_COMPARISON_PATH.exists():
        raise FileNotFoundError(
            f"Fisierul de comparatie nu exista: {MODEL_COMPARISON_PATH}. Ruleaza mai intai compare_models.py."
        )

    df = pd.read_csv(MODEL_COMPARISON_PATH)
    model_labels = [name.replace(" Rebuilt", "") for name in df["Model"].tolist()]

    x = np.arange(len(model_labels))
    width = 0.18

    plt.figure(figsize=(10, 6))
    for index, metric in enumerate(METRIC_COLUMNS):
        offset = (index - 1.5) * width
        bars = plt.bar(x + offset, df[metric], width, label=metric)
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.005,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.xticks(x, model_labels)
    plt.ylim(0, 1.0)
    plt.ylabel("Scor")
    plt.title("Comparatie performanta modele rebuilt")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(MODEL_COMPARISON_PLOT_PATH, bbox_inches="tight")
    plt.close()

    print(f"Grafic comparatie salvat la: {MODEL_COMPARISON_PLOT_PATH}")


if __name__ == "__main__":
    main()
