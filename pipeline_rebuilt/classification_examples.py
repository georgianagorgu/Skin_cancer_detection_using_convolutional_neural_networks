import argparse
import shutil
from pathlib import Path

import keras
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import create_eval_generator, ensure_dirs, set_global_seed
from config import EXAMPLES_DIR, TEST_DIR
from model_registry import get_model_spec


def class_name(value):
    return "malignant" if value == 1 else "benign"


def copy_examples(df, data_dir, model_key, limit):
    correct_dir = EXAMPLES_DIR / model_key / "correct"
    incorrect_dir = EXAMPLES_DIR / model_key / "incorrect"
    ensure_dirs(correct_dir, incorrect_dir)

    for _, row in df[df["is_correct"]].head(limit).iterrows():
        src = data_dir / row["filename"]
        dst = correct_dir / f"true_{row['true_class']}_pred_{row['pred_class']}_{Path(row['filename']).name}"
        shutil.copy2(src, dst)

    for _, row in df[~df["is_correct"]].head(limit).iterrows():
        src = data_dir / row["filename"]
        dst = incorrect_dir / f"true_{row['true_class']}_pred_{row['pred_class']}_{Path(row['filename']).name}"
        shutil.copy2(src, dst)


def save_montage(df, data_dir, model_key, limit):
    samples = pd.concat([df[df["is_correct"]].head(limit), df[~df["is_correct"]].head(limit)])
    if samples.empty:
        return

    cols = min(4, len(samples))
    rows = int((len(samples) + cols - 1) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = getattr(axes, "ravel", lambda: [axes])()

    for ax, (_, row) in zip(axes, samples.iterrows()):
        img = plt.imread(data_dir / row["filename"])
        ax.imshow(img)
        status = "corect" if row["is_correct"] else "incorect"
        ax.set_title(
            f"{status}\nreal: {row['true_class']} | pred: {row['pred_class']}\nP(mal)={row['p_malignant']:.3f}",
            fontsize=9,
        )
        ax.axis("off")

    for ax in axes[len(samples) :]:
        ax.axis("off")

    plt.tight_layout()
    output_path = EXAMPLES_DIR / model_key / f"{model_key}_classification_examples.png"
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Exporta exemple de clasificare corecta si incorecta.")
    parser.add_argument("--model", choices=["efficientnet", "resnet"], default="resnet")
    parser.add_argument("--data-dir", default=str(TEST_DIR))
    parser.add_argument("--limit", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    set_global_seed()
    spec = get_model_spec(args.model)
    data_dir = Path(args.data_dir)
    ensure_dirs(EXAMPLES_DIR / args.model)

    gen = create_eval_generator(data_dir, spec.preprocess_function)
    model = keras.models.load_model(spec.path, compile=False)
    y_prob = model.predict(gen, verbose=0).ravel()
    y_pred = (y_prob >= spec.threshold).astype(int)

    df = pd.DataFrame(
        {
            "filename": gen.filenames,
            "true_label": gen.classes,
            "true_class": [class_name(value) for value in gen.classes],
            "p_benign": 1.0 - y_prob,
            "p_malignant": y_prob,
            "pred_label": y_pred,
            "pred_class": [class_name(value) for value in y_pred],
            "is_correct": y_pred == gen.classes,
        }
    )
    csv_path = EXAMPLES_DIR / args.model / f"{args.model}_classification_examples.csv"
    df.to_csv(csv_path, index=False)
    copy_examples(df, data_dir, args.model, args.limit)
    save_montage(df, data_dir, args.model, args.limit)

    print(f"CSV exemple salvat la: {csv_path}")
    print(f"Imagini copiate in: {EXAMPLES_DIR / args.model}")


if __name__ == "__main__":
    main()
