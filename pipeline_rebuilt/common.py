import os
import random
from pathlib import Path

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import BATCH_SIZE, IMG_SIZE, SEED


def ensure_dirs(*paths: Path):
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)


def create_train_val_generators(train_dir, preprocess_function):
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_function,
        rotation_range=20,
        zoom_range=0.15,
        horizontal_flip=True,
        width_shift_range=0.08,
        height_shift_range=0.08,
        brightness_range=(0.9, 1.1),
        validation_split=0.2,
    )

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="training",
        shuffle=True,
        seed=SEED,
    )

    val_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="validation",
        shuffle=False,
        seed=SEED,
    )

    classes = train_gen.classes
    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(classes),
        y=classes,
    )
    class_weights = dict(enumerate(class_weights_array))
    return train_gen, val_gen, class_weights


def create_eval_generator(data_dir, preprocess_function, batch_size=BATCH_SIZE):
    datagen = ImageDataGenerator(preprocessing_function=preprocess_function)
    return datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,
    )


def create_test_generator(test_dir, preprocess_function):
    return create_eval_generator(test_dir, preprocess_function)


def binary_probs_to_two_columns(y_prob):
    y_prob = np.asarray(y_prob).reshape(-1)
    return np.column_stack([1.0 - y_prob, y_prob])


def standard_metrics():
    return [
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ]


def plot_history(hist1, hist2, plot_path, title_suffix=""):
    acc = hist1.history["accuracy"] + hist2.history["accuracy"]
    val_acc = hist1.history["val_accuracy"] + hist2.history["val_accuracy"]

    loss = hist1.history["loss"] + hist2.history["loss"]
    val_loss = hist1.history["val_loss"] + hist2.history["val_loss"]

    auc = hist1.history["auc"] + hist2.history["auc"]
    val_auc = hist1.history["val_auc"] + hist2.history["val_auc"]

    recall = hist1.history["recall"] + hist2.history["recall"]
    val_recall = hist1.history["val_recall"] + hist2.history["val_recall"]

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(acc, label="Train Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.plot(recall, label="Train Recall")
    plt.plot(val_recall, label="Validation Recall")
    plt.title(f"Accuracy / Recall{title_suffix}")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label="Train Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.plot(auc, label="Train AUC")
    plt.plot(val_auc, label="Validation AUC")
    plt.title(f"Loss / AUC{title_suffix}")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()

    plt.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
