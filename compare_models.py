import os
from pathlib import Path

import numpy as np
import pandas as pd
import keras
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_eff
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

BASE_DIR = Path(__file__).resolve().parent
TEST_DIR = BASE_DIR / "dataset" / "new_dataset" / "test"

MODEL_1_PATH = BASE_DIR / "models" / "new_best_model_v3.keras"
MODEL_2_PATH = BASE_DIR / "models" / "model_2_resnet50.keras"

SAVE_TABLE_PATH = BASE_DIR / "results" / "model_comparison_current_check.csv"


def create_generator(preprocess_function):
    datagen = ImageDataGenerator(preprocessing_function=preprocess_function)

    generator = datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False
    )
    return generator


def evaluate_model(model_path, preprocess_function, threshold, model_name):
    gen = create_generator(preprocess_function)
    y_true = gen.classes

    model = keras.models.load_model(model_path, compile=False)
    y_prob = model.predict(gen, verbose=0).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
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
        "TP": int(tp)
    }


results = []

# pragurile tale finale
results.append(
    evaluate_model(
        MODEL_1_PATH,
        preprocess_eff,
        threshold=0.40,
        model_name="Model 1 - EfficientNetB0"
    )
)

results.append(
    evaluate_model(
        MODEL_2_PATH,
        preprocess_resnet,
        threshold=0.60,
        model_name="Model 2 - ResNet50"
    )
)

df = pd.DataFrame(results)
print("\nComparatie modele:\n")
print(df)

SAVE_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(SAVE_TABLE_PATH, index=False)
print(f"\nTabel salvat la: {SAVE_TABLE_PATH}")
