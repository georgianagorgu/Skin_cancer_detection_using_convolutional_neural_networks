import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import keras
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_eff
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

BASE_DIR = Path(__file__).resolve().parent
TEST_DIR = BASE_DIR / "dataset" / "new_dataset" / "test"

MODEL_1_PATH = BASE_DIR / "models" / "new_best_model_v3.keras"
MODEL_2_PATH = BASE_DIR / "models" / "model_2_resnet50.keras"

SAVE_PATH = BASE_DIR / "results" / "plots" / "roc_curve_models_current_check.png"


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


def get_probs(model_path, preprocess_function):
    gen = create_generator(preprocess_function)
    y_true = gen.classes

    model = keras.models.load_model(model_path, compile=False)
    y_prob = model.predict(gen, verbose=0).ravel()

    return y_true, y_prob


# Model 1
y_true_1, y_prob_1 = get_probs(MODEL_1_PATH, preprocess_eff)
fpr1, tpr1, _ = roc_curve(y_true_1, y_prob_1)
roc_auc1 = auc(fpr1, tpr1)

# Model 2
y_true_2, y_prob_2 = get_probs(MODEL_2_PATH, preprocess_resnet)
fpr2, tpr2, _ = roc_curve(y_true_2, y_prob_2)
roc_auc2 = auc(fpr2, tpr2)

plt.figure(figsize=(8, 6))
plt.plot(fpr1, tpr1, label=f"EfficientNetB0 (AUC = {roc_auc1:.4f})")
plt.plot(fpr2, tpr2, label=f"ResNet50 (AUC = {roc_auc2:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Comparatie modele")
plt.legend()
plt.grid(True)

SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(SAVE_PATH, bbox_inches="tight")
plt.show()

print(f"Grafic ROC salvat la: {SAVE_PATH}")
print(f"AUC EfficientNetB0: {roc_auc1:.4f}")
print(f"AUC ResNet50: {roc_auc2:.4f}")
