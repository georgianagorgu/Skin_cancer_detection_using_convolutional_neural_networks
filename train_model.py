import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers, models

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

DATASET_PATH = "D:/PROIECT_LICENTA/dataset/new_dataset"
TRAIN_DIR = os.path.join(DATASET_PATH, "train")

MODEL_PATH = "D:/PROIECT_LICENTA/models/new_best_model_v3.keras"
PLOT_PATH = "D:/PROIECT_LICENTA/results/plots/training_plot_new_best_v3.png"

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    zoom_range=0.15,
    horizontal_flip=True,
    width_shift_range=0.08,
    height_shift_range=0.08,
    brightness_range=(0.9, 1.1),
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=True
)

val_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

classes = train_gen.classes
class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(classes),
    y=classes
)
class_weights = dict(enumerate(class_weights_array))

print("Class weights:", class_weights)

base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.45)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.30)(x)
output = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=4,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6
    ),
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_auc",
        mode="max",
        save_best_only=True
    )
]

print("\n=== Etapa 1 ===")
history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=callbacks,
    class_weight=class_weights
)

print("\n=== Etapa 2: fine-tuning ===")
base_model.trainable = True

for layer in base_model.layers[:-40]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=8e-6),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=8,
    callbacks=callbacks,
    class_weight=class_weights
)

def plot_history(hist1, hist2):
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
    plt.title("Accuracy / Recall")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label="Train Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.plot(auc, label="Train AUC")
    plt.plot(val_auc, label="Validation AUC")
    plt.title("Loss / AUC")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()

    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.show()

plot_history(history1, history2)

print(f"Model salvat la: {MODEL_PATH}")