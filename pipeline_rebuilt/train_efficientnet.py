from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

from common import (
    create_train_val_generators,
    ensure_dirs,
    plot_history,
    set_global_seed,
    standard_metrics,
)
from config import EFFICIENTNET_MODEL_PATH, EFFICIENTNET_PLOT_PATH, MODELS_DIR, PLOTS_DIR, TRAIN_DIR
import tensorflow as tf


def main():
    set_global_seed()
    ensure_dirs(MODELS_DIR, PLOTS_DIR)

    train_gen, val_gen, class_weights = create_train_val_generators(TRAIN_DIR, preprocess_input)
    print("Class weights:", class_weights)

    base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
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
        metrics=standard_metrics(),
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=4,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            EFFICIENTNET_MODEL_PATH,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
        ),
    ]

    print("\n=== EfficientNetB0 etapa 1 ===")
    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10,
        callbacks=callbacks,
        class_weight=class_weights,
    )

    print("\n=== EfficientNetB0 etapa 2: fine-tuning ===")
    base_model.trainable = True
    for layer in base_model.layers[:-40]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=8e-6),
        loss="binary_crossentropy",
        metrics=standard_metrics(),
    )

    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=8,
        callbacks=callbacks,
        class_weight=class_weights,
    )

    plot_history(history1, history2, EFFICIENTNET_PLOT_PATH)
    print(f"Model salvat la: {EFFICIENTNET_MODEL_PATH}")
    print(f"Grafic salvat la: {EFFICIENTNET_PLOT_PATH}")


if __name__ == "__main__":
    main()
