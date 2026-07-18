from dataclasses import dataclass
from pathlib import Path

from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_eff
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet

from config import (
    EFFICIENTNET_MODEL_PATH,
    EFFICIENTNET_THRESHOLD,
    RESNET_MODEL_PATH,
    RESNET_THRESHOLD,
)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    path: Path
    preprocess_function: object
    threshold: float
    last_conv_layer: str


MODEL_SPECS = {
    "efficientnet": ModelSpec(
        key="efficientnet",
        display_name="EfficientNetB0",
        path=EFFICIENTNET_MODEL_PATH,
        preprocess_function=preprocess_eff,
        threshold=EFFICIENTNET_THRESHOLD,
        last_conv_layer="top_activation",
    ),
    "resnet": ModelSpec(
        key="resnet",
        display_name="ResNet50",
        path=RESNET_MODEL_PATH,
        preprocess_function=preprocess_resnet,
        threshold=RESNET_THRESHOLD,
        last_conv_layer="conv5_block3_out",
    ),
}


def get_model_spec(model_key: str):
    try:
        return MODEL_SPECS[model_key]
    except KeyError as exc:
        valid = ", ".join(sorted(MODEL_SPECS))
        raise ValueError(f"Model necunoscut: {model_key}. Optiuni: {valid}") from exc


def available_model_specs(model_keys=None):
    keys = model_keys or MODEL_SPECS.keys()
    return [MODEL_SPECS[key] for key in keys if MODEL_SPECS[key].path.exists()]
