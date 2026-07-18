from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PIPELINE_DIR = BASE_DIR / "pipeline_rebuilt"
OUTPUT_DIR = BASE_DIR / "pipeline_rebuilt_outputs"

DATASET_DIR = BASE_DIR / "dataset" / "new_dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"
EXTERNAL_TEST_DIR = BASE_DIR / "dataset" / "external_test"

MODELS_DIR = OUTPUT_DIR / "models"
PLOTS_DIR = OUTPUT_DIR / "plots"
CONFUSION_DIR = OUTPUT_DIR / "confusion_matrix"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
GRADCAM_DIR = OUTPUT_DIR / "gradcam"
REPORTS_DIR = OUTPUT_DIR / "reports"
EXAMPLES_DIR = OUTPUT_DIR / "classification_examples"
XAI_DIR = OUTPUT_DIR / "xai"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

EFFICIENTNET_NAME = "new_best_model_v3_rebuilt"
RESNET_NAME = "model_2_resnet50_rebuilt"

EFFICIENTNET_MODEL_PATH = MODELS_DIR / f"{EFFICIENTNET_NAME}.keras"
RESNET_MODEL_PATH = MODELS_DIR / f"{RESNET_NAME}.keras"

EFFICIENTNET_PLOT_PATH = PLOTS_DIR / f"training_plot_{EFFICIENTNET_NAME}.png"
RESNET_PLOT_PATH = PLOTS_DIR / f"training_plot_{RESNET_NAME}.png"

MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison_rebuilt.csv"
OUTPUT_FUSION_REPORT_PATH = REPORTS_DIR / "output_fusion_report.csv"
OUTPUT_FUSION_PROBABILITIES_PATH = REPORTS_DIR / "output_fusion_probabilities.csv"
EXTERNAL_VALIDATION_REPORT_PATH = REPORTS_DIR / "external_validation_report.csv"
MODEL_COMPARISON_PLOT_PATH = PLOTS_DIR / "model_comparison_rebuilt.png"
ROC_CURVE_PATH = PLOTS_DIR / "roc_curve_models_rebuilt.png"

EFFICIENTNET_CM_PATH = CONFUSION_DIR / f"confusion_matrix_{EFFICIENTNET_NAME}.png"
RESNET_CM_PATH = CONFUSION_DIR / f"confusion_matrix_{RESNET_NAME}.png"
OUTPUT_FUSION_CM_PATH = CONFUSION_DIR / "confusion_matrix_output_fusion.png"

EFFICIENTNET_THRESHOLD = 0.40
RESNET_THRESHOLD = 0.60
OUTPUT_FUSION_THRESHOLD = 0.50

DEFAULT_PREDICT_IMAGE = TEST_DIR / "benign" / "31.jpg"
DEFAULT_GRADCAM_IMAGE = TEST_DIR / "benign" / "1.jpg"
