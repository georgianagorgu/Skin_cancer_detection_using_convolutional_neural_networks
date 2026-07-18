import subprocess
import sys
from pathlib import Path

from config import EXTERNAL_TEST_DIR


def run_step(script_name, *args):
    script_path = Path(__file__).resolve().parent / script_name
    command = [sys.executable, str(script_path), *args]
    print(f"\n=== Rulez: {' '.join(command)} ===\n")
    completed = subprocess.run(command, check=True)
    return completed.returncode


def main():
    run_step("train_efficientnet.py")
    run_step("train_resnet.py")
    run_step("evaluate_model.py", "--model", "efficientnet")
    run_step("evaluate_model.py", "--model", "resnet")
    run_step("compare_models.py")
    run_step("output_fusion.py")
    run_step("compare_graph.py")
    run_step("roc_curve_models.py")
    run_step("predict_image.py", "--model", "resnet")
    run_step("classification_examples.py", "--model", "resnet")
    run_step("grad_cam.py", "--model", "resnet")
    run_step("xai_methods.py", "--model", "resnet")
    if EXTERNAL_TEST_DIR.exists():
        run_step("external_validation.py")
    print("\nPipeline rebuilt finalizat.")


if __name__ == "__main__":
    main()
