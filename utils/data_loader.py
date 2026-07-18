from pathlib import Path
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_ROOT / "dataset"
CSV_FILE = BASE_DIR / "HAM10000_metadata.csv"
OUTPUT_DIR = BASE_DIR / "balanced_binary"

MALIGNANT_CLASSES = {"mel", "bcc", "akiec"}
BENIGN_CLASSES = {"nv", "bkl", "df", "vasc"}

RANDOM_STATE = 42


def create_folders():
    for split in ["train", "valid", "test"]:
        for label in ["benign", "malignant"]:
            folder_path = OUTPUT_DIR / split / label
            folder_path.mkdir(parents=True, exist_ok=True)


def clear_output_dir():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    create_folders()


def get_image_path(image_id):
    image_name = f"{image_id}.jpg"

    path1 = BASE_DIR / "HAM10000_images_part_1" / image_name
    path2 = BASE_DIR / "HAM10000_images_part_2" / image_name

    if path1.exists():
        return path1
    if path2.exists():
        return path2
    return None


def map_label(dx_value):
    if dx_value in MALIGNANT_CLASSES:
        return "malignant"
    if dx_value in BENIGN_CLASSES:
        return "benign"
    return None


def copy_images(split_df, split_name):
    total = len(split_df)
    print(f"\nÎncep copierea pentru {split_name}: {total} imagini")

    for index, (_, row) in enumerate(split_df.iterrows(), start=1):
        image_id = row["image_id"]
        label = row["label"]

        source_path = get_image_path(image_id)
        if source_path is None:
            print(f"Imagine lipsă: {image_id}")
            continue

        destination_path = OUTPUT_DIR / split_name / label / f"{image_id}.jpg"
        shutil.copy2(source_path, destination_path)

        if index % 200 == 0 or index == total:
            print(f"{split_name}: {index}/{total}")


def process_data():
    if not CSV_FILE.exists():
        print(f"Nu găsesc fișierul CSV: {CSV_FILE}")
        return

    clear_output_dir()

    df = pd.read_csv(CSV_FILE)
    df["label"] = df["dx"].apply(map_label)
    df = df[df["label"].notna()].copy()

    benign_df = df[df["label"] == "benign"].copy()
    malignant_df = df[df["label"] == "malignant"].copy()

    print("Distribuție inițială:")
    print(df["label"].value_counts())

    # echilibrare: luăm același număr ca numărul de imagini maligne
    target_count = len(malignant_df)
    benign_df = benign_df.sample(n=target_count, random_state=RANDOM_STATE)

    balanced_df = pd.concat([benign_df, malignant_df], ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    print("\nDistribuție după echilibrare:")
    print(balanced_df["label"].value_counts())

    train_df, temp_df = train_test_split(
        balanced_df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=balanced_df["label"]
    )

    valid_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_df["label"]
    )

    print("\nTrain:")
    print(train_df["label"].value_counts())

    print("\nValid:")
    print(valid_df["label"].value_counts())

    print("\nTest:")
    print(test_df["label"].value_counts())

    copy_images(train_df, "train")
    copy_images(valid_df, "valid")
    copy_images(test_df, "test")

    print(f"\nDataset echilibrat creat cu succes la: {OUTPUT_DIR}")


if __name__ == "__main__":
    process_data()