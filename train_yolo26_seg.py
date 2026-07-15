# -*- coding: utf-8 -*-
"""
YOLO26 semantic-segmentation benchmark for adenovirus in TEM images.

This script trains YOLO26n/s/m/l/x semantic-segmentation model variants using a selected
augmentation profile, preparing the segmentation dataset from image folders and
binary masks, evaluating each model on validation and test splits, saving the
best-performing weights, and exporting predictions and comparison metrics.

@author: Olivier.Rukundo, Ph.D., University of Eastern Finland, July 15, 2026
"""

import csv
import gc
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

# =========================================================
# CONFIGURATION
# =========================================================

base_path = Path(__file__).resolve().parent

models_to_train = {
    "26n_sem": "yolo26n-sem.pt",
    "26s_sem": "yolo26s-sem.pt",
    "26m_sem": "yolo26m-sem.pt",
    "26l_sem": "yolo26l-sem.pt",
    "26x_sem": "yolo26x-sem.pt",
}

epochs = 300
patience = 100
image_size = 1376
batch_size = 8 
random_seed = 42
workers = 4

model_batch_sizes = {
    "26n_sem": batch_size,
    "26s_sem": batch_size,
    "26m_sem": batch_size,
    "26l_sem": 1,
    "26x_sem": 1,
}

semantic_dataset_root = base_path / "semantic_dataset"

augmentation_profiles = {
    "no_augmentation": {
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "close_mosaic": 0,
    },
    "geometric_only": {
        "degrees": 180.0,
        "translate": 0.10,
        "scale": 0.20,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.5,
        "fliplr": 0.5,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "close_mosaic": 0,
    },
    "geometric_mosaic": {
        "degrees": 180.0,
        "translate": 0.10,
        "scale": 0.20,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.5,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "close_mosaic": 10,
    },
}

augmentation_name = "geometric_only"
augmentation_settings = augmentation_profiles[augmentation_name]

device = 0 if torch.cuda.is_available() else "cpu"

dataset_yaml_path = semantic_dataset_root / "dataset.yaml"
runs_folder = base_path / "runs_semantic"
models_folder = base_path / "trained_models_semantic"
results_folder = base_path / "comparison_results_semantic"

validation_csv = results_folder / "yolo26_semantic_validation_comparison.csv"
test_csv = results_folder / "yolo26_semantic_test_comparison.csv"
failures_csv = results_folder / "yolo26_semantic_failed_models.csv"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
EPSILON = 1e-12


# =========================================================
# DATASET PREPARATION AND VALIDATION
# =========================================================

def prepare_semantic_dataset() -> None:
    """
    Build a prepared dataset without modifying the original masks.

    Source masks may contain either {0, 1} or {0, 255}. Every prepared
    mask is saved as an 8-bit PNG containing only class IDs {0, 1}.
    """
    if semantic_dataset_root.exists():
        shutil.rmtree(semantic_dataset_root)

    for split in ("train", "val", "test"):
        source_images_folder = base_path / "images" / split
        source_masks_folder = base_path / "masks" / split

        if not source_images_folder.exists():
            raise FileNotFoundError(
                f"Image folder not found: {source_images_folder}"
            )

        if not source_masks_folder.exists():
            raise FileNotFoundError(
                f"Mask folder not found: {source_masks_folder}"
            )

        # prepared_images_folder = semantic_dataset_root / split / "images"
        # prepared_masks_folder = semantic_dataset_root / split / "masks"
        
        prepared_images_folder = semantic_dataset_root / "images" / split
        prepared_masks_folder = semantic_dataset_root / "masks" / split

        prepared_images_folder.mkdir(parents=True, exist_ok=True)
        prepared_masks_folder.mkdir(parents=True, exist_ok=True)

        image_paths = sorted(
            path
            for path in source_images_folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not image_paths:
            raise RuntimeError(
                f"No images found in: {source_images_folder}"
            )

        print("\n" + "-" * 70)
        print(f"Preparing split: {split}")
        print(f"Source images: {source_images_folder}")
        print(f"Source masks:  {source_masks_folder}")
        print("-" * 70)

        for index, image_path in enumerate(image_paths, start=1):
            source_mask_path = source_masks_folder / image_path.name

            if not source_mask_path.exists():
                raise FileNotFoundError(
                    f"Mask not found for image:\n"
                    f"  Image: {image_path}\n"
                    f"  Expected mask: {source_mask_path}"
                )

            with Image.open(image_path) as image_object:
                image_size_pixels = image_object.size

            mask = np.asarray(
                Image.open(source_mask_path).convert("L")
            )

            if (mask.shape[1], mask.shape[0]) != image_size_pixels:
                raise ValueError(
                    f"Image and mask dimensions do not match:\n"
                    f"  Image: {image_path} -> {image_size_pixels}\n"
                    f"  Mask:  {source_mask_path} -> "
                    f"{(mask.shape[1], mask.shape[0])}"
                )

            binary_mask = (mask > 0).astype(np.uint8)

            prepared_image_path = prepared_images_folder / image_path.name
            prepared_mask_path = prepared_masks_folder / f"{image_path.stem}.png"

            try:
                prepared_image_path.symlink_to(image_path.resolve())
            except OSError:
                shutil.copy2(image_path, prepared_image_path)

            Image.fromarray(binary_mask, mode="L").save(
                prepared_mask_path
            )

            source_values = np.unique(mask).tolist()
            prepared_values = np.unique(binary_mask).tolist()

            print(
                f"[{index}/{len(image_paths)}] "
                f"{image_path.name} <-> {source_mask_path.name}; "
                f"source_values={source_values}, "
                f"prepared_values={prepared_values}"
            )

    dataset_yaml = f"""
path: {semantic_dataset_root}
train: images/train
val: images/val
test: images/test
masks_dir: masks

names:
  0: background
  1: adenovirus
"""

    dataset_yaml_path.write_text(
        dataset_yaml.strip() + "\n",
        encoding="utf-8",
    )

    print("\nPrepared semantic dataset:")
    print(semantic_dataset_root)
    print(f"Dataset YAML: {dataset_yaml_path}")


def validate_prepared_dataset() -> None:
    """
    Validate the prepared semantic-segmentation dataset.

    Every prepared mask must contain only class IDs 0 and 1.
    """
    if not dataset_yaml_path.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {dataset_yaml_path}"
        )

    total_images = 0
    total_foreground = 0
    total_pixels = 0

    for split in ("train", "val", "test"):
        images_folder = semantic_dataset_root / "images" / split
        masks_folder = semantic_dataset_root / "masks" / split

        if not images_folder.exists():
            raise FileNotFoundError(
                f"Image folder not found: {images_folder}"
            )

        if not masks_folder.exists():
            raise FileNotFoundError(
                f"Mask folder not found: {masks_folder}"
            )

        image_paths = sorted(
            path
            for path in images_folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        mask_paths = sorted(
            path
            for path in masks_folder.iterdir()
            if path.is_file()
            and path.suffix.lower() == ".png"
        )

        if not image_paths:
            raise RuntimeError(
                f"No images found in: {images_folder}"
            )

        image_stems = {path.stem for path in image_paths}
        mask_lookup = {path.stem: path for path in mask_paths}

        missing_masks = sorted(
            stem
            for stem in image_stems
            if stem not in mask_lookup
        )

        extra_masks = sorted(
            stem
            for stem in mask_lookup
            if stem not in image_stems
        )

        if missing_masks:
            raise FileNotFoundError(
                f"{len(missing_masks)} image(s) in '{split}' "
                f"have no matching mask:\n"
                + "\n".join(f"  - {stem}" for stem in missing_masks)
            )

        if extra_masks:
            print(
                f"Warning: {len(extra_masks)} extra mask(s) in '{split}' "
                f"have no matching image."
            )

        split_foreground = 0
        split_pixels = 0

        print("\n" + "-" * 70)
        print(f"Validating split: {split}")
        print(f"Images: {images_folder}")
        print(f"Masks:  {masks_folder}")
        print("-" * 70)

        for index, image_path in enumerate(image_paths, start=1):
            mask_path = mask_lookup[image_path.stem]

            mask = np.asarray(Image.open(mask_path).convert("L"))
            unique_values = np.unique(mask)

            invalid_values = [
                int(value)
                for value in unique_values
                if value not in (0, 1)
            ]

            if invalid_values:
                raise ValueError(
                    f"Prepared mask contains values other than 0 and 1:\n"
                    f"  Mask: {mask_path}\n"
                    f"  Values: {unique_values.tolist()}"
                )

            foreground_pixels = int((mask == 1).sum())
            pixel_count = int(mask.size)

            split_foreground += foreground_pixels
            split_pixels += pixel_count

            print(
                f"[{index}/{len(image_paths)}] "
                f"{image_path.name} <-> {mask_path.name}; "
                f"values={unique_values.tolist()}, "
                f"foreground={foreground_pixels:,}"
            )

        total_images += len(image_paths)
        total_foreground += split_foreground
        total_pixels += split_pixels

        print(
            f"Validated {len(image_paths)} {split} image-mask pairs. "
            f"Foreground proportion: "
            f"{split_foreground / max(split_pixels, 1):.8f}"
        )

    print("\n" + "=" * 70)
    print("DATASET VALIDATION COMPLETED")
    print("=" * 70)
    print(f"Dataset YAML: {dataset_yaml_path}")
    print(f"Total images: {total_images}")
    print(
        "Overall foreground proportion: "
        f"{total_foreground / max(total_pixels, 1):.8f}"
    )


# =========================================================
# METRIC HELPERS
# =========================================================

fieldnames = [
    "Model", "Weights", "Split", "mIoU", "Foreground IoU", "Dice",
    "Pixel precision", "Pixel recall", "Specificity", "Pixel accuracy",
    "True positives", "True negatives", "False positives", "False negatives",
    "Parameters", "Inference time (ms/image)", "Training time (seconds)",
    "Training time (minutes)", "Training time (hours)", "Best model path",
    "Evaluation output directory",
]


def write_results_csv(csv_path: Path, rows: list[dict]) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def count_parameters(model: YOLO) -> int:
    return sum(p.numel() for p in model.model.parameters())


def safe_float(value, default: float = 0.0) -> float:
    try:
        if hasattr(value, "item"):
            value = value.item()
        return float(value)
    except (TypeError, ValueError):
        return default


def get_inference_time(metrics) -> float:
    speed = getattr(metrics, "speed", {})
    return safe_float(speed.get("inference", 0.0)) if isinstance(speed, dict) else 0.0


def get_miou(metrics) -> float:
    value = getattr(metrics, "miou", None)
    if value is not None:
        return safe_float(value)
    results_dict = getattr(metrics, "results_dict", {})
    if isinstance(results_dict, dict):
        for key, value in results_dict.items():
            if "miou" in key.lower().replace(" ", ""):
                return safe_float(value)
    return 0.0


def get_pixel_accuracy(metrics) -> float:
    for attribute in ("pixel_accuracy", "pixel_acc", "pixacc"):
        value = getattr(metrics, attribute, None)
        if value is not None:
            return safe_float(value)
    results_dict = getattr(metrics, "results_dict", {})
    if isinstance(results_dict, dict):
        for key, value in results_dict.items():
            normalized = key.lower().replace(" ", "").replace("_", "").replace("-", "")
            if "pixelaccuracy" in normalized or "pixelacc" in normalized or "pixacc" in normalized:
                return safe_float(value)
    return 0.0


def extract_binary_confusion_counts(metrics) -> tuple[float, float, float, float]:
    cm_object = getattr(metrics, "confusion_matrix", None)
    if cm_object is None:
        raise RuntimeError("Validation results did not expose a confusion matrix")
    matrix = getattr(cm_object, "matrix", None)
    if matrix is None:
        raise RuntimeError("Validation confusion matrix is unavailable")
    matrix = matrix.detach().cpu().numpy() if torch.is_tensor(matrix) else np.asarray(matrix)
    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        raise RuntimeError(f"Expected at least 2x2 confusion matrix, got {matrix.shape}")

    # Ultralytics semantic confusion matrix is exposed as [predicted, ground truth].
    tn = float(matrix[0, 0])
    fn = float(matrix[0, 1])
    fp = float(matrix[1, 0])
    tp = float(matrix[1, 1])
    return tn, fn, fp, tp


def calculate_binary_metrics(tn: float, fn: float, fp: float, tp: float) -> dict[str, float]:
    return {
        "Foreground IoU": tp / (tp + fp + fn + EPSILON),
        "Dice": (2.0 * tp) / (2.0 * tp + fp + fn + EPSILON),
        "Pixel precision": tp / (tp + fp + EPSILON),
        "Pixel recall": tp / (tp + fn + EPSILON),
        "Specificity": tn / (tn + fp + EPSILON),
        "Pixel accuracy calculated": (tp + tn) / (tp + tn + fp + fn + EPSILON),
    }


# =========================================================
# EVALUATION
# =========================================================

def evaluate_model(
    trained_model: YOLO,
    split: str,
    model_name: str,
    weights: str,
    training_seconds: float,
    parameter_count: int,
    best_model_path: Path,
    evaluation_batch_size: int,
) -> dict:
    evaluation = trained_model.val(
        task="semantic",
        data=str(dataset_yaml_path),
        split=split,
        imgsz=image_size,
        batch=evaluation_batch_size,
        device=device,
        plots=True,
        save_json=True,
        project=str(results_folder / split),
        name=f"{model_name}_{split}",
        exist_ok=True,
        verbose=True,
    )

    tn, fn, fp, tp = extract_binary_confusion_counts(evaluation)
    binary = calculate_binary_metrics(tn, fn, fp, tp)
    reported_pixel_accuracy = get_pixel_accuracy(evaluation)
    pixel_accuracy = (
        reported_pixel_accuracy
        if reported_pixel_accuracy > 0.0
        else binary["Pixel accuracy calculated"]
    )

    return {
        "Model": model_name,
        "Weights": weights,
        "Split": split,
        "mIoU": round(get_miou(evaluation), 6),
        "Foreground IoU": round(binary["Foreground IoU"], 6),
        "Dice": round(binary["Dice"], 6),
        "Pixel precision": round(binary["Pixel precision"], 6),
        "Pixel recall": round(binary["Pixel recall"], 6),
        "Specificity": round(binary["Specificity"], 6),
        "Pixel accuracy": round(pixel_accuracy, 6),
        "True positives": int(round(tp)),
        "True negatives": int(round(tn)),
        "False positives": int(round(fp)),
        "False negatives": int(round(fn)),
        "Parameters": parameter_count,
        "Inference time (ms/image)": round(get_inference_time(evaluation), 4),
        "Training time (seconds)": round(training_seconds, 2),
        "Training time (minutes)": round(training_seconds / 60, 2),
        "Training time (hours)": round(training_seconds / 3600, 3),
        "Best model path": str(best_model_path),
        "Evaluation output directory": str(getattr(evaluation, "save_dir", "")),
    }


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    models_folder.mkdir(parents=True, exist_ok=True)
    results_folder.mkdir(parents=True, exist_ok=True)
    runs_folder.mkdir(parents=True, exist_ok=True)

    prepare_semantic_dataset()
    validate_prepared_dataset()

    print(f"\nDevice: {device}")
    print(f"Models: {', '.join(models_to_train.keys())}")
    print(f"Augmentation profile: {augmentation_name}")

    validation_results = []
    test_results = []
    failed_models = []

    for model_name, weights in models_to_train.items():
        current_batch_size = model_batch_sizes[model_name]
        model = None
        trained_model = None

        print("\n" + "=" * 70)
        print(f"Training {model_name}")
        print(f"Initial weights: {weights}")
        print(f"Batch size: {current_batch_size}")
        print("=" * 70)

        try:
            model = YOLO(weights, task="semantic")
            parameter_count = count_parameters(model)
            print(f"Parameters: {parameter_count:,}")

            start_time = time.perf_counter()
            training_result = model.train(
                task="semantic",
                data=str(dataset_yaml_path),
                epochs=epochs,
                imgsz=image_size,
                batch=current_batch_size,
                patience=patience,
                workers=workers,
                project=str(runs_folder),
                name=f"adenovirus_{model_name}",
                exist_ok=True,
                device=device,
                seed=random_seed,
                deterministic=True,
                optimizer="auto",
                plots=True,
                save=True,
                verbose=True,
                **augmentation_settings,
            )
            training_seconds = time.perf_counter() - start_time

            training_directory = Path(training_result.save_dir)
            best_model_source = training_directory / "weights" / "best.pt"
            if not best_model_source.exists():
                raise FileNotFoundError(f"Best weights not found: {best_model_source}")

            best_model_destination = models_folder / f"best_adenovirus_{model_name}.pt"
            shutil.copy2(best_model_source, best_model_destination)

            trained_model = YOLO(str(best_model_destination), task="semantic")
            parameter_count = count_parameters(trained_model)

            validation_row = evaluate_model(
                trained_model, "val", model_name, weights, training_seconds,
                parameter_count, best_model_destination, current_batch_size,
            )
            validation_results.append(validation_row)
            write_results_csv(validation_csv, validation_results)

            test_row = evaluate_model(
                trained_model, "test", model_name, weights, training_seconds,
                parameter_count, best_model_destination, current_batch_size,
            )
            test_results.append(test_row)
            write_results_csv(test_csv, test_results)

            print(f"\n{model_name} test results")
            print(f"mIoU:          {test_row['mIoU']:.6f}")
            print(f"Foreground IoU:{test_row['Foreground IoU']:.6f}")
            print(f"Dice:          {test_row['Dice']:.6f}")
            print(f"Precision:     {test_row['Pixel precision']:.6f}")
            print(f"Recall:        {test_row['Pixel recall']:.6f}")
            print(f"Specificity:   {test_row['Specificity']:.6f}")
            print(f"Pixel accuracy:{test_row['Pixel accuracy']:.6f}")

        except Exception as error:
            print(f"{model_name} failed")
            print(f"Reason: {error}")
            failed_models.append({"Model": model_name, "Weights": weights, "Error": str(error)})
            with failures_csv.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=["Model", "Weights", "Error"])
                writer.writeheader()
                writer.writerows(failed_models)

        finally:
            if model is not None:
                del model
            if trained_model is not None:
                del trained_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("YOLO26 SEMANTIC-SEGMENTATION COMPARISON COMPLETED")
    print("=" * 70)
    print(f"Validation CSV: {validation_csv}")
    print(f"Test CSV: {test_csv}")
    if failed_models:
        print(f"Failed-model report: {failures_csv}")

    print("\nTest-set comparison")
    header = (
        f"{'Model':<12}{'mIoU':<12}{'Fg IoU':<12}{'Dice':<12}"
        f"{'Precision':<12}{'Recall':<12}{'Specificity':<14}"
        f"{'PixAcc':<12}{'Parameters':<14}{'Infer ms':<12}{'Train h':<12}"
    )
    print(header)
    print("-" * len(header))
    for row in test_results:
        print(
            f"{row['Model']:<12}{row['mIoU']:<12.6f}{row['Foreground IoU']:<12.6f}"
            f"{row['Dice']:<12.6f}{row['Pixel precision']:<12.6f}"
            f"{row['Pixel recall']:<12.6f}{row['Specificity']:<14.6f}"
            f"{row['Pixel accuracy']:<12.6f}{row['Parameters']:<14,}"
            f"{row['Inference time (ms/image)']:<12.4f}"
            f"{row['Training time (hours)']:<12.3f}"
        )


if __name__ == "__main__":
    main()
