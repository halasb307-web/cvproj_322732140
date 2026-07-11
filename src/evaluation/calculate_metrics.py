from pathlib import Path
import csv

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


def load_rgb_image(path: Path) -> np.ndarray:
    """
    Load an RGB image and return a float32 NumPy array
    with values in the range [0, 1].
    """
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


def match_image_size(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Ensure that prediction and ground truth have the same dimensions.

    If their sizes differ, the prediction is resized to the
    ground-truth dimensions.
    """
    if prediction.shape == ground_truth.shape:
        return prediction, ground_truth

    gt_height, gt_width = ground_truth.shape[:2]

    prediction_image = Image.fromarray(
        np.clip(prediction * 255.0, 0, 255).astype(np.uint8)
    )

    prediction_image = prediction_image.resize(
        (gt_width, gt_height),
        Image.Resampling.BICUBIC,
    )

    resized_prediction = (
        np.asarray(prediction_image, dtype=np.float32) / 255.0
    )

    return resized_prediction, ground_truth


def calculate_image_metrics(
    prediction_path: Path,
    ground_truth_path: Path,
) -> tuple[float, float]:
    prediction = load_rgb_image(prediction_path)
    ground_truth = load_rgb_image(ground_truth_path)

    prediction, ground_truth = match_image_size(
        prediction,
        ground_truth,
    )

    psnr_value = peak_signal_noise_ratio(
        ground_truth,
        prediction,
        data_range=1.0,
    )

    ssim_value = structural_similarity(
        ground_truth,
        prediction,
        channel_axis=2,
        data_range=1.0,
    )

    return float(psnr_value), float(ssim_value)


def evaluate_model(
    model_name: str,
    prediction_dir: Path,
    ground_truth_dir: Path,
) -> list[dict[str, str | float]]:
    results: list[dict[str, str | float]] = []

    supported_extensions = {".png", ".jpg", ".jpeg"}

    prediction_files = sorted(
        path
        for path in prediction_dir.iterdir()
        if path.suffix.lower() in supported_extensions
    )

    if not prediction_files:
        print(f"No prediction images found for {model_name}.")
        return results

    for prediction_path in prediction_files:
        ground_truth_path = ground_truth_dir / prediction_path.name

        if not ground_truth_path.exists():
            print(
                f"Skipping {prediction_path.name}: "
                "matching ground-truth image was not found."
            )
            continue

        psnr_value, ssim_value = calculate_image_metrics(
            prediction_path,
            ground_truth_path,
        )

        results.append(
            {
                "model": model_name,
                "image": prediction_path.name,
                "psnr": psnr_value,
                "ssim": ssim_value,
            }
        )

        print(
            f"{model_name} | {prediction_path.name} | "
            f"PSNR: {psnr_value:.4f} | "
            f"SSIM: {ssim_value:.6f}"
        )

    return results


def save_results(
    results: list[dict[str, str | float]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["model", "image", "psnr", "ssim"],
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nMetrics saved to: {output_path}")


def print_model_average(
    model_name: str,
    results: list[dict[str, str | float]],
) -> None:
    model_results = [
        result
        for result in results
        if result["model"] == model_name
    ]

    if not model_results:
        print(f"No valid results available for {model_name}.")
        return

    average_psnr = np.mean(
        [float(result["psnr"]) for result in model_results]
    )

    average_ssim = np.mean(
        [float(result["ssim"]) for result in model_results]
    )

    print(
        f"{model_name} average | "
        f"PSNR: {average_psnr:.4f} | "
        f"SSIM: {average_ssim:.6f}"
    )


def main() -> None:
    base_dir = Path("src/evaluation/test_data")

    ground_truth_dir = base_dir / "clean"
    aod_output_dir = base_dir / "outputs" / "aod_net"
    ffa_output_dir = base_dir / "outputs" / "ffa_net"

    for directory in (
        ground_truth_dir,
        aod_output_dir,
        ffa_output_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    all_results: list[dict[str, str | float]] = []

    all_results.extend(
        evaluate_model(
            model_name="AOD-Net",
            prediction_dir=aod_output_dir,
            ground_truth_dir=ground_truth_dir,
        )
    )

    all_results.extend(
        evaluate_model(
            model_name="FFA-Net",
            prediction_dir=ffa_output_dir,
            ground_truth_dir=ground_truth_dir,
        )
    )

    save_results(
        all_results,
        Path("results/comparisons/quantitative_metrics.csv"),
    )

    print()
    print_model_average("AOD-Net", all_results)
    print_model_average("FFA-Net", all_results)


if __name__ == "__main__":
    main()