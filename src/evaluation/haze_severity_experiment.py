from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from skimage.transform import resize


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXPERIMENT_DIR = (
    PROJECT_ROOT
    / "src"
    / "evaluation"
    / "haze_severity_data"
)

HAZY_DIR = EXPERIMENT_DIR / "hazy"
CLEAN_DIR = EXPERIMENT_DIR / "clean"

OUTPUTS_DIR = EXPERIMENT_DIR / "outputs"
AOD_OUTPUT_DIR = OUTPUTS_DIR / "aod_net"
FFA_OUTPUT_DIR = OUTPUTS_DIR / "ffa_net"

AOD_SCRIPT = (
    PROJECT_ROOT
    / "src"
    / "aod_net"
    / "aod_inference.py"
)

FFA_NET_DIR = (
    PROJECT_ROOT
    / "src"
    / "ffa_net"
    / "net"
)

FFA_TEST_SCRIPT = FFA_NET_DIR / "test.py"
FFA_TEMP_INPUT_DIR = FFA_NET_DIR / "severity_test_imgs"
FFA_PREDICTION_DIR = FFA_NET_DIR / "pred_FFA_its"

RESULTS_CSV = (
    PROJECT_ROOT
    / "results"
    / "comparisons"
    / "haze_severity_results.csv"
)

PSNR_GRAPH_PATH = (
    PROJECT_ROOT
    / "results"
    / "graphs"
    / "haze_severity_psnr.png"
)

SSIM_GRAPH_PATH = (
    PROJECT_ROOT
    / "results"
    / "graphs"
    / "haze_severity_ssim.png"
)

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def create_required_directories() -> None:
    HAZY_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    AOD_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FFA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    PSNR_GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)


def find_test_images() -> list[Path]:
    images = sorted(
        image_path
        for image_path in HAZY_DIR.iterdir()
        if image_path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not images:
        raise FileNotFoundError(
            "No haze-severity images were found in:\n"
            f"{HAZY_DIR}\n\n"
            "Add images such as 1442_1.png through "
            "1442_5.png and run the script again."
        )

    return images


def find_clean_image(hazy_image: Path) -> Path:
    same_name_path = CLEAN_DIR / hazy_image.name

    if same_name_path.exists():
        return same_name_path

    base_scene_name = hazy_image.stem.split("_")[0]

    for extension in SUPPORTED_EXTENSIONS:
        candidate = CLEAN_DIR / f"{base_scene_name}{extension}"

        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"No clean ground-truth image was found for "
        f"{hazy_image.name}.\n"
        f"Expected either {hazy_image.name} or "
        f"{base_scene_name}.png inside:\n{CLEAN_DIR}"
    )


def run_command(
    command: list[str],
    working_directory: Path,
    environment: dict[str, str] | None = None,
) -> None:
    completed_process = subprocess.run(
        command,
        cwd=working_directory,
        env=environment,
        capture_output=True,
        text=True,
    )

    if completed_process.returncode != 0:
        print(completed_process.stdout)
        print(completed_process.stderr)

        raise RuntimeError(
            "Command failed:\n" + " ".join(command)
        )


def run_aod_net(
    input_path: Path,
    output_path: Path,
) -> None:
    command = [
        sys.executable,
        str(AOD_SCRIPT),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
    ]

    run_command(
        command=command,
        working_directory=PROJECT_ROOT,
    )


def prepare_ffa_input(input_path: Path) -> None:
    if FFA_TEMP_INPUT_DIR.exists():
        shutil.rmtree(FFA_TEMP_INPUT_DIR)

    FFA_TEMP_INPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        input_path,
        FFA_TEMP_INPUT_DIR / input_path.name,
    )


def run_ffa_net(
    input_path: Path,
    output_path: Path,
) -> None:
    prepare_ffa_input(input_path)

    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"

    command = [
        sys.executable,
        str(FFA_TEST_SCRIPT),
        "--task=its",
        "--test_imgs=severity_test_imgs",
    ]

    run_command(
        command=command,
        working_directory=FFA_NET_DIR,
        environment=environment,
    )

    generated_filename = (
        f"{input_path.stem}_FFA.png"
    )

    generated_path = (
        FFA_PREDICTION_DIR / generated_filename
    )

    if not generated_path.exists():
        raise FileNotFoundError(
            "FFA-Net output was not created:\n"
            f"{generated_path}"
        )

    shutil.copy2(
        generated_path,
        output_path,
    )


def load_rgb_image(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")

    return np.asarray(
        rgb_image,
        dtype=np.float32,
    ) / 255.0


def match_image_size(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
) -> np.ndarray:
    if prediction.shape == ground_truth.shape:
        return prediction

    resized_prediction = resize(
        prediction,
        ground_truth.shape,
        preserve_range=True,
        anti_aliasing=True,
    )

    return resized_prediction.astype(np.float32)


def calculate_image_metrics(
    prediction_path: Path,
    ground_truth_path: Path,
) -> tuple[float, float]:
    prediction = load_rgb_image(prediction_path)
    ground_truth = load_rgb_image(ground_truth_path)

    prediction = match_image_size(
        prediction,
        ground_truth,
    )

    prediction = np.clip(prediction, 0.0, 1.0)
    ground_truth = np.clip(ground_truth, 0.0, 1.0)

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


def extract_haze_level(image_path: Path) -> int:
    filename_parts = image_path.stem.split("_")

    if len(filename_parts) < 2:
        raise ValueError(
            f"Cannot determine haze level from "
            f"filename: {image_path.name}"
        )

    try:
        return int(filename_parts[-1])
    except ValueError as error:
        raise ValueError(
            f"The filename must end with a numeric "
            f"haze level, for example 1442_1.png."
        ) from error


def save_results(
    results: list[dict[str, str | int | float]],
) -> None:
    with RESULTS_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "model",
                "image",
                "haze_level",
                "psnr",
                "ssim",
            ],
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {RESULTS_CSV}")


def create_metric_graph(
    results: list[dict[str, str | int | float]],
    metric_name: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    model_names = ["AOD-Net", "FFA-Net"]

    plt.figure(figsize=(9, 6))

    for model_name in model_names:
        model_results = sorted(
            (
                result
                for result in results
                if result["model"] == model_name
            ),
            key=lambda result: int(
                result["haze_level"]
            ),
        )

        haze_levels = [
            int(result["haze_level"])
            for result in model_results
        ]

        metric_values = [
            float(result[metric_name])
            for result in model_results
        ]

        plt.plot(
            haze_levels,
            metric_values,
            marker="o",
            label=model_name,
        )

        for haze_level, value in zip(
            haze_levels,
            metric_values,
        ):
            plt.text(
                haze_level,
                value,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.title(title)
    plt.xlabel("Haze Level")
    plt.ylabel(ylabel)
    plt.xticks(sorted({
        int(result["haze_level"])
        for result in results
    }))
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Graph saved to: {output_path}")


def print_summary(
    results: list[dict[str, str | int | float]],
) -> None:
    print("\nHaze-severity experiment results:\n")

    for result in results:
        print(
            f"{result['model']:7} | "
            f"{result['image']:12} | "
            f"Level: {result['haze_level']} | "
            f"PSNR: {float(result['psnr']):.4f} | "
            f"SSIM: {float(result['ssim']):.6f}"
        )


def clean_temporary_files() -> None:
    if FFA_TEMP_INPUT_DIR.exists():
        shutil.rmtree(FFA_TEMP_INPUT_DIR)


def main() -> None:
    create_required_directories()

    test_images = find_test_images()

    results: list[
        dict[str, str | int | float]
    ] = []

    try:
        for hazy_image_path in test_images:
            clean_image_path = find_clean_image(
                hazy_image_path
            )

            haze_level = extract_haze_level(
                hazy_image_path
            )

            print(
                f"\nProcessing {hazy_image_path.name} "
                f"(haze level {haze_level})"
            )

            aod_output_path = (
                AOD_OUTPUT_DIR / hazy_image_path.name
            )

            ffa_output_path = (
                FFA_OUTPUT_DIR / hazy_image_path.name
            )

            print("Running AOD-Net...")

            run_aod_net(
                input_path=hazy_image_path,
                output_path=aod_output_path,
            )

            aod_psnr, aod_ssim = (
                calculate_image_metrics(
                    prediction_path=aod_output_path,
                    ground_truth_path=clean_image_path,
                )
            )

            results.append(
                {
                    "model": "AOD-Net",
                    "image": hazy_image_path.name,
                    "haze_level": haze_level,
                    "psnr": aod_psnr,
                    "ssim": aod_ssim,
                }
            )

            print(
                f"AOD-Net: PSNR={aod_psnr:.4f}, "
                f"SSIM={aod_ssim:.6f}"
            )

            print("Running FFA-Net...")

            run_ffa_net(
                input_path=hazy_image_path,
                output_path=ffa_output_path,
            )

            ffa_psnr, ffa_ssim = (
                calculate_image_metrics(
                    prediction_path=ffa_output_path,
                    ground_truth_path=clean_image_path,
                )
            )

            results.append(
                {
                    "model": "FFA-Net",
                    "image": hazy_image_path.name,
                    "haze_level": haze_level,
                    "psnr": ffa_psnr,
                    "ssim": ffa_ssim,
                }
            )

            print(
                f"FFA-Net: PSNR={ffa_psnr:.4f}, "
                f"SSIM={ffa_ssim:.6f}"
            )

        results.sort(
            key=lambda result: (
                str(result["model"]),
                int(result["haze_level"]),
            )
        )

        save_results(results)
        print_summary(results)

        create_metric_graph(
            results=results,
            metric_name="psnr",
            ylabel="PSNR (dB)",
            title=(
                "Effect of Haze Severity on PSNR"
            ),
            output_path=PSNR_GRAPH_PATH,
        )

        create_metric_graph(
            results=results,
            metric_name="ssim",
            ylabel="SSIM",
            title=(
                "Effect of Haze Severity on SSIM"
            ),
            output_path=SSIM_GRAPH_PATH,
        )

    finally:
        clean_temporary_files()


if __name__ == "__main__":
    main()