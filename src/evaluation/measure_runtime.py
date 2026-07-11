from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

HAZY_DIR = (
    PROJECT_ROOT
    / "src"
    / "evaluation"
    / "test_data"
    / "hazy"
)

AOD_SCRIPT = (
    PROJECT_ROOT
    / "src"
    / "aod_net"
    / "aod_inference.py"
)

FFA_DIR = (
    PROJECT_ROOT
    / "src"
    / "ffa_net"
    / "net"
)

FFA_TEST_SCRIPT = FFA_DIR / "test.py"

TEMP_DIR = (
    PROJECT_ROOT
    / "src"
    / "evaluation"
    / "runtime_temp"
)

AOD_TEMP_OUTPUT = TEMP_DIR / "aod_output.png"
FFA_TEMP_INPUT_DIR = FFA_DIR / "runtime_test_imgs"

RESULTS_CSV = (
    PROJECT_ROOT
    / "results"
    / "comparisons"
    / "runtime_results.csv"
)

SUMMARY_CSV = (
    PROJECT_ROOT
    / "results"
    / "comparisons"
    / "runtime_summary.csv"
)

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def run_command(
    command: list[str],
    working_directory: Path,
    environment: dict[str, str] | None = None,
) -> float:
    start_time = time.perf_counter()

    completed_process = subprocess.run(
        command,
        cwd=working_directory,
        env=environment,
        capture_output=True,
        text=True,
    )

    elapsed_time = time.perf_counter() - start_time

    if completed_process.returncode != 0:
        print(completed_process.stdout)
        print(completed_process.stderr)

        raise RuntimeError(
            "Command failed:\n" + " ".join(command)
        )

    return elapsed_time


def benchmark_aod(image_path: Path) -> float:
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(AOD_SCRIPT),
        "--input",
        str(image_path),
        "--output",
        str(AOD_TEMP_OUTPUT),
    ]

    return run_command(
        command=command,
        working_directory=PROJECT_ROOT,
    )


def prepare_ffa_input(image_path: Path) -> None:
    if FFA_TEMP_INPUT_DIR.exists():
        shutil.rmtree(FFA_TEMP_INPUT_DIR)

    FFA_TEMP_INPUT_DIR.mkdir(parents=True, exist_ok=True)

    destination = FFA_TEMP_INPUT_DIR / image_path.name
    shutil.copy2(image_path, destination)


def benchmark_ffa(image_path: Path) -> float:
    prepare_ffa_input(image_path)

    environment = os.environ.copy()

    # Prevent Matplotlib windows from stopping the benchmark.
    environment["MPLBACKEND"] = "Agg"

    command = [
        sys.executable,
        str(FFA_TEST_SCRIPT),
        "--task=its",
        "--test_imgs=runtime_test_imgs",
    ]

    return run_command(
        command=command,
        working_directory=FFA_DIR,
        environment=environment,
    )


def save_detailed_results(
    results: list[dict[str, str | float]],
) -> None:
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)

    with RESULTS_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["model", "image", "runtime_seconds"],
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nRuntime results saved to: {RESULTS_CSV}")


def save_summary(
    results: list[dict[str, str | float]],
) -> None:
    models = sorted(
        {str(result["model"]) for result in results}
    )

    summary_rows: list[dict[str, str | float | int]] = []

    for model in models:
        model_times = [
            float(result["runtime_seconds"])
            for result in results
            if result["model"] == model
        ]

        average_time = sum(model_times) / len(model_times)

        summary_rows.append(
            {
                "model": model,
                "number_of_images": len(model_times),
                "average_runtime_seconds": average_time,
            }
        )

        print(
            f"{model} average runtime: "
            f"{average_time:.6f} seconds"
        )

    with SUMMARY_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "model",
                "number_of_images",
                "average_runtime_seconds",
            ],
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Runtime summary saved to: {SUMMARY_CSV}")


def clean_temporary_files() -> None:
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)

    if FFA_TEMP_INPUT_DIR.exists():
        shutil.rmtree(FFA_TEMP_INPUT_DIR)


def main() -> None:
    images = sorted(
        path
        for path in HAZY_DIR.iterdir()
        if path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not images:
        raise FileNotFoundError(
            f"No test images were found in: {HAZY_DIR}"
        )

    results: list[dict[str, str | float]] = []

    try:
        for image_path in images:
            print(f"\nTesting image: {image_path.name}")

            aod_time = benchmark_aod(image_path)

            results.append(
                {
                    "model": "AOD-Net",
                    "image": image_path.name,
                    "runtime_seconds": aod_time,
                }
            )

            print(
                f"AOD-Net runtime: {aod_time:.6f} seconds"
            )

            ffa_time = benchmark_ffa(image_path)

            results.append(
                {
                    "model": "FFA-Net",
                    "image": image_path.name,
                    "runtime_seconds": ffa_time,
                }
            )

            print(
                f"FFA-Net runtime: {ffa_time:.6f} seconds"
            )

        save_detailed_results(results)

        print()
        save_summary(results)

    finally:
        clean_temporary_files()


if __name__ == "__main__":
    main()