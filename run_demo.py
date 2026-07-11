from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_IMAGE = (
    PROJECT_ROOT
    / "src"
    / "aod_net"
    / "sample_images"
    / "hazy.jpg"
)

AOD_SCRIPT = (
    PROJECT_ROOT
    / "src"
    / "aod_net"
    / "aod_inference.py"
)

AOD_WEIGHT = (
    PROJECT_ROOT
    / "src"
    / "aod_net"
    / "weights"
    / "AOD_net_epoch_relu_10.pth"
)

FFA_DIR = (
    PROJECT_ROOT
    / "src"
    / "ffa_net"
    / "net"
)

FFA_SCRIPT = FFA_DIR / "test.py"

FFA_WEIGHT = (
    FFA_DIR
    / "trained_models"
    / "its_train_ffa_3_19.pk"
)

FFA_TEST_DIR = FFA_DIR / "demo_test_imgs"
FFA_PREDICTION_DIR = FFA_DIR / "pred_FFA_its"

RESULTS_DIR = PROJECT_ROOT / "results" / "demo"

AOD_OUTPUT = RESULTS_DIR / "aod_demo.png"
FFA_OUTPUT = RESULTS_DIR / "ffa_demo.png"
COMPARISON_OUTPUT = RESULTS_DIR / "demo_comparison.png"


def check_required_files() -> None:
    missing_files: list[Path] = []

    required_files = [
        INPUT_IMAGE,
        AOD_SCRIPT,
        AOD_WEIGHT,
        FFA_SCRIPT,
        FFA_WEIGHT,
    ]

    for file_path in required_files:
        if not file_path.exists():
            missing_files.append(file_path)

    if missing_files:
        print("\nThe demo cannot run because files are missing:\n")

        for file_path in missing_files:
            print(f"- {file_path}")

        print(
            "\nPlease download the pretrained weights "
            "according to the instructions in README.md."
        )

        raise SystemExit(1)


def run_command(
    command: list[str],
    working_directory: Path,
    environment: dict[str, str] | None = None,
) -> None:
    completed_process = subprocess.run(
        command,
        cwd=working_directory,
        env=environment,
        text=True,
    )

    if completed_process.returncode != 0:
        raise RuntimeError(
            "The following command failed:\n"
            + " ".join(command)
        )


def run_aod_net() -> None:
    print("\nRunning AOD-Net...")

    command = [
        sys.executable,
        str(AOD_SCRIPT),
        "--input",
        str(INPUT_IMAGE),
        "--output",
        str(AOD_OUTPUT),
    ]

    run_command(
        command=command,
        working_directory=PROJECT_ROOT,
    )

    print(f"AOD-Net output saved to: {AOD_OUTPUT}")


def prepare_ffa_input() -> None:
    if FFA_TEST_DIR.exists():
        shutil.rmtree(FFA_TEST_DIR)

    FFA_TEST_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        INPUT_IMAGE,
        FFA_TEST_DIR / "demo.jpg",
    )


def run_ffa_net() -> None:
    print("\nRunning FFA-Net...")

    prepare_ffa_input()

    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"

    command = [
        sys.executable,
        str(FFA_SCRIPT),
        "--task=its",
        "--test_imgs=demo_test_imgs",
    ]

    run_command(
        command=command,
        working_directory=FFA_DIR,
        environment=environment,
    )

    generated_output = (
        FFA_PREDICTION_DIR / "demo_FFA.png"
    )

    if not generated_output.exists():
        raise FileNotFoundError(
            "FFA-Net did not create the expected output:\n"
            f"{generated_output}"
        )

    shutil.copy2(
        generated_output,
        FFA_OUTPUT,
    )

    print(f"FFA-Net output saved to: {FFA_OUTPUT}")


def resize_to_same_height(
    image: Image.Image,
    target_height: int,
) -> Image.Image:
    new_width = round(
        image.width * target_height / image.height
    )

    return image.resize(
        (new_width, target_height),
        Image.Resampling.LANCZOS,
    )


def create_comparison() -> None:
    print("\nCreating comparison image...")

    images = [
        Image.open(INPUT_IMAGE).convert("RGB"),
        Image.open(AOD_OUTPUT).convert("RGB"),
        Image.open(FFA_OUTPUT).convert("RGB"),
    ]

    titles = [
        "Input Hazy Image",
        "AOD-Net",
        "FFA-Net",
    ]

    target_height = min(
        image.height for image in images
    )

    resized_images = [
        resize_to_same_height(
            image,
            target_height,
        )
        for image in images
    ]

    title_height = 55
    spacing = 20

    total_width = (
        sum(image.width for image in resized_images)
        + spacing * (len(resized_images) - 1)
    )

    canvas = Image.new(
        "RGB",
        (
            total_width,
            target_height + title_height,
        ),
        "white",
    )

    draw = ImageDraw.Draw(canvas)

    current_x = 0

    for title, image in zip(
        titles,
        resized_images,
    ):
        text_box = draw.textbbox(
            (0, 0),
            title,
        )

        text_width = text_box[2] - text_box[0]

        text_x = (
            current_x
            + (image.width - text_width) // 2
        )

        draw.text(
            (text_x, 18),
            title,
            fill="black",
        )

        canvas.paste(
            image,
            (current_x, title_height),
        )

        current_x += image.width + spacing

    canvas.save(
        COMPARISON_OUTPUT,
        quality=95,
    )

    print(
        "Comparison image saved to: "
        f"{COMPARISON_OUTPUT}"
    )


def clean_temporary_files() -> None:
    if FFA_TEST_DIR.exists():
        shutil.rmtree(FFA_TEST_DIR)


def main() -> None:
    print("=" * 55)
    print("Single Image Dehazing Demo")
    print("AOD-Net vs. FFA-Net")
    print("=" * 55)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    check_required_files()

    try:
        run_aod_net()
        run_ffa_net()
        create_comparison()

    finally:
        clean_temporary_files()

    print("\nDemo completed successfully.")
    print(f"Open this file:\n{COMPARISON_OUTPUT}")


if __name__ == "__main__":
    main()