from pathlib import Path
import argparse
import time

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from aod_model import AODNet


def load_model(weights_path: Path, device: torch.device) -> nn.Module:
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    checkpoint = torch.load(
        weights_path,
        map_location=device,
        weights_only=False,
    )

    # The checkpoint contains the complete trained PyTorch model.
    if isinstance(checkpoint, nn.Module):
        model = checkpoint.to(device)
        model.eval()
        return model

    # Support checkpoints that contain only a state_dict.
    model = AODNet().to(device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Unsupported checkpoint type: {type(checkpoint).__name__}"
        )

    cleaned_state_dict = {
        key.replace("module.", ""): value
        for key, value in checkpoint.items()
    }

    model.load_state_dict(cleaned_state_dict, strict=True)
    model.eval()

    return model


def load_image(image_path: Path, device: torch.device) -> torch.Tensor:
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")

    transform = transforms.ToTensor()
    tensor = transform(image).unsqueeze(0)

    return tensor.to(device)


def run_inference(
    model: AODNet,
    image_tensor: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    with torch.no_grad():
        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()
        output = model(image_tensor)

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed_time = time.perf_counter() - start_time

    # Limit values before saving the image.
    output = torch.clamp(output, 0.0, 1.0)

    return output, elapsed_time


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run AOD-Net inference on one hazy image."
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the hazy input image.",
    )

    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("src/aod_net/weights/AOD_net_epoch_relu_10.pth"),
        help="Path to the pretrained AOD-Net weights.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/aod_net/aod_output.png"),
        help="Path where the dehazed result will be saved.",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Input: {args.input}")
    print(f"Weights: {args.weights}")

    model = load_model(args.weights, device)
    image_tensor = load_image(args.input, device)

    output, elapsed_time = run_inference(model, image_tensor, device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_image(output.cpu(), args.output)

    print(f"Output saved to: {args.output}")
    print(f"Inference time: {elapsed_time:.6f} seconds")
    print(f"Input shape: {tuple(image_tensor.shape)}")
    print(f"Output shape: {tuple(output.shape)}")


if __name__ == "__main__":
    main()