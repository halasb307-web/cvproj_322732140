from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def fit_image(image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    """
    Resize an image while preserving its aspect ratio,
    then place it on a white canvas of the requested size.
    """
    target_width, target_height = target_size

    image = image.convert("RGB")
    image.thumbnail((target_width, target_height))

    canvas = Image.new("RGB", target_size, "white")

    x = (target_width - image.width) // 2
    y = (target_height - image.height) // 2

    canvas.paste(image, (x, y))
    return canvas


def create_comparison(
    input_path: Path,
    aod_path: Path,
    ffa_path: Path,
    output_path: Path,
) -> None:
    for path in (input_path, aod_path, ffa_path):
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

    input_image = Image.open(input_path)
    aod_image = Image.open(aod_path)
    ffa_image = Image.open(ffa_path)

    panel_size = (600, 450)
    title_height = 60
    margin = 20

    input_panel = fit_image(input_image, panel_size)
    aod_panel = fit_image(aod_image, panel_size)
    ffa_panel = fit_image(ffa_image, panel_size)

    total_width = panel_size[0] * 3 + margin * 4
    total_height = panel_size[1] + title_height + margin * 2

    comparison = Image.new(
        "RGB",
        (total_width, total_height),
        "white",
    )

    draw = ImageDraw.Draw(comparison)
    font = ImageFont.load_default()

    titles = ["Input Hazy Image", "AOD-Net", "FFA-Net"]
    panels = [input_panel, aod_panel, ffa_panel]

    for index, (title, panel) in enumerate(zip(titles, panels)):
        x = margin + index * (panel_size[0] + margin)
        y = margin + title_height

        comparison.paste(panel, (x, y))

        title_box = draw.textbbox((0, 0), title, font=font)
        title_width = title_box[2] - title_box[0]

        title_x = x + (panel_size[0] - title_width) // 2
        title_y = margin + 20

        draw.text(
            (title_x, title_y),
            title,
            fill="black",
            font=font,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.save(output_path)

    print(f"Comparison saved to: {output_path}")


def main() -> None:
    create_comparison(
        input_path=Path("src/aod_net/sample_images/hazy.jpg"),
        aod_path=Path("results/aod_net/aod_output.png"),
        ffa_path=Path("results/ffa_net/ffa_output.png"),
        output_path=Path("results/comparisons/comparison_1.png"),
    )

    create_comparison(
        input_path=Path("src/aod_net/sample_images/hazy2.jpg"),
        aod_path=Path("results/aod_net/aod_output2.png"),
        ffa_path=Path("results/ffa_net/ffa_output2.png"),
        output_path=Path("results/comparisons/comparison_2.png"),
    )


if __name__ == "__main__":
    main()