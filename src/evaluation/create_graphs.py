from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# נתיבי הקבצים
METRICS_CSV_PATH = Path(
    "results/comparisons/quantitative_metrics.csv"
)

RUNTIME_CSV_PATH = Path(
    "results/comparisons/runtime_summary.csv"
)

GRAPHS_DIR = Path("results/graphs")

FINAL_SUMMARY_PATH = Path(
    "results/comparisons/final_summary.csv"
)


def load_metrics() -> pd.DataFrame:
    """
    Load the per-image PSNR and SSIM results.
    """
    if not METRICS_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Metrics file was not found: {METRICS_CSV_PATH}"
        )

    metrics_df = pd.read_csv(METRICS_CSV_PATH)

    required_columns = {
        "model",
        "image",
        "psnr",
        "ssim",
    }

    missing_columns = (
        required_columns - set(metrics_df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing columns in quantitative metrics CSV: "
            f"{sorted(missing_columns)}"
        )

    if metrics_df.empty:
        raise ValueError(
            "The quantitative metrics CSV file is empty."
        )

    metrics_df["psnr"] = pd.to_numeric(
        metrics_df["psnr"],
        errors="raise",
    )

    metrics_df["ssim"] = pd.to_numeric(
        metrics_df["ssim"],
        errors="raise",
    )

    return metrics_df


def load_runtime_summary() -> pd.DataFrame:
    """
    Load the average runtime results.
    """
    if not RUNTIME_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Runtime summary file was not found: "
            f"{RUNTIME_CSV_PATH}"
        )

    runtime_df = pd.read_csv(RUNTIME_CSV_PATH)

    required_columns = {
        "model",
        "number_of_images",
        "average_runtime_seconds",
    }

    missing_columns = (
        required_columns - set(runtime_df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing columns in runtime summary CSV: "
            f"{sorted(missing_columns)}"
        )

    if runtime_df.empty:
        raise ValueError(
            "The runtime summary CSV file is empty."
        )

    runtime_df["average_runtime_seconds"] = (
        pd.to_numeric(
            runtime_df["average_runtime_seconds"],
            errors="raise",
        )
    )

    return runtime_df


def calculate_metric_averages(
    metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate average PSNR and SSIM for each model.
    """
    averages_df = (
        metrics_df.groupby("model", as_index=False)
        .agg(
            average_psnr=("psnr", "mean"),
            average_ssim=("ssim", "mean"),
        )
    )

    return averages_df


def create_final_summary(
    averages_df: pd.DataFrame,
    runtime_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge quality metrics and runtime results.
    """
    final_summary_df = averages_df.merge(
        runtime_df[
            [
                "model",
                "number_of_images",
                "average_runtime_seconds",
            ]
        ],
        on="model",
        how="left",
    )

    if (
        final_summary_df[
            "average_runtime_seconds"
        ].isna().any()
    ):
        raise ValueError(
            "Runtime data could not be matched "
            "with one or more models."
        )

    return final_summary_df


def add_value_labels(
    bars,
    values,
    decimal_places: int,
) -> None:
    """
    Add numeric values above bar chart columns.
    """
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.{decimal_places}f}",
            ha="center",
            va="bottom",
        )


def create_psnr_graph(
    summary_df: pd.DataFrame,
) -> None:
    """
    Create the average PSNR comparison graph.
    """
    output_path = GRAPHS_DIR / "average_psnr.png"

    plt.figure(figsize=(8, 6))

    bars = plt.bar(
        summary_df["model"],
        summary_df["average_psnr"],
    )

    plt.title("Average PSNR Comparison")
    plt.xlabel("Model")
    plt.ylabel("Average PSNR (dB)")
    plt.grid(axis="y", alpha=0.3)

    add_value_labels(
        bars,
        summary_df["average_psnr"],
        decimal_places=4,
    )

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"PSNR graph saved to: {output_path}")


def create_ssim_graph(
    summary_df: pd.DataFrame,
) -> None:
    """
    Create the average SSIM comparison graph.
    """
    output_path = GRAPHS_DIR / "average_ssim.png"

    plt.figure(figsize=(8, 6))

    bars = plt.bar(
        summary_df["model"],
        summary_df["average_ssim"],
    )

    plt.title("Average SSIM Comparison")
    plt.xlabel("Model")
    plt.ylabel("Average SSIM")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", alpha=0.3)

    add_value_labels(
        bars,
        summary_df["average_ssim"],
        decimal_places=6,
    )

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"SSIM graph saved to: {output_path}")


def create_runtime_graph(
    summary_df: pd.DataFrame,
) -> None:
    """
    Create the average inference runtime comparison graph.
    """
    output_path = (
        GRAPHS_DIR / "average_runtime.png"
    )

    plt.figure(figsize=(8, 6))

    bars = plt.bar(
        summary_df["model"],
        summary_df["average_runtime_seconds"],
    )

    plt.title("Average Inference Runtime Comparison")
    plt.xlabel("Model")
    plt.ylabel("Average Runtime (seconds)")
    plt.grid(axis="y", alpha=0.3)

    add_value_labels(
        bars,
        summary_df["average_runtime_seconds"],
        decimal_places=4,
    )

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(
        f"Runtime graph saved to: {output_path}"
    )


def save_final_summary(
    summary_df: pd.DataFrame,
) -> None:
    """
    Save the combined quality and runtime summary.
    """
    FINAL_SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_df.to_csv(
        FINAL_SUMMARY_PATH,
        index=False,
        float_format="%.6f",
    )

    print(
        f"Final summary saved to: "
        f"{FINAL_SUMMARY_PATH}"
    )


def print_summary(
    summary_df: pd.DataFrame,
) -> None:
    """
    Print the final results in the terminal.
    """
    print("\nFinal average results:\n")

    display_df = summary_df.copy()

    display_df["average_psnr"] = (
        display_df["average_psnr"]
        .map(lambda value: f"{value:.4f}")
    )

    display_df["average_ssim"] = (
        display_df["average_ssim"]
        .map(lambda value: f"{value:.6f}")
    )

    display_df["average_runtime_seconds"] = (
        display_df["average_runtime_seconds"]
        .map(lambda value: f"{value:.4f}")
    )

    print(display_df.to_string(index=False))


def main() -> None:
    GRAPHS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_df = load_metrics()
    runtime_df = load_runtime_summary()

    averages_df = calculate_metric_averages(
        metrics_df
    )

    final_summary_df = create_final_summary(
        averages_df,
        runtime_df,
    )

    print_summary(final_summary_df)

    create_psnr_graph(final_summary_df)
    create_ssim_graph(final_summary_df)
    create_runtime_graph(final_summary_df)

    save_final_summary(final_summary_df)


if __name__ == "__main__":
    main()