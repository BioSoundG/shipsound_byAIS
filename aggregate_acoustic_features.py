"""Aggregate multiple acoustic_features.csv files and generate summary plots."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd

from acoustic_features import plot_feature_vs_vessel_params


DEFAULT_PATTERN = "acoustic_features.csv"


def find_acoustic_feature_csvs(inputs: list[str], pattern: str = DEFAULT_PATTERN) -> list[Path]:
    """Find acoustic feature CSV files from file and directory inputs."""
    paths: list[Path] = []

    for input_path in inputs:
        path = Path(input_path)
        if path.is_file():
            paths.append(path)
            continue
        if path.is_dir():
            paths.extend(path.rglob(pattern))
            continue
        print(f"Warning: input path does not exist and will be skipped: {path}")

    unique_paths = sorted({p.resolve() for p in paths})
    return unique_paths


def read_and_combine_acoustic_features(csv_paths: list[Path]) -> pd.DataFrame:
    """Read acoustic feature CSV files and combine them into one DataFrame."""
    frames = []

    for csv_path in csv_paths:
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"Warning: failed to read {csv_path}: {exc}")
            continue

        if df.empty:
            print(f"Warning: empty CSV skipped: {csv_path}")
            continue

        df["source_file"] = str(csv_path)
        df["source_dir"] = str(csv_path.parent)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    if "min_distance_time" in combined.columns:
        combined["min_distance_time"] = pd.to_datetime(
            combined["min_distance_time"], errors="coerce"
        )
    return combined


def filter_by_speed_range(
    df: pd.DataFrame,
    speed_min: float | None = None,
    speed_max: float | None = None,
) -> pd.DataFrame:
    """Filter rows by vessel_speed_knots when a speed range is specified."""
    if speed_min is None and speed_max is None:
        return df
    if "vessel_speed_knots" not in df.columns:
        raise KeyError("vessel_speed_knots column is required for speed filtering.")
    if speed_min is not None and speed_max is not None and speed_min > speed_max:
        raise ValueError("--speed-min must be less than or equal to --speed-max.")

    filtered = df.copy()
    speed = pd.to_numeric(filtered["vessel_speed_knots"], errors="coerce")
    mask = speed.notna()
    if speed_min is not None:
        mask &= speed >= speed_min
    if speed_max is not None:
        mask &= speed <= speed_max

    return filtered.loc[mask].copy()


def save_summary(
    combined_df: pd.DataFrame,
    csv_paths: list[Path],
    output_dir: Path,
    loaded_rows: int,
    speed_min: float | None = None,
    speed_max: float | None = None,
) -> None:
    """Save a small text summary of the aggregation."""
    summary_path = output_dir / "combined_acoustic_features_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Combined acoustic feature analysis\n")
        f.write("=" * 40 + "\n")
        f.write(f"Input CSV files: {len(csv_paths)}\n")
        f.write(f"Loaded rows: {loaded_rows}\n")
        f.write(f"Analyzed rows: {len(combined_df)}\n")
        if speed_min is not None or speed_max is not None:
            min_label = "-inf" if speed_min is None else speed_min
            max_label = "inf" if speed_max is None else speed_max
            f.write(f"Speed filter: {min_label} to {max_label} knots\n")
        if "mmsi" in combined_df.columns:
            f.write(f"Unique MMSI: {combined_df['mmsi'].nunique(dropna=True)}\n")
        f.write("\nInput files:\n")
        for csv_path in csv_paths:
            f.write(f"- {csv_path}\n")


def aggregate_and_plot(
    inputs: list[str],
    output_dir: str,
    pattern: str = DEFAULT_PATTERN,
    speed_min: float | None = None,
    speed_max: float | None = None,
) -> pd.DataFrame:
    """Aggregate acoustic feature CSVs, save the combined CSV, and create plots."""
    csv_paths = find_acoustic_feature_csvs(inputs, pattern)
    if not csv_paths:
        raise FileNotFoundError(
            f"No acoustic feature CSV files found. Pattern: {pattern}, inputs: {inputs}"
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(csv_paths)} acoustic feature CSV file(s).")
    combined_df = read_and_combine_acoustic_features(csv_paths)
    if combined_df.empty:
        raise ValueError("No valid acoustic feature rows were loaded.")

    loaded_rows = len(combined_df)
    combined_df = filter_by_speed_range(combined_df, speed_min, speed_max)
    if combined_df.empty:
        raise ValueError("No acoustic feature rows remain after speed filtering.")

    combined_csv_path = output_path / "combined_acoustic_features.csv"
    combined_df.to_csv(combined_csv_path, index=False)
    print(f"Combined CSV saved: {combined_csv_path}")

    save_summary(combined_df, csv_paths, output_path, loaded_rows, speed_min, speed_max)
    plot_feature_vs_vessel_params(combined_df, str(output_path))
    return combined_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine multiple acoustic_features.csv files and generate plots "
            "using the existing acoustic feature analysis plotting routine."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        nargs="+",
        required=True,
        help=(
            "Input acoustic_features.csv files or directories. Directories are "
            "searched recursively."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="combined_acoustic_features_analysis",
        help="Directory where the combined CSV, summary, and plots will be saved.",
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help="Filename pattern to search for when an input is a directory.",
    )
    parser.add_argument(
        "--speed-min",
        type=float,
        default=None,
        help="Minimum vessel_speed_knots value to include in the analysis.",
    )
    parser.add_argument(
        "--speed-max",
        type=float,
        default=None,
        help="Maximum vessel_speed_knots value to include in the analysis.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aggregate_and_plot(
        args.input,
        args.output_dir,
        args.pattern,
        speed_min=args.speed_min,
        speed_max=args.speed_max,
    )


if __name__ == "__main__":
    main()
