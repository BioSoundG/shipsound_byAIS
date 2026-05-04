"""Estimate vessel tonnage from AIS CSV static fields and plot a distribution."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DIMENSION_COEFFICIENT = 0.7
MMSI_PATTERN = re.compile(r"\bmmsi\s*:\s*(\d+)", re.IGNORECASE)


def find_files(inputs: list[str], suffix: str) -> list[Path]:
    """Find files with suffix from multiple file or directory inputs."""
    paths: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_file() and path.suffix.lower() == suffix:
            paths.append(path)
        elif path.is_dir():
            paths.extend(path.rglob(f"*{suffix}"))
        else:
            print(f"Warning: input path skipped: {path}")
    return sorted({p.resolve() for p in paths})


def find_column_case_insensitive(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first matching column name, ignoring case."""
    lower_map = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        found = lower_map.get(candidate.lower())
        if found is not None:
            return found
    return None


def read_toml_text(toml_path: Path) -> str:
    """Read a TOML file as text, tolerating common encodings."""
    for encoding in ("utf-8", "utf-8-sig", "cp932"):
        try:
            return toml_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return toml_path.read_text(errors="ignore")


def collect_mmsis_from_toml(toml_paths: list[Path]) -> set[int]:
    """Collect target MMSIs from cut metadata TOML files."""
    mmsis: set[int] = set()
    for toml_path in toml_paths:
        text = read_toml_text(toml_path)
        matches = MMSI_PATTERN.findall(text)
        if matches:
            mmsis.update(int(match) for match in matches)
        else:
            print(f"Warning: MMSI not found in TOML: {toml_path}")
    return mmsis


def first_existing_numeric(df: pd.DataFrame, candidates: list[str]) -> pd.Series | None:
    """Return the first candidate column converted to numeric."""
    column = find_column_case_insensitive(df, candidates)
    if column is None:
        return None
    return pd.to_numeric(df[column], errors="coerce")


def sum_numeric_columns(df: pd.DataFrame, candidates_a: list[str], candidates_b: list[str]) -> pd.Series | None:
    """Return numeric sum of two candidate columns when both exist."""
    col_a = first_existing_numeric(df, candidates_a)
    col_b = first_existing_numeric(df, candidates_b)
    if col_a is None or col_b is None:
        return None
    return col_a + col_b


def normalize_ais_static_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize MMSI, dimensions, draught, and optional vessel metadata."""
    normalized = pd.DataFrame(index=df.index)

    mmsi = first_existing_numeric(df, ["mmsi", "MMSI"])
    if mmsi is None:
        return pd.DataFrame()
    normalized["mmsi"] = mmsi

    name_col = find_column_case_insensitive(df, ["vessel_name", "name", "shipname", "vesselName"])
    type_col = find_column_case_insensitive(df, ["vessel_type", "cargoType", "TYPE", "ship_type"])
    if name_col:
        normalized["vessel_name"] = df[name_col]
    if type_col:
        normalized["vessel_type"] = df[type_col]

    length = first_existing_numeric(df, ["length", "length_m", "LENGTH"])
    if length is None:
        length = sum_numeric_columns(
            df,
            ["dimA", "DimA", "DIMA", "dim a", "DIM A", "A"],
            ["dimB", "DimB", "DIMB", "dim b", "DIM B", "B"],
        )

    width = first_existing_numeric(df, ["width", "width_m", "beam", "BEAM", "breadth"])
    if width is None:
        width = sum_numeric_columns(
            df,
            ["dimC", "DimC", "DIMC", "dim c", "DIM C", "C"],
            ["dimD", "DimD", "DIMD", "dim d", "DIM D", "D"],
        )

    draught = first_existing_numeric(
        df,
        ["draught", "draft", "DRAUGHT", "DRAFT", "max_draught", "current_draught"],
    )

    normalized["length_m"] = length if length is not None else np.nan
    normalized["width_m"] = width if width is not None else np.nan
    normalized["draught_m"] = draught if draught is not None else np.nan
    return normalized


def read_static_vessel_rows(csv_paths: list[Path], target_mmsis: set[int]) -> pd.DataFrame:
    """Read static vessel fields from AIS CSVs for target MMSIs."""
    frames = []
    for csv_path in csv_paths:
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"Warning: failed to read {csv_path}: {exc}")
            continue

        normalized = normalize_ais_static_fields(df)
        if normalized.empty:
            print(f"Warning: MMSI column not found: {csv_path}")
            continue
        normalized["source_file"] = str(csv_path)

        normalized = normalized.dropna(subset=["mmsi"]).copy()
        normalized["mmsi"] = normalized["mmsi"].astype("int64")
        normalized = normalized[normalized["mmsi"].isin(target_mmsis)]
        if not normalized.empty:
            frames.append(normalized)

    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined


def estimate_tonnage_from_ais(row: pd.Series, coefficient: float) -> tuple[float | None, str]:
    """Estimate tonnage from AIS length, width, and draught."""
    length = row.get("length_m")
    width = row.get("width_m")
    draught = row.get("draught_m")

    if pd.notna(length) and pd.notna(width) and pd.notna(draught):
        if length > 0 and width > 0 and draught > 0:
            return float(length * width * draught * coefficient), "length_width_draught"

    if pd.notna(length) and pd.notna(width):
        if length > 0 and width > 0:
            return float(length * width * coefficient), "length_width_area"

    return None, "missing_dimensions"


def build_tonnage_dataframe(static_rows: pd.DataFrame, coefficient: float) -> pd.DataFrame:
    """Collapse AIS rows to one estimated tonnage row per MMSI."""
    rows = []
    for mmsi, group in static_rows.groupby("mmsi", sort=True):
        best = group.copy()
        best["dimension_score"] = best[["length_m", "width_m", "draught_m"]].notna().sum(axis=1)
        best = best.sort_values("dimension_score", ascending=False).iloc[0]
        tonnage, source = estimate_tonnage_from_ais(best, coefficient)
        rows.append(
            {
                "mmsi": int(mmsi),
                "vessel_name": best.get("vessel_name"),
                "vessel_type": best.get("vessel_type"),
                "estimated_tonnage": tonnage,
                "tonnage_source": source,
                "length_m": best.get("length_m"),
                "width_m": best.get("width_m"),
                "draught_m": best.get("draught_m"),
                "source_file": best.get("source_file"),
            }
        )
    return pd.DataFrame(rows)


def append_missing_target_mmsis(tonnage_df: pd.DataFrame, target_mmsis: set[int]) -> pd.DataFrame:
    """Add missing target MMSIs with empty tonnage rows."""
    existing = set(tonnage_df["mmsi"].dropna().astype("int64")) if not tonnage_df.empty else set()
    missing = sorted(target_mmsis - existing)
    if not missing:
        return tonnage_df

    missing_df = pd.DataFrame(
        {
            "mmsi": missing,
            "vessel_name": np.nan,
            "vessel_type": np.nan,
            "estimated_tonnage": np.nan,
            "tonnage_source": "missing_ais_static_info",
            "length_m": np.nan,
            "width_m": np.nan,
            "draught_m": np.nan,
            "source_file": np.nan,
        }
    )
    return pd.concat([tonnage_df, missing_df], ignore_index=True, sort=False)


def plot_tonnage_distribution(
    tonnage_df: pd.DataFrame,
    output_dir: Path,
    min_tonnage: float = 300.0,
    bins: int = 25,
) -> Path:
    """Plot frequency distribution of vessel tonnage on a log x-axis."""
    valid = tonnage_df.dropna(subset=["estimated_tonnage"]).copy()
    plot_data = valid[valid["estimated_tonnage"] >= min_tonnage]
    under_min_count = int((valid["estimated_tonnage"] < min_tonnage).sum())
    missing_count = int(tonnage_df["estimated_tonnage"].isna().sum())

    if plot_data.empty:
        raise ValueError(f"No vessels have estimated tonnage >= {min_tonnage:g}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    max_tonnage = plot_data["estimated_tonnage"].max()
    if max_tonnage == min_tonnage:
        max_tonnage = min_tonnage * 1.1
    bin_edges = np.logspace(np.log10(min_tonnage), np.log10(max_tonnage), bins + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(plot_data["estimated_tonnage"], bins=bin_edges, edgecolor="black", alpha=0.75)
    ax.set_xscale("log")
    ax.set_xlim(left=min_tonnage)
    ax.set_xlabel("Estimated tonnage (tons, log scale)")
    ax.set_ylabel("Number of vessels")
    ax.set_title("Vessel Tonnage Frequency Distribution")
    ax.grid(True, which="both", alpha=0.3)

    summary_text = (
        f"Excluded < {min_tonnage:g} tons: {under_min_count}\n"
        f"Missing tonnage: {missing_count}\n"
        f"Plotted vessels: {len(plot_data)}"
    )
    ax.text(
        0.02,
        0.98,
        summary_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "0.5", "alpha": 0.9},
    )
    fig.tight_layout()

    plot_path = output_dir / "vessel_tonnage_distribution.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


def save_summary(tonnage_df: pd.DataFrame, output_dir: Path, min_tonnage: float, csv_count: int) -> None:
    """Save numeric counts including the excluded 0-min_tonnage group."""
    valid = tonnage_df.dropna(subset=["estimated_tonnage"])
    under_min_count = int((valid["estimated_tonnage"] < min_tonnage).sum())
    plotted_count = int((valid["estimated_tonnage"] >= min_tonnage).sum())
    missing_count = int(tonnage_df["estimated_tonnage"].isna().sum())

    summary_path = output_dir / "vessel_tonnage_distribution_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Vessel tonnage distribution summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Input AIS CSV files: {csv_count}\n")
        f.write(f"Total MMSI: {len(tonnage_df)}\n")
        f.write(f"Plotted vessels (>= {min_tonnage:g} tons): {plotted_count}\n")
        f.write(f"Excluded vessels (0 - {min_tonnage:g} tons): {under_min_count}\n")
        f.write(f"Missing tonnage: {missing_count}\n")
        f.write("\nTonnage source counts:\n")
        for source, count in tonnage_df["tonnage_source"].value_counts(dropna=False).items():
            f.write(f"- {source}: {count}\n")


def run_analysis(
    ais_inputs: list[str],
    toml_inputs: list[str],
    output_dir: str,
    min_tonnage: float = 300.0,
    bins: int = 25,
    coefficient: float = DIMENSION_COEFFICIENT,
) -> pd.DataFrame:
    """Read cut TOMLs and AIS CSVs, estimate target vessel tonnage, and save outputs."""
    csv_paths = find_files(ais_inputs, ".csv")
    if not csv_paths:
        raise FileNotFoundError("No AIS CSV files found.")

    toml_paths = find_files(toml_inputs, ".toml")
    if not toml_paths:
        raise FileNotFoundError("No cut metadata TOML files found.")

    target_mmsis = collect_mmsis_from_toml(toml_paths)
    if not target_mmsis:
        raise ValueError("No target MMSI values were found in cut metadata TOML files.")

    static_rows = read_static_vessel_rows(csv_paths, target_mmsis)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if static_rows.empty:
        tonnage_df = pd.DataFrame()
    else:
        tonnage_df = build_tonnage_dataframe(static_rows, coefficient)
    tonnage_df = append_missing_target_mmsis(tonnage_df, target_mmsis)

    table_path = output_path / "vessel_tonnage_estimates.csv"
    tonnage_df.to_csv(table_path, index=False)

    plot_path = plot_tonnage_distribution(tonnage_df, output_path, min_tonnage, bins)
    save_summary(tonnage_df, output_path, min_tonnage, len(csv_paths))

    print(f"Read AIS CSV files: {len(csv_paths)}")
    print(f"Read cut TOML files: {len(toml_paths)}")
    print(f"Target MMSI count: {len(target_mmsis)}")
    print(f"Tonnage table saved: {table_path}")
    print(f"Distribution plot saved: {plot_path}")
    return tonnage_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate vessel tonnage from AIS static fields and plot a log-scale "
            "frequency distribution."
        )
    )
    parser.add_argument(
        "-a",
        "--ais-input",
        nargs="+",
        required=True,
        help=(
            "AIS CSV files or directories. You can specify multiple directories; "
            "all CSV files inside each directory are read recursively."
        ),
    )
    parser.add_argument(
        "-t",
        "--toml-input",
        nargs="+",
        required=True,
        help=(
            "Directories or files containing cut metadata TOML files. You can specify "
            "multiple directories; all TOML files inside each directory are read recursively."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="vessel_tonnage_distribution",
        help="Directory for CSV, summary, and plot outputs.",
    )
    parser.add_argument(
        "--min-tonnage",
        type=float,
        default=300.0,
        help="Minimum tonnage shown on the plot. Lower values are counted separately.",
    )
    parser.add_argument("--bins", type=int, default=25, help="Number of histogram bins.")
    parser.add_argument(
        "--coefficient",
        type=float,
        default=DIMENSION_COEFFICIENT,
        help=(
            "Coefficient used for AIS dimension-based estimation. "
            "With draught: length * width * draught * coefficient. "
            "Without draught: length * width * coefficient."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_analysis(
        args.ais_input,
        args.toml_input,
        args.output_dir,
        min_tonnage=args.min_tonnage,
        bins=args.bins,
        coefficient=args.coefficient,
    )


if __name__ == "__main__":
    main()
