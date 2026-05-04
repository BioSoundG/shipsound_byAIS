"""Fetch vessel tonnage by MMSI and plot a log-scale frequency distribution."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VESSELFINDER_VESSELS_URL = "https://api.vesselfinder.com/vessels"
DEFAULT_CACHE_NAME = "vesselfinder_vessel_tonnage_cache.json"


def find_ais_csvs(inputs: list[str]) -> list[Path]:
    """Find AIS CSV files from file and directory inputs."""
    paths: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_file() and path.suffix.lower() == ".csv":
            paths.append(path)
        elif path.is_dir():
            paths.extend(path.rglob("*.csv"))
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


def collect_mmsis(csv_paths: list[Path]) -> list[int]:
    """Collect unique MMSI values from AIS CSV files."""
    mmsis: set[int] = set()
    for csv_path in csv_paths:
        try:
            df = pd.read_csv(csv_path, usecols=lambda col: col.lower() == "mmsi")
        except ValueError:
            print(f"Warning: MMSI column not found: {csv_path}")
            continue
        except Exception as exc:
            print(f"Warning: failed to read {csv_path}: {exc}")
            continue

        mmsi_col = find_column_case_insensitive(df, ["mmsi"])
        if mmsi_col is None:
            continue

        numeric_mmsi = pd.to_numeric(df[mmsi_col], errors="coerce").dropna()
        mmsis.update(int(value) for value in numeric_mmsi.astype("int64"))
    return sorted(mmsis)


def load_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
    """Load cached VesselFinder records keyed by MMSI."""
    if not cache_path.exists():
        return {}
    try:
        with cache_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"Warning: failed to load cache {cache_path}: {exc}")
        return {}
    return data if isinstance(data, dict) else {}


def save_cache(cache: dict[str, dict[str, Any]], cache_path: Path) -> None:
    """Save VesselFinder records keyed by MMSI."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2, sort_keys=True)


def chunked(values: list[int], size: int) -> list[list[int]]:
    """Split a list into chunks."""
    return [values[i : i + size] for i in range(0, len(values), size)]


def fetch_vesselfinder_records(
    mmsis: list[int],
    userkey: str,
    cache: dict[str, dict[str, Any]],
    batch_size: int = 50,
    request_interval: float = 0.2,
) -> dict[str, dict[str, Any]]:
    """Fetch VesselFinder AIS and Master data for MMSIs, reusing cached records."""
    missing = [mmsi for mmsi in mmsis if str(mmsi) not in cache]
    if not missing:
        return cache

    for batch in chunked(missing, batch_size):
        params = {
            "userkey": userkey,
            "format": "json",
            "extradata": "master",
            "mmsi": ",".join(str(mmsi) for mmsi in batch),
        }
        url = f"{VESSELFINDER_VESSELS_URL}?{urllib.parse.urlencode(params)}"
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            print(f"Warning: VesselFinder request failed for MMSI batch {batch}: {exc}")
            continue

        if isinstance(payload, dict) and "error" in payload:
            print(f"Warning: VesselFinder API error for batch {batch}: {payload['error']}")
            continue
        if not isinstance(payload, list):
            print(f"Warning: unexpected VesselFinder response for batch {batch}: {payload}")
            continue

        for item in payload:
            if not isinstance(item, dict):
                continue
            ais = item.get("AIS") or {}
            master = item.get("MASTER") or item.get("Master") or {}
            mmsi = ais.get("MMSI") or master.get("MMSI")
            if mmsi is None:
                continue
            cache[str(int(mmsi))] = {"AIS": ais, "MASTER": master}

        time.sleep(request_interval)

    return cache


def first_numeric(mapping: dict[str, Any], keys: list[str]) -> float | None:
    """Return the first finite numeric value found in a mapping."""
    lower_map = {str(key).lower(): value for key, value in mapping.items()}
    for key in keys:
        value = lower_map.get(key.lower())
        if value is None:
            continue
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.notna(numeric) and float(numeric) > 0:
            return float(numeric)
    return None


def estimate_tonnage(record: dict[str, Any]) -> tuple[float | None, str]:
    """Estimate vessel tonnage from VesselFinder Master/AIS fields."""
    master = record.get("MASTER") or {}
    ais = record.get("AIS") or {}

    dwt = first_numeric(master, ["DWT", "DEADWEIGHT", "deadweight"])
    if dwt is not None:
        return dwt, "DWT"

    gt = first_numeric(master, ["GT", "GROSS_TONNAGE", "gross_tonnage", "GROSS TONNAGE"])
    if gt is not None:
        return gt, "GT"

    nt = first_numeric(master, ["NT", "NET_TONNAGE", "net_tonnage", "NET TONNAGE"])
    if nt is not None:
        return nt, "NT"

    length = first_numeric(master, ["LENGTH", "length"]) or (
        first_numeric(ais, ["A"]) or 0
    ) + (first_numeric(ais, ["B"]) or 0)
    beam = first_numeric(master, ["BEAM", "WIDTH", "width"]) or (
        first_numeric(ais, ["C"]) or 0
    ) + (first_numeric(ais, ["D"]) or 0)
    draught = first_numeric(master, ["DRAUGHT", "MAX_DRAUGHT", "max_draught"]) or first_numeric(
        ais, ["DRAUGHT"]
    )
    if length and beam and draught:
        estimated_displacement = length * beam * draught * 0.7
        return estimated_displacement, "estimated_displacement"

    return None, "missing"


def build_tonnage_dataframe(
    mmsis: list[int],
    records: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Build a vessel tonnage table from cached VesselFinder records."""
    rows = []
    for mmsi in mmsis:
        record = records.get(str(mmsi), {})
        ais = record.get("AIS") or {}
        master = record.get("MASTER") or {}
        tonnage, tonnage_source = estimate_tonnage(record) if record else (None, "missing")
        rows.append(
            {
                "mmsi": mmsi,
                "vessel_name": ais.get("NAME") or master.get("NAME"),
                "imo": ais.get("IMO") or master.get("IMO"),
                "estimated_tonnage": tonnage,
                "tonnage_source": tonnage_source,
                "gross_tonnage": first_numeric(master, ["GT", "GROSS_TONNAGE", "gross_tonnage"]),
                "net_tonnage": first_numeric(master, ["NT", "NET_TONNAGE", "net_tonnage"]),
                "deadweight_tonnage": first_numeric(master, ["DWT", "DEADWEIGHT", "deadweight"]),
            }
        )
    return pd.DataFrame(rows)


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


def save_summary(tonnage_df: pd.DataFrame, output_dir: Path, min_tonnage: float) -> None:
    """Save numeric counts including the excluded 0-min_tonnage group."""
    valid = tonnage_df.dropna(subset=["estimated_tonnage"])
    under_min_count = int((valid["estimated_tonnage"] < min_tonnage).sum())
    plotted_count = int((valid["estimated_tonnage"] >= min_tonnage).sum())
    missing_count = int(tonnage_df["estimated_tonnage"].isna().sum())

    summary_path = output_dir / "vessel_tonnage_distribution_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Vessel tonnage distribution summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Total MMSI: {len(tonnage_df)}\n")
        f.write(f"Plotted vessels (>= {min_tonnage:g} tons): {plotted_count}\n")
        f.write(f"Excluded vessels (0 - {min_tonnage:g} tons): {under_min_count}\n")
        f.write(f"Missing tonnage: {missing_count}\n")
        f.write("\nTonnage source counts:\n")
        for source, count in tonnage_df["tonnage_source"].value_counts(dropna=False).items():
            f.write(f"- {source}: {count}\n")


def run_analysis(
    inputs: list[str],
    output_dir: str,
    userkey: str,
    min_tonnage: float = 300.0,
    bins: int = 25,
    cache_json: str | None = None,
    batch_size: int = 50,
    request_interval: float = 0.2,
) -> pd.DataFrame:
    """Collect MMSIs, fetch VesselFinder data, and save tonnage outputs."""
    csv_paths = find_ais_csvs(inputs)
    if not csv_paths:
        raise FileNotFoundError("No AIS CSV files found.")

    mmsis = collect_mmsis(csv_paths)
    if not mmsis:
        raise ValueError("No MMSI values found in AIS CSV files.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    cache_path = Path(cache_json) if cache_json else output_path / DEFAULT_CACHE_NAME
    cache = load_cache(cache_path)

    print(f"Found {len(mmsis)} unique MMSI values.")
    records = fetch_vesselfinder_records(
        mmsis,
        userkey,
        cache,
        batch_size=batch_size,
        request_interval=request_interval,
    )
    save_cache(records, cache_path)

    tonnage_df = build_tonnage_dataframe(mmsis, records)
    table_path = output_path / "vessel_tonnage_estimates.csv"
    tonnage_df.to_csv(table_path, index=False)

    plot_path = plot_tonnage_distribution(tonnage_df, output_path, min_tonnage, bins)
    save_summary(tonnage_df, output_path, min_tonnage)

    print(f"Tonnage table saved: {table_path}")
    print(f"Distribution plot saved: {plot_path}")
    return tonnage_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch vessel tonnage from VesselFinder by MMSI and plot a log-scale "
            "frequency distribution."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        nargs="+",
        required=True,
        help="AIS CSV files or directories containing AIS CSV files.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="vessel_tonnage_distribution",
        help="Directory for CSV, summary, cache, and plot outputs.",
    )
    parser.add_argument(
        "--userkey",
        default=os.environ.get("VESSELFINDER_API_KEY"),
        help="VesselFinder API key. Defaults to VESSELFINDER_API_KEY environment variable.",
    )
    parser.add_argument(
        "--min-tonnage",
        type=float,
        default=300.0,
        help="Minimum tonnage shown on the plot. Lower values are counted separately.",
    )
    parser.add_argument("--bins", type=int, default=25, help="Number of histogram bins.")
    parser.add_argument(
        "--cache-json",
        default=None,
        help="Path to a JSON cache for VesselFinder responses.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of MMSIs per VesselFinder API request.",
    )
    parser.add_argument(
        "--request-interval",
        type=float,
        default=0.2,
        help="Seconds to wait between VesselFinder API requests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.userkey:
        raise ValueError(
            "VesselFinder API key is required. Pass --userkey or set VESSELFINDER_API_KEY."
        )
    run_analysis(
        args.input,
        args.output_dir,
        args.userkey,
        min_tonnage=args.min_tonnage,
        bins=args.bins,
        cache_json=args.cache_json,
        batch_size=args.batch_size,
        request_interval=args.request_interval,
    )


if __name__ == "__main__":
    main()
