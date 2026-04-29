import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from visualization import plot_geolocation


def read_all_ais(ais_paths, low_memory=False):
    """
    Read and combine multiple AIS CSV files into a single normalized DataFrame.

    Args:
        ais_paths (list[str]): List of AIS CSV file paths
        low_memory (bool): If True, use memory-efficient processing (slower but uses less RAM)

    Returns:
        DataFrame: Combined AIS with normalized headers, deduplicated and sorted by (mmsi, dt_pos_utc)
    """
    if not low_memory:
        # Original fast method: load all into memory
        frames = []
        for p in ais_paths:
            try:
                df = read_ais(p)
                frames.append(df)
            except Exception:
                continue
        if not frames:
            return pd.DataFrame()
        combined = pd.concat(frames, ignore_index=True)
        if "dt_pos_utc" in combined.columns:
            combined["dt_pos_utc"] = pd.to_datetime(
                combined["dt_pos_utc"]
            )  # idempotent
        combined = combined.drop_duplicates(
            subset=[c for c in ["mmsi", "dt_pos_utc"] if c in combined.columns]
        )
        sort_keys = [k for k in ["mmsi", "dt_pos_utc"] if k in combined.columns]
        if sort_keys:
            combined = combined.sort_values(sort_keys).reset_index(drop=True)
        return combined
    else:
        # Memory-efficient method: process by MMSI groups
        print("Using low-memory mode for AIS data processing...")
        mmsi_data = {}  # {mmsi: list of dataframes}

        # First pass: collect data per MMSI
        for i, p in enumerate(ais_paths):
            try:
                print(f"  Reading file {i+1}/{len(ais_paths)}: {os.path.basename(p)}")
                df = read_ais(p)

                # Group by MMSI and store
                for mmsi in df["mmsi"].unique():
                    if mmsi not in mmsi_data:
                        mmsi_data[mmsi] = []
                    vessel_df = df[df["mmsi"] == mmsi].copy()
                    mmsi_data[mmsi].append(vessel_df)

                # Clear DataFrame from memory
                del df
            except Exception as e:
                print(f"  Warning: Could not read {p}: {e}")
                continue

        if not mmsi_data:
            return pd.DataFrame()

        # Second pass: combine per MMSI and concatenate
        print(f"  Combining data for {len(mmsi_data)} unique vessels...")
        combined_list = []
        for mmsi, frames in mmsi_data.items():
            vessel_combined = pd.concat(frames, ignore_index=True)
            if "dt_pos_utc" in vessel_combined.columns:
                vessel_combined["dt_pos_utc"] = pd.to_datetime(
                    vessel_combined["dt_pos_utc"]
                )
            vessel_combined = vessel_combined.drop_duplicates(
                subset=[
                    c for c in ["mmsi", "dt_pos_utc"] if c in vessel_combined.columns
                ]
            )
            vessel_combined = vessel_combined.sort_values("dt_pos_utc").reset_index(
                drop=True
            )
            combined_list.append(vessel_combined)

        # Final combination
        combined = pd.concat(combined_list, ignore_index=True)
        sort_keys = [k for k in ["mmsi", "dt_pos_utc"] if k in combined.columns]
        if sort_keys:
            combined = combined.sort_values(sort_keys).reset_index(drop=True)

        return combined


def read_ais(ais_data):
    """
    Reads AIS data from a CSV file and converts the datetime format.

    Args:
        ais_data (str): Path to the AIS data CSV file.

    Returns:
        DataFrame: AIS data with sorted MMSI and converted datetime column.
    """

    def _get_type_candidates() -> dict:
        """Return candidate column names per canonical field (absorb header variations)."""
        # Canonical fields we standardize to: mmsi, dt_pos_utc, latitude, longitude,
        # vessel_name, vessel_type, length, width, speed
        return {
            "mmsi": ["mmsi", "MMSI"],
            "dt_pos_utc": [
                "dt_pos_utc",
                "MESSAGE TIMESTAMP",
                "timeUpdated(utc)",
            ],
            "latitude": ["latitude", "lat", "LATITUDE"],
            "longitude": ["longitude", "lon", "LONGITUDE"],
            "vessel_name": ["vessel_name", "name", "shipname", "vesselName"],
            "vessel_type": ["vessel_type", "cargoType"],
            "length": ["length", "length_m", "LENGTH"],
            "width": ["width", "width_m", "beam", "BEAM", "breadth"],
            "speed": ["speed", "Speed", "SPEED", "SOG", "sog", "speedOverGround", "speed_over_ground"],
            # Specialized dimension headers (some AIS provide hull dims as A/B/C/D)
            # length = dimA + dimB, width = dimC + dimD
            # We keep component keys separate to avoid renaming original columns
            "length_dimA": ["dimA", "DimA", "DIMA", "dim a", "DIM A"],
            "length_dimB": ["dimB", "DimB", "DIMB", "dim b", "DIM B"],
            "width_dimC": ["dimC", "DimC", "DIMC", "dim c", "DIM C"],
            "width_dimD": ["dimD", "DimD", "DIMD", "dim d", "DIM D"],
        }

    def _find_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
        """Find a column in df matching any candidate (case-insensitive)."""
        lower_map = {c.lower(): c for c in df.columns}
        for cand in candidates:
            key = cand.lower()
            if key in lower_map:
                return lower_map[key]
        return None

    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        candidates = _get_type_candidates()
        rename_map = {}
        # Do not rename dim component keys; only rename canonical targets
        skip_keys = {"length_dimA", "length_dimB", "width_dimC", "width_dimD"}
        for canonical, cand_list in candidates.items():
            if canonical in skip_keys:
                continue
            found = _find_existing_column(df, cand_list)
            if found is not None:
                rename_map[found] = canonical
        if rename_map:
            df = df.rename(columns=rename_map)

        # Compose length/width from dimA/B/C/D if necessary
        def _find_one(df_local: pd.DataFrame, name_candidates: list[str]) -> str | None:
            lower_map = {c.lower(): c for c in df_local.columns}
            for n in name_candidates:
                key = n.lower()
                if key in lower_map:
                    return lower_map[key]
            return None

        if "length" not in df.columns:
            a_col = _find_one(df, candidates.get("length_dimA", []))
            b_col = _find_one(df, candidates.get("length_dimB", []))
            if a_col is not None and b_col is not None:
                df["length"] = pd.to_numeric(
                    df[a_col], errors="coerce"
                ) + pd.to_numeric(df[b_col], errors="coerce")

        if "width" not in df.columns:
            c_col = _find_one(df, candidates.get("width_dimC", []))
            d_col = _find_one(df, candidates.get("width_dimD", []))
            if c_col is not None and d_col is not None:
                df["width"] = pd.to_numeric(df[c_col], errors="coerce") + pd.to_numeric(
                    df[d_col], errors="coerce"
                )
        return df

    ais_df = pd.read_csv(ais_data)
    # Normalize headers
    ais_df = _normalize_columns(ais_df)

    # Convert datetime if present
    if "dt_pos_utc" in ais_df.columns:
        ais_df["dt_pos_utc"] = pd.to_datetime(ais_df["dt_pos_utc"])

    # Sort if keys exist
    sort_keys = [k for k in ["mmsi", "dt_pos_utc"] if k in ais_df.columns]
    if sort_keys:
        ais_df.sort_values(sort_keys, inplace=True)

    return ais_df


def complement_trajectory(
    data, record_pos=None, output_dir=None, plot_before_after: bool = False, min_distance_info=None
):
    """
    Complements vessel trajectories by resampling and interpolating data to fill in missing points.

    Args:
        data (str or DataFrame): Path to the CSV file or DataFrame containing vessel data.
        record_pos (tuple): Recording position (latitude, longitude)
        output_dir (str): Output directory for plots
        plot_before_after (bool): Whether to plot before/after trajectories
        min_distance_info (dict): Dictionary of {mmsi: {'min_distance_pos': (lat, lon), 'min_distance [m]': float}}

    Returns:
        DataFrame: Resampled and interpolated vessel data.
    """
    # AISの読み込みは共通関数に統一（DataFrameも受け付け）
    if isinstance(data, pd.DataFrame):
        data = data.copy()
    else:
        data = read_ais(data)
    # 補完前の軌跡を表示（オプション）
    if plot_before_after and record_pos is not None and output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        plot_geolocation("traj_before", data, record_pos, output_dir)
    data.drop_duplicates(subset=["mmsi", "dt_pos_utc"], inplace=True)
    resampled_data_list = []

    # Process each vessel group
    total_vessels = data["mmsi"].nunique()
    for i, (mmsi, group) in enumerate(data.groupby("mmsi")):
        if (i + 1) % 100 == 0:  # Progress indicator for large datasets
            print(f"  Processing vessel {i+1}/{total_vessels}...")

        group = group.copy()  # Avoid SettingWithCopyWarning
        group.set_index("dt_pos_utc", inplace=True)
        # Infer object types before resampling to avoid FutureWarning
        group = group.infer_objects(copy=False)
        resampled = group.resample("1s").asfreq()
        # Infer object types again after resampling
        resampled = resampled.infer_objects(copy=False)
        group_resampled = resampled.interpolate().ffill()
        resampled_data_list.append(group_resampled)

    resampled_data = pd.concat(resampled_data_list).reset_index()
    resampled_data["depth"] = 0
    # 補完後の軌跡を表示（オプション）
    if plot_before_after and record_pos is not None and output_dir is not None:
        plot_geolocation("traj_after", resampled_data, record_pos, output_dir)

        # 各船舶ごとに補完前後を重ねた図を生成
        per_vessel_dir = os.path.join(output_dir, "traj_per_vessel")
        os.makedirs(per_vessel_dir, exist_ok=True)

        if data.empty:
            before_mmsi = []
        else:
            before_clean = data.dropna(subset=["latitude", "longitude"])
            before_clean = before_clean[
                np.isfinite(before_clean["latitude"])
                & np.isfinite(before_clean["longitude"])
            ]
            before_mmsi = before_clean["mmsi"].unique()

        after_clean = resampled_data.dropna(subset=["latitude", "longitude"])
        after_clean = after_clean[
            np.isfinite(after_clean["latitude"]) & np.isfinite(after_clean["longitude"])
        ]
        after_mmsi = after_clean["mmsi"].unique() if not after_clean.empty else []

        all_mmsi = set(before_mmsi).union(set(after_mmsi))

        for mmsi in all_mmsi:
            vb = (
                before_clean[before_clean["mmsi"] == mmsi]
                if len(before_mmsi)
                else pd.DataFrame()
            )
            va = (
                after_clean[after_clean["mmsi"] == mmsi]
                if len(after_mmsi)
                else pd.DataFrame()
            )

            fig, ax = plt.subplots()
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            # rec_pos
            ax.scatter(record_pos[1], record_pos[0], c="blue", marker="*", s=100)
            ax.text(
                record_pos[1],
                record_pos[0],
                "rec_pos",
                fontsize=9,
                ha="right",
                va="bottom",
            )

            # before (gray, alpha=0.2)
            if not vb.empty:
                vb_sorted = vb.sort_values("dt_pos_utc")
                ax.plot(
                    vb_sorted["longitude"],
                    vb_sorted["latitude"],
                    color=(0.6, 0.6, 0.6),
                    alpha=0.8,
                    linestyle="-",
                    label="before",
                )
                ax.scatter(
                    vb_sorted["longitude"],
                    vb_sorted["latitude"],
                    color=(0.6, 0.6, 0.6),
                    alpha=0.8,
                    s=12,
                )

            # after (colored)
            if not va.empty:
                va_sorted = va.sort_values("dt_pos_utc")
                ax.plot(
                    va_sorted["longitude"],
                    va_sorted["latitude"],
                    color="#1f77b4",
                    alpha=0.9,
                    linestyle="-",
                    label="after",
                )
                ax.scatter(
                    va_sorted["longitude"],
                    va_sorted["latitude"],
                    color="#1f77b4",
                    alpha=1.0,
                    s=16,
                )

            # Plot minimum distance point if available
            if min_distance_info is not None and mmsi in min_distance_info:
                min_info = min_distance_info[mmsi]
                min_pos = min_info.get("min_distance_pos")
                min_dist = min_info.get("min_distance [m]")
                if min_pos is not None:
                    # min_pos is (lat, lon)
                    ax.scatter(
                        min_pos[1], min_pos[0], 
                        c="red", 
                        marker="X", 
                        s=150, 
                        label=f"min dist ({min_dist:.1f}m)",
                        edgecolors="black",
                        linewidths=1.5,
                        zorder=10
                    )
                    # Draw line from rec_pos to min_distance_pos
                    ax.plot(
                        [record_pos[1], min_pos[1]],
                        [record_pos[0], min_pos[0]],
                        color="red",
                        linestyle="--",
                        alpha=0.6,
                        linewidth=1.5,
                        zorder=5
                    )

            ax.grid(True, linestyle="--", alpha=0.6)
            ax.set_title(f"MMSI {mmsi} trajectory (before vs after)")
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc="best", fontsize=8)
            out_path = os.path.join(per_vessel_dir, f"traj_mmsi_{mmsi}.png")
            plt.savefig(out_path, bbox_inches="tight", dpi=150)
            plt.close(fig)
    return resampled_data
