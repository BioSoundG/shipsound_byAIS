import os
import argparse
from natsort import natsorted
import glob
import tomli
import pandas as pd
import warnings

# Suppress FutureWarning messages
warnings.filterwarnings("ignore", category=FutureWarning)

# Use tomllib if available (Python 3.11+), otherwise keep tomli
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # Fallback to tomli if tomllib not found

from data_processing import read_ais, read_all_ais, complement_trajectory
from distance_calculation import (
    calculate_shortest_distance,
    calculate_distance_timeseries,
    haversine,
)
from visualization import plot_geolocation, plot_mother_source_spectrogram
from audio_processing import cut_wav_and_make_metadata
from acoustic_features import (
    analyze_vessel_acoustic_features,
    plot_feature_vs_vessel_params,
)


def read_toml_file(file_path):
    try:
        with open(file_path, "rb") as file:  # Binary mode required for tomllib/tomli
            data = tomllib.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: Toml file not found at {file_path}")
        return None
    except tomllib.TOMLDecodeError as e:
        print(f"Error decoding Toml file {file_path}: {e}")
        return None


# Function to load the main configuration
def load_config(config_path="config.toml"):
    print(f"Loading configuration from: {config_path}")
    config = read_toml_file(config_path)
    if config is None:
        print("Error: Configuration file could not be loaded. Exiting.")
        exit(1)  # Exit if config is essential
    # You might add validation or default values here if needed
    print("Configuration loaded successfully.")
    return config


def main(
    ais_path,
    wav_path,
    toml_path,
    record_start_time,
    flag_fig,
    flag_movie,
    flag_csv,
    config_path="config.toml",  # default for tests and CLI
):
    # Load configuration first
    config = load_config(config_path)
    if config is None:
        print("Exiting due to configuration loading failure.")
        return
    print("Configuration loaded successfully.")

    # Extract necessary sub-configs
    vis_config = config.get("visualization", {})
    audio_config = config.get("audio_processing", {})
    data_proc_config = config.get("data_processing", {})
    combine_all_ais = data_proc_config.get("combine_all_ais", False)
    low_memory_mode = data_proc_config.get("low_memory_mode", False)

    # Add cut_margin_minutes from audio_config to vis_config for plot_mother_source_spectrogram
    vis_config["cut_margin_minutes"] = audio_config.get(
        "cut_margin_minutes", 1
    )  # Default to 1 if not found

    ais_list = natsorted(glob.glob(f"{ais_path}/*.csv"))
    wav_list = natsorted(glob.glob(f"{wav_path}/*.WAV"))
    meta_data = read_toml_file(toml_path)
    if meta_data is None:
        print("Error: Metadata could not be loaded. Exiting.")
        return

    # Use the provided record_start_time parameter instead of metadata.toml
    start_tim = pd.to_datetime(record_start_time)

    # Position is now a list, so we can use it directly without eval()
    record_pos = meta_data["observation_info"]["location_info"]["position"]

    record_depth = meta_data["observation_info"]["location_info"]["installation_depth"]

    # Collect all distance dataframes to process spectrograms once per mother source
    all_distances_dfs = []

    # Statistics for final summary
    total_vessels = 0
    cut_target_vessels = 0
    actual_cut_count = 0

    if combine_all_ais:
        print("Combining all AIS CSVs into a single dataset...")
        ais_df_all = read_all_ais(ais_list, low_memory=low_memory_mode)
        combined_output_dir = os.path.join(os.path.dirname(ais_list[0]), "combined")
        os.makedirs(combined_output_dir, exist_ok=True)

        # First pass: complement trajectory without plotting
        comp_df_all = complement_trajectory(
            ais_df_all,
            record_pos=record_pos,
            output_dir=combined_output_dir,
            plot_before_after=False,  # Disable plotting in first pass
        )

        distance_list_df_all = calculate_distance_timeseries(
            comp_df_all, record_pos, record_depth
        )
        distances_all = calculate_shortest_distance(
            comp_df_all, record_pos, record_depth
        )
        distances_df_all = pd.DataFrame(distances_all)

        # Update total vessels count
        total_vessels = len(distances_df_all)

        # Create min_distance_info dictionary for plotting
        min_distance_info = {}
        for d in distances_all:
            min_distance_info[d["mmsi"]] = {
                "min_distance_pos": d["min_distance_pos"],
                "min_distance [m]": d["min_distance [m]"],
            }

        # Second pass: plot with min distance info if requested
        if flag_fig:
            comp_df_all = complement_trajectory(
                ais_df_all,
                record_pos=record_pos,
                output_dir=combined_output_dir,
                plot_before_after=True,
                min_distance_info=min_distance_info,
            )

        # Determine target MMSIs for plotting (colored) based on cut criteria
        target_mmsis = set()
        if not distances_df_all.empty:
            max_cut_distance = audio_config.get("max_cut_distance", float("inf"))
            check_other_vessels = audio_config.get("check_other_vessels", False)
            for _, row in distances_df_all.iterrows():
                if row["min_distance [m]"] > max_cut_distance:
                    continue
                mmsi = row["mmsi"]
                if check_other_vessels and not distance_list_df_all.empty:
                    min_distance_time = row["min_distance_time"]
                    is_closest = True
                    for other_mmsi in distances_df_all["mmsi"].unique():
                        if other_mmsi == mmsi:
                            continue
                        other = distance_list_df_all[
                            distance_list_df_all["mmsi"] == other_mmsi
                        ]
                        if other.empty:
                            continue
                        tmp = other.copy()
                        tmp["time_diff"] = abs(tmp["dt_pos_utc"] - min_distance_time)
                        closest = tmp.loc[tmp["time_diff"].idxmin()]
                        if closest["distance [m]"] < row["min_distance [m]"]:
                            is_closest = False
                            break
                    if not is_closest:
                        continue
                target_mmsis.add(mmsi)

        # Update cut target vessels count
        cut_target_vessels = len(target_mmsis)

        if flag_fig:
            print("plot geolocation (combined).....")
            plot_geolocation(
                "combined", ais_df_all, record_pos, combined_output_dir, target_mmsis
            )

        # For spectrograms
        all_distances_dfs.append(distances_df_all)

        if flag_csv:
            distances_df_all.to_csv(
                os.path.join(combined_output_dir, "distances_all.csv"), index=False
            )

        actual_cut_count = cut_wav_and_make_metadata(
            wav_list,
            meta_data,
            start_tim,
            distances_df_all,
            distance_list_df_all,
            combined_output_dir,
            record_pos,
            audio_config,
        )
    else:
        # Process each AIS file (legacy per-file mode)
        for idx, ais_data in enumerate(ais_list):
            print(
                f"Processing AIS data: {idx+1}/{len(ais_list)}, {os.path.basename(ais_data)}"
            )
            output_dir = os.path.join(
                os.path.dirname(ais_data),
                os.path.splitext(os.path.basename(ais_data))[0],
            )
            os.makedirs(output_dir, exist_ok=True)

            ais_df = read_ais(ais_data)
            # First pass: complement trajectory without plotting
            comp_df = complement_trajectory(
                ais_data,
                record_pos=record_pos,
                output_dir=output_dir,
                plot_before_after=False,  # Disable plotting in first pass
            )
            # Compute per-time distances for other-vessel comparison, if needed by audio_processing
            distance_list_df = calculate_distance_timeseries(
                comp_df, record_pos, record_depth
            )
            distances = calculate_shortest_distance(comp_df, record_pos, record_depth)
            distances_df = pd.DataFrame(distances)

            # Update total vessels count
            total_vessels += len(distances_df)

            # Create min_distance_info dictionary for plotting
            min_distance_info = {}
            for d in distances:
                min_distance_info[d["mmsi"]] = {
                    "min_distance_pos": d["min_distance_pos"],
                    "min_distance [m]": d["min_distance [m]"],
                }

            # Second pass: plot with min distance info if requested
            if flag_fig:
                comp_df = complement_trajectory(
                    ais_data,
                    record_pos=record_pos,
                    output_dir=output_dir,
                    plot_before_after=True,
                    min_distance_info=min_distance_info,
                )

            # Determine target MMSIs for plotting (colored) based on cut criteria
            target_mmsis = set()
            if not distances_df.empty:
                max_cut_distance = audio_config.get("max_cut_distance", float("inf"))
                check_other_vessels = audio_config.get("check_other_vessels", False)
                for _, row in distances_df.iterrows():
                    if row["min_distance [m]"] > max_cut_distance:
                        continue
                    mmsi = row["mmsi"]
                    if check_other_vessels and not distance_list_df.empty:
                        min_distance_time = row["min_distance_time"]
                        is_closest = True
                        for other_mmsi in distances_df["mmsi"].unique():
                            if other_mmsi == mmsi:
                                continue
                            other = distance_list_df[
                                distance_list_df["mmsi"] == other_mmsi
                            ]
                            if other.empty:
                                continue
                            tmp = other.copy()
                            tmp["time_diff"] = abs(
                                tmp["dt_pos_utc"] - min_distance_time
                            )
                            closest = tmp.loc[tmp["time_diff"].idxmin()]
                            if closest["distance [m]"] < row["min_distance [m]"]:
                                is_closest = False
                                break
                        if not is_closest:
                            continue
                    target_mmsis.add(mmsi)

            # Update cut target vessels count
            cut_target_vessels += len(target_mmsis)

            if flag_fig:
                print("plot geolocation.....")
                plot_geolocation(idx + 1, ais_df, record_pos, output_dir, target_mmsis)

            # Add to collection for spectrograms
            all_distances_dfs.append(distances_df)

            if flag_csv:
                distances_df.to_csv(
                    os.path.join(output_dir, f"distances_{idx+1}.csv"), index=False
                )

            # Pass audio_config to audio_processing
            actual_cut_count += cut_wav_and_make_metadata(
                wav_list,
                meta_data,
                start_tim,
                distances_df,
                distance_list_df,
                output_dir,
                record_pos,
                audio_config,  # Pass audio config dictionary
            )

    # Create mother source spectrograms with cut indicators if flag_fig is True
    enable_spectrogram = vis_config.get("enable_spectrogram", True)
    if flag_fig and wav_list and enable_spectrogram:
        print(f"Generating time-averaged spectrograms for all mother source files...")
        # Create an overall output directory for spectrograms
        overall_output_dir = os.path.join(
            os.path.dirname(ais_list[0]), "all_spectrograms"
        )
        os.makedirs(overall_output_dir, exist_ok=True)

        # Call the spectrogram function with all distance DataFrames and the modified vis_config
        plot_mother_source_spectrogram(
            wav_path,
            all_distances_dfs,
            start_tim,
            overall_output_dir,
            vis_config,  # Pass the modified visualization config including cut_margin_minutes
        )

    # Acoustic feature analysis
    acoustic_config = config.get("acoustic_features", {})
    enable_feature_plots = acoustic_config.get("enable_feature_plots", False)
    analysis_time_window = acoustic_config.get("analysis_time_window", 10)

    if enable_feature_plots and len(all_distances_dfs) > 0 and wav_list:
        print(f"\n音響特徴量分析を開始...")

        # Combine all distances for feature analysis
        if combine_all_ais:
            # Use combined data
            features_output_dir = os.path.join(
                os.path.dirname(ais_list[0]), "combined", "acoustic_features"
            )
            os.makedirs(features_output_dir, exist_ok=True)

            # Analyze acoustic features
            from wav_index import WavFileIndex

            wav_index = WavFileIndex(wav_list, start_tim)

            features_df = analyze_vessel_acoustic_features(
                wav_list,
                wav_index,
                all_distances_dfs[0] if len(all_distances_dfs) > 0 else pd.DataFrame(),
                comp_df_all if "comp_df_all" in locals() else pd.DataFrame(),
                analysis_window=analysis_time_window,
                output_dir=features_output_dir,
            )

            # Plot feature vs vessel parameters
            if len(features_df) > 0:
                plot_feature_vs_vessel_params(features_df, features_output_dir)
        else:
            # Process each AIS file separately
            for i, ais_data in enumerate(ais_list):
                output_dir = os.path.join(os.path.dirname(ais_data), "output")
                features_output_dir = os.path.join(output_dir, "acoustic_features")
                os.makedirs(features_output_dir, exist_ok=True)

                if i < len(all_distances_dfs) and len(all_distances_dfs[i]) > 0:
                    from wav_index import WavFileIndex

                    wav_index = WavFileIndex(wav_list, start_tim)

                    # Get complemented data for this AIS file
                    comp_df_local = complement_trajectory(
                        ais_data, plot_before_after=False
                    )

                    features_df = analyze_vessel_acoustic_features(
                        wav_list,
                        wav_index,
                        all_distances_dfs[i],
                        comp_df_local,
                        analysis_window=analysis_time_window,
                        output_dir=features_output_dir,
                    )

                    if len(features_df) > 0:
                        plot_feature_vs_vessel_params(features_df, features_output_dir)

    # Print final statistics
    print("\n" + "=" * 60)
    print("処理完了統計:")
    print("=" * 60)
    print(f"全体の船舶数:        {total_vessels:>6} 隻")
    print(f"切り出し対象船舶数:  {cut_target_vessels:>6} 隻")
    print(f"実際の切り出し数:    {actual_cut_count:>6} 個")
    if total_vessels > 0:
        percentage = (cut_target_vessels / total_vessels) * 100
        print(f"対象船舶割合:        {percentage:>6.2f} %")
    print("-" * 60)
    # Consistency check
    if cut_target_vessels == actual_cut_count:
        print("整合性チェック: OK (対象船舶数 == 切り出し数)")
    else:
        print(
            f"整合性チェック: WARNING (対象船舶数 {cut_target_vessels} != 切り出し数 {actual_cut_count})"
        )
        print("  ※差分の原因: WAVファイルの範囲外、または切り出し処理の失敗")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a", "--ais_path", type=str, required=True, help="Path to the AIS folder."
    )
    parser.add_argument(
        "-w", "--wav_path", type=str, required=True, help="Path to the WAV folder."
    )
    parser.add_argument(
        "-m",
        "--toml_path",
        type=str,
        required=True,
        help="Path to the TOML metadata file.",
    )
    parser.add_argument(
        "-t",
        "--record_start_time",
        type=str,
        required=True,
        help="Record start time (ISO format, e.g., '2024-03-19T06:53:00').",
        # default="2024-03-19T06:53:00", # Remove default, make required
    )
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        default="config.toml",
        help="Path to the configuration TOML file.",
    )
    parser.add_argument(
        "-ff",
        "--fig_flag",
        action="store_true",
        help="Flag for making figures (geolocation and spectrograms).",
        default=False,
    )
    parser.add_argument(
        "-mf",
        "--movie_flag",
        action="store_true",
        help="Flag for making movies (Not currently implemented).",
        default=False,
    )
    parser.add_argument(
        "-cf",
        "--csv_flag",
        action="store_true",
        help="Flag for saving distance calculation results as CSV files.",
        default=False,
    )
    args = parser.parse_args()

    main(
        args.ais_path,
        args.wav_path,
        args.toml_path,
        args.record_start_time,
        args.fig_flag,
        args.movie_flag,
        args.csv_flag,
        args.config_path,  # Pass config path to main
    )
