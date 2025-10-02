import os
import argparse
from natsort import natsorted
import glob
import tomli
import pandas as pd

# Use tomllib if available (Python 3.11+), otherwise keep tomli
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # Fallback to tomli if tomllib not found

from data_processing import read_ais, complement_trajectory
from distance_calculation import calculate_shortest_distance, haversine
from visualization import plot_geolocation, plot_mother_source_spectrogram
from audio_processing import cut_wav_and_make_metadata


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
    config_path,  # Add config_path
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

    # Process each AIS file
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
        if flag_fig:
            # Pass vis_config if plot_geolocation needs it in the future
            print("plot geolocation.....")
            plot_geolocation(idx + 1, ais_df, record_pos, output_dir)
        comp_df = complement_trajectory(ais_data)
        distances = calculate_shortest_distance(comp_df, record_pos, record_depth)
        distances_df = pd.DataFrame(distances)

        # Add to collection for spectrograms
        all_distances_dfs.append(distances_df)

        if flag_csv:
            distances_df.to_csv(
                os.path.join(output_dir, f"distances_{idx+1}.csv"), index=False
            )

        # Pass audio_config to audio_processing
        cut_wav_and_make_metadata(
            wav_list,
            meta_data,
            start_tim,
            distances_df,
            pd.DataFrame(),  # You can handle the distance list as needed.
            output_dir,
            record_pos,
            audio_config,  # Pass audio config dictionary
        )

    # Create mother source spectrograms with cut indicators if flag_fig is True
    if flag_fig and wav_list:
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
