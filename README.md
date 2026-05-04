# AIS and WAV Processing Project

This project is designed to process AIS (Automatic Identification System) data, WAV audio files, and associated metadata to analyze vessel trajectories, calculate distances, cut relevant audio segments, and generate visualizations like geolocation plots and time-averaged spectrograms. The project is modular, with functionalities separated across multiple files and configured via a central TOML file.

## Project Structure

```bash
.
├── audio_processing.py       # Audio processing: WAV file cutting and metadata generation
├── data_processing.py        # Data processing: reading and interpolating AIS data
├── distance_calculation.py   # Distance calculation between vessels and recording position
├── visualization.py          # Visualization: geographical plots and spectrograms
├── main.py                   # Main execution script coordinating the workflow
├── config.toml               # Configuration file for parameters and flags
├── metadata.toml             # Metadata template
├── metadata_izuoshima.toml   # Location-specific metadata file 
├── check_csv.py              # CSV data checking utility
├── test_plot_geo.py          # Geographical plot testing utility
├── test_plot_geo_debug.py    # Debug version of geographical plot utility
├── test_visualization.py     # Visualization testing utility
├── requirements.txt          # Dependencies required for the project
├── test/                     # Directory containing unit tests
│   ├── test_audio_processing.py  # Tests for audio processing
│   ├── test_data_processing.py   # Tests for data processing
│   ├── test_distance_calculation.py # Tests for distance calculations
│   ├── test_main.py              # Tests for main script
│   ├── test_visualization.py     # Tests for visualization
│   ├── run_tests.py              # Test runner script
│   ├── README.md                 # Test specifications 
│   └── __init__.py               # Package initialization file
├── test_output/              # Test output directory
├── test_output_debug/        # Debug test output directory
├── ais_example/              # Sample AIS data
├── wav_example/              # Sample WAV files
├── prompt/                   # Project prompt information
├── Cases_of_caution.md       # Notes on development practices and optimization
├── .gitignore                # Specifies intentionally untracked files for Git
└── README.md                 # Project documentation (this file)
```

## Modules Overview

### `data_processing.py`
Contains functions for reading and processing AIS data, converting timestamps, and complementing vessel trajectories with interpolated data.

### `distance_calculation.py`
Contains functions to calculate the shortest distance between vessels and the recording position, using the Haversine formula.

### `visualization.py`
Includes functions for generating visualizations:
- **Geolocation Plots:** Show vessel trajectories relative to the recording point.
- **Time-Averaged Spectrograms:** Display the frequency content of the mother source WAV file over its entire duration. For long files, the spectrogram represents time-averaged power spectral density to manage memory usage. Vertical lines indicate the calculated cut sections for nearby vessels.

### `audio_processing.py`
Contains functions to cut WAV audio files based on calculated time ranges (derived from vessel proximity) and generate corresponding metadata appendix files. The cutting process considers multiple conditions:

1. **Time Range Selection:** The time range for each cut is centered around the time of minimum distance between the vessel and recording position.
2. **Distance Threshold:** Only vessels that come within a configurable maximum distance (`max_cut_distance` in config.toml) are considered for cutting.
3. **Isolation Condition:** A vessel segment is only cut when, at its time of minimum distance, it is the closest vessel to the recording position (compared to all other vessels).
4. **Margin Configuration:** The time margin before and after the minimum distance point is configurable (`cut_margin_minutes` in config.toml).

The cut WAV files are stored with metadata about the vessel, including MMSI, vessel type, vessel name, minimum distance, and timestamps.

### `main.py`
The main script that integrates all modules. It reads the configuration from `config.toml`, executes the pipeline (reading data, calculating distances, optionally creating visualizations and cutting WAV files based on the config), and saves the results.

### `config.toml`
A configuration file using the TOML format to set parameters for various processing steps, such as file paths, calculation thresholds, plot settings, and output flags (e.g., whether to generate figures or cut audio). This allows easy modification of the project's behavior without changing the code.

Key audio processing parameters include:
- `cut_margin_minutes`: The time margin (in minutes) before and after the point of minimum distance for WAV cutting
- `max_cut_distance`: Maximum distance threshold (in meters) for vessel consideration
- `check_other_vessels`: Whether to check if a vessel is the closest at its minimum distance time

## Requirements

1.  **Python:** (Specify version if applicable, e.g., Python 3.8+ recommended)
2.  **Dependencies:** Install the necessary libraries using `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Configuration File:** A `config.toml` file must be present in the project root directory. You can copy and modify the provided template or create your own based on the required parameters. See `config.toml.template` (if provided) or the parameter usage within the scripts for details.

## Usage

To execute the pipeline, run the `main.py` script, providing the necessary command-line arguments:

```bash
python main.py -a AIS_PATH -w WAV_PATH -m TOML_PATH -t RECORD_START_TIME [-c CONFIG_PATH] [-ff] [-mf] [-cf]
```

To combine multiple `acoustic_features.csv` files and generate plots from the
combined dataset:

```bash
python aggregate_acoustic_features.py -i OUTPUT_DIR_OR_CSV [MORE_DIRS_OR_CSVS ...] -o combined_acoustic_features_analysis
```

To limit the analysis to a vessel speed range, add `--speed-min` and/or
`--speed-max`:

```bash
python aggregate_acoustic_features.py -i OUTPUT_DIR_OR_CSV -o combined_acoustic_features_analysis --speed-min 1 --speed-max 40
```

When an input is a directory, the script searches it recursively for files named
`acoustic_features.csv`. It saves `combined_acoustic_features.csv`, a summary
text file, and the generated figures in the output directory.

To estimate vessel tonnage from AIS static fields and plot a log-scale tonnage
frequency distribution:

```bash
python plot_vessel_tonnage_distribution.py -a AIS_CSV_DIR [MORE_AIS_DIRS_OR_CSVS ...] -t CUT_TOML_DIR [MORE_CUT_TOML_DIRS_OR_FILES ...] -o vessel_tonnage_distribution
```

The script reads target MMSIs from the cut metadata TOML files, then looks up
those vessels in all AIS CSV files found under the AIS input directories.
Tonnage is estimated from AIS `length`, `width`, and `draught` fields when
available. If draught is unavailable, it falls back to `length * width *
coefficient`. The plot uses 300 tons as the default lower x-axis limit. Vessels
below 300 tons are excluded from the plotted histogram and reported separately
in `vessel_tonnage_distribution_summary.txt`. The script also saves
`vessel_tonnage_estimates.csv`.

### Command-line Arguments

The script accepts the following arguments:

- `-a`, `--ais_path` (required): Path to the folder containing AIS data CSV files.
- `-w`, `--wav_path` (required): Path to the folder containing WAV audio files.
- `-m`, `--toml_path` (required): Path to the TOML metadata file with observation information.
- `-t`, `--record_start_time` (required): Record start time in ISO format (e.g., '2024-03-19T06:53:00').
- `-c`, `--config_path` (optional): Path to the configuration TOML file. Defaults to 'config.toml'.
- `-ff`, `--fig_flag` (flag): Generate visualization figures (geolocation plots and spectrograms).
- `-mf`, `--movie_flag` (flag): Generate movies (not currently implemented).
- `
