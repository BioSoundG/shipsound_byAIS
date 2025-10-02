import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import soundfile as sf
from matplotlib import colors
from scipy import signal
import math  # Added for ceil


def plot_geolocation(idx, df, record_pos, output_dir):
    """
    Plots geolocation data for vessels and saves the output as a PNG file.

    Args:
        idx (int): Index for saving the output file.
        df (DataFrame): DataFrame containing the AIS data.
        record_pos (tuple): Tuple of the recording position (longitude, latitude).
        output_dir (str): Path to the output directory where the image will be saved.
    """
    fig, ax = plt.subplots()

    # 軸ラベルを最初に設定
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # NaN値が含まれる場合はフィルタリング
    if df.empty:
        unique_mmsi = []
    else:
        # 緯度・経度が欠損しているデータを除外
        df_clean = df.dropna(subset=["latitude", "longitude"])

        # 無限大の値を除外
        df_clean = df_clean[
            np.isfinite(df_clean["latitude"]) & np.isfinite(df_clean["longitude"])
        ]

        unique_mmsi = df_clean["mmsi"].unique()

    # 録音位置をプロット
    ax.scatter(
        record_pos[1], record_pos[0], c="blue", label="rec_pos", marker="*", s=100
    )  # Made color explicit and larger
    ax.text(
        record_pos[1],
        record_pos[0],
        "rec_pos",
        fontsize=10,
        ha="right",
        va="bottom",  # Adjusted alignment slightly
    )

    handles, labels = [], []
    # Use a colormap for vessel tracks
    cmap = plt.get_cmap("tab10")
    num_vessels = len(unique_mmsi)

    for i, mmsi in enumerate(unique_mmsi):
        # 特定のMMSIの船舶データを抽出
        vessel_df = df[df["mmsi"] == mmsi]

        # 欠損値や無限大の値をフィルタリング
        vessel_df = vessel_df.dropna(subset=["latitude", "longitude"])
        vessel_df = vessel_df[
            np.isfinite(vessel_df["latitude"]) & np.isfinite(vessel_df["longitude"])
        ]

        # データが空でない場合のみプロット
        if not vessel_df.empty:
            vessel_name = vessel_df["vessel_name"].iloc[0]
            color = cmap(
                i / num_vessels if num_vessels > 0 else 0
            )  # Assign color based on index

            # 船舶の位置をプロット (scatter for points)
            scatter = ax.scatter(
                vessel_df["longitude"],
                vessel_df["latitude"],
                label=vessel_name,
                color=color,
                s=20,  # Smaller points
            )
            handles.append(scatter)
            labels.append(vessel_name)

            # 時間順にソートして軌跡をプロット (plot for line)
            vessel_df_sorted = vessel_df.sort_values("dt_pos_utc")
            ax.plot(
                vessel_df_sorted["longitude"],
                vessel_df_sorted["latitude"],
                linestyle="-",
                color=color,
                alpha=0.7,
            )

            # 各位置に時間情報をテキストとして表示 (Consider reducing frequency if too cluttered)
            # Add only start and end times? Or every Nth point?
            # For now, keep as is, but be aware it can get cluttered.
            # for _, row in vessel_df.iterrows():
            #     # NaNや無限大でないことを再確認 (Redundant check, already filtered)
            #     ax.text(
            #         row["longitude"],
            #         row["latitude"],
            #         row["dt_pos_utc"].strftime(
            #             "%H:%M:%S"
            #         ),  # Shorter time format for points
            #         fontsize=7,  # Smaller font
            #         ha="left",
            #         va="top",
            #         color=color,
            #         alpha=0.8,
            #     )

    # 凡例は船舶がある場合のみ表示
    # if handles:
    #     ax.legend(
    #         handles, labels, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8
    #     )

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.6)

    # 画像として保存
    plt.savefig(
        os.path.join(output_dir, f"{idx}.png"), bbox_inches="tight", dpi=150
    )  # Slightly higher DPI
    plt.close(fig)  # メモリリーク防止のためフィギュアを閉じる


def plot_mother_source_spectrogram(
    wav_path,
    distances_df_list,
    record_start_time,
    output_dir,
    vis_config,  # Config dictionary now includes 'cut_margin_minutes'
):
    """
    Creates a time-averaged spectrogram for each mother source WAV file by processing
    in chunks. The X-axis represents the full duration with absolute time labels,
    but the plotted intensity is averaged over specified time intervals.
    Parameters are controlled via the vis_config dictionary.

    Args:
        wav_path (str): Path to the directory containing WAV files.
        distances_df_list (list or pd.DataFrame): List of DataFrames or a single DataFrame
                                                 containing distance information for cut annotations.
        record_start_time (datetime): Start time of the recording (absolute).
        output_dir (str): Path to the output directory where images will be saved.
        vis_config (dict): Dictionary containing visualization parameters from config.toml.
    """
    # Load parameters from config
    chunk_duration_seconds = vis_config.get("chunk_duration_seconds", 600)
    nperseg = vis_config.get("spectrogram_nperseg", 4096)
    noverlap = nperseg // 2
    max_freq_bins = vis_config.get("plot_max_freq_bins", 200)
    db_min = vis_config.get("plot_db_min", -80)
    db_max = vis_config.get("plot_db_max", -10)
    max_cuts_to_display = vis_config.get("plot_max_cuts", 15)
    plot_dpi = vis_config.get("plot_dpi", 150)
    # Get cut_margin_minutes from the passed vis_config
    cut_margin_min = vis_config.get(
        "cut_margin_minutes", 1
    )  # Default to 1 if not found

    # Create a directory for spectrograms if it doesn't exist
    spec_output_dir = os.path.join(output_dir, "spectrograms")
    os.makedirs(spec_output_dir, exist_ok=True)

    # Convert single DataFrame to list if necessary
    if isinstance(distances_df_list, pd.DataFrame):
        distances_df_list = [distances_df_list]
    elif distances_df_list is None:
        distances_df_list = []

    # Process each WAV file in the wav_path directory
    wav_files = [f for f in os.listdir(wav_path) if f.upper().endswith(".WAV")]
    wav_files.sort()  # Sort to ensure order

    if not wav_files:
        print("No WAV files found in the specified directory.")
        return

    # Combine all distance DataFrames for cut annotation lookup
    combined_distances_df = (
        pd.concat(distances_df_list, ignore_index=True)
        if distances_df_list
        else pd.DataFrame()
    )
    if combined_distances_df.empty:
        print("No distance data available. Cut annotations will not be shown.")

    # Track cumulative time to know the absolute start time of each file
    cumulative_time = 0

    # Process each mother source WAV file
    for wav_file in wav_files:  # Changed loop variable name for clarity
        absolute_wav_path = os.path.join(wav_path, wav_file)
        try:
            # Get file information using SoundFile context manager
            with sf.SoundFile(absolute_wav_path) as f_soundfile:
                original_samplerate = f_soundfile.samplerate
                original_total_samples = len(f_soundfile)
                original_duration_full = original_total_samples / original_samplerate

            file_name = os.path.basename(wav_file)
            file_start_time = record_start_time + pd.Timedelta(seconds=cumulative_time)
            file_end_time_abs = file_start_time + pd.Timedelta(
                seconds=original_duration_full
            )

            print(f"Processing time-averaged spectrogram for {file_name}...")
            print(
                f"  Total duration: {original_duration_full:.2f}s, Sample rate: {original_samplerate}Hz"
            )
            print(
                f"  Absolute time range: {file_start_time.strftime('%Y-%m-%d %H:%M:%S')} - {file_end_time_abs.strftime('%Y-%m-%d %H:%M:%S')}"
            )

            # --- Chunk Processing Setup ---
            chunk_duration_seconds = chunk_duration_seconds
            avg_spectra_list = []
            chunk_start_times_sec = []
            frequency_axis = None

            # Re-open file for reading chunks within the context manager
            with sf.SoundFile(absolute_wav_path) as f_soundfile:
                num_chunks = math.ceil(original_duration_full / chunk_duration_seconds)
                samples_per_chunk = int(chunk_duration_seconds * original_samplerate)
                print(
                    f"  Processing in {num_chunks} chunks of ~{chunk_duration_seconds}s..."
                )

                for chunk_idx in range(num_chunks):
                    start_sample = chunk_idx * samples_per_chunk
                    num_samples_to_read = min(
                        samples_per_chunk, original_total_samples - start_sample
                    )

                    if num_samples_to_read <= 0:
                        break

                    print(
                        f"    Processing chunk {chunk_idx + 1}/{num_chunks} ({(chunk_idx * chunk_duration_seconds):.1f}s - {((chunk_idx + 1) * chunk_duration_seconds):.1f}s)..."
                    )

                    # Read data for the current chunk
                    f_soundfile.seek(start_sample)
                    data = f_soundfile.read(
                        num_samples_to_read, dtype="float32", always_2d=False
                    )

                    if len(data) == 0:
                        print(
                            f"    Warning: Read 0 samples for chunk {chunk_idx + 1}. Skipping."
                        )
                        continue

                    # If stereo, convert to mono
                    if len(data.shape) > 1 and data.shape[1] > 1:
                        data = data[:, 0]

                    # Calculate spectrogram parameters
                    # nperseg = 4096 # Keep consistent for frequency axis
                    # noverlap = nperseg // 2

                    try:
                        # Calculate spectrogram for the chunk
                        f, t_chunk, Sxx_chunk = signal.spectrogram(
                            data,
                            original_samplerate,
                            nperseg=nperseg,
                            noverlap=noverlap,
                        )

                        if frequency_axis is None:
                            frequency_axis = (
                                f  # Store frequency axis from the first chunk
                            )

                        # Average spectrogram power over time axis for this chunk
                        avg_spectrum_chunk_power = np.mean(Sxx_chunk, axis=1)

                        # Store results
                        chunk_start_times_sec.append(chunk_idx * chunk_duration_seconds)
                        avg_spectra_list.append(avg_spectrum_chunk_power)

                    except ValueError as ve:
                        print(
                            f"    Error calculating spectrogram for chunk {chunk_idx + 1}: {ve}. Check if chunk length ({len(data)} samples) is sufficient for nperseg ({nperseg}). Skipping chunk."
                        )
                        continue
                    finally:
                        del data, Sxx_chunk  # Free memory explicitly

            # --- Combine Chunk Results ---
            if not avg_spectra_list:
                print(
                    f"  No valid spectrogram chunks processed for {file_name}. Skipping plot."
                )
                cumulative_time += (
                    original_duration_full  # Still advance time for next file
                )
                continue

            avg_spectra_all_chunks = np.array(
                avg_spectra_list
            ).T  # Shape: [freq_bins, num_chunks]
            chunk_time_axis = np.array(chunk_start_times_sec)
            f = frequency_axis  # Use frequency axis from first valid chunk

            # Reduce frequency resolution for plotting if necessary (similar to old logic)
            if len(f) > max_freq_bins:
                bin_size = math.ceil(
                    len(f) / max_freq_bins
                )  # Use ceil to ensure coverage
                f_reduced = np.array(
                    [
                        f[i : min(i + bin_size, len(f))].mean()
                        for i in range(0, len(f), bin_size)
                    ]
                )
                Sxx_reduced = np.array(
                    [
                        avg_spectra_all_chunks[i : min(i + bin_size, len(f)), :].mean(
                            axis=0
                        )
                        for i in range(0, len(f), bin_size)
                    ]
                )
                f = f_reduced
                avg_spectra_all_chunks = Sxx_reduced
                print(f"  Reduced frequency resolution to {len(f)} bins for plotting.")

            # --- Plotting the Time-Averaged Spectrogram ---
            plt.figure(figsize=(12, 7))

            # Convert average power to dB
            db_avg_spectra = 10 * np.log10(avg_spectra_all_chunks + 1e-10)

            # Define time bin edges for pcolormesh for better alignment
            time_edges = np.append(
                chunk_time_axis, chunk_time_axis[-1] + chunk_duration_seconds
            )
            # Ensure time_edges has correct length relative to data
            if len(time_edges) != db_avg_spectra.shape[1] + 1:
                print(
                    f"Warning: Time edges length ({len(time_edges)}) mismatch with data columns ({db_avg_spectra.shape[1]}). Adjusting edges."
                )
                # Fallback or correction logic might be needed here if the primary logic fails
                # For now, assume the calculation is correct and proceed

            # Calculate frequency bin edges required for shading='flat'
            if f is not None and len(f) > 1:
                f_midpoints = (f[:-1] + f[1:]) / 2
                f_first_edge = (
                    f[0] - (f_midpoints[0] - f[0])
                    if len(f_midpoints) > 0
                    else f[0] * 0.9
                )  # Handle len(f)==2 case
                f_last_edge = (
                    f[-1] + (f[-1] - f_midpoints[-1])
                    if len(f_midpoints) > 0
                    else f[-1] * 1.1
                )  # Handle len(f)==2 case
                frequency_edges = np.concatenate(
                    ([max(0, f_first_edge)], f_midpoints, [f_last_edge])
                )
                # Ensure positivity and monotonicity
                frequency_edges[frequency_edges < 0] = 0
                frequency_edges = np.sort(frequency_edges)
            elif f is not None and len(f) == 1:
                delta_f = f[0] * 0.1 if f[0] > 0 else 1.0  # Avoid delta_f=0
                frequency_edges = np.array(
                    [max(0, f[0] - delta_f / 2), f[0] + delta_f / 2]
                )
            else:
                print("Error: Frequency axis (f) is invalid. Using dummy edges.")
                frequency_edges = np.linspace(
                    0, 1, db_avg_spectra.shape[0] + 1
                )  # Dummy based on data rows

            plt.pcolormesh(
                time_edges,  # Use edges of time bins
                frequency_edges,  # Use frequency edges
                db_avg_spectra,
                shading="flat",  # 'flat' requires X/Y length = data dim + 1
                norm=colors.Normalize(vmin=db_min, vmax=db_max),  # Use config values
                cmap="viridis",
                rasterized=True,
            )

            plt.ylabel("Frequency [Hz]")
            plt.xlabel(f"Time (averaged over {chunk_duration_seconds}s intervals)")

            # Set the x-axis limits explicitly to the full duration
            plt.xlim(0, original_duration_full)

            # Format x-axis with real time using the full duration context
            time_format = "%Y-%m-%d %H:%M:%S"
            plt.gca().xaxis.set_major_formatter(
                plt.FuncFormatter(
                    lambda x, pos: (file_start_time + pd.Timedelta(seconds=x)).strftime(
                        time_format
                    )
                )
            )
            # Adjust x-axis ticks for better time display
            num_ticks = min(
                10, int(original_duration_full / (chunk_duration_seconds / 2)) + 1
            )  # Heuristic based on chunks
            plt.gca().xaxis.set_major_locator(
                plt.MaxNLocator(num_ticks, prune="both")
            )  # Prune ends if needed
            plt.xticks(rotation=45, ha="right")  # Rotate more, adjust alignment

            plt.colorbar(label="Average Intensity [dB]")

            # Show absolute time range in title (using full duration)
            plt.title(
                f"Time-Averaged Spectrogram of {file_name} (Interval: {chunk_duration_seconds}s)\n"
                f"Time range: {file_start_time.strftime('%Y-%m-%d %H:%M:%S')} - "
                f"{file_end_time_abs.strftime('%Y-%m-%d %H:%M:%S')}",
                fontsize=10,
            )

            # --- Find and Plot Cuts ---
            file_related_cuts = []
            if (
                not combined_distances_df.empty
                and "min_distance_time" in combined_distances_df.columns
            ):
                print(
                    f"  Checking for cuts within {file_start_time} to {file_end_time_abs}..."
                )
                for _, row in combined_distances_df.iterrows():
                    if pd.isna(row.get("min_distance_time")):
                        continue

                    min_distance_time = row["min_distance_time"]
                    # Use cut_margin_min from vis_config here
                    start_cut_time_abs = min_distance_time - pd.Timedelta(
                        minutes=cut_margin_min
                    )
                    end_cut_time_abs = min_distance_time + pd.Timedelta(
                        minutes=cut_margin_min
                    )

                    # Check if cut overlaps with the file's time range
                    if (start_cut_time_abs < file_end_time_abs) and (
                        end_cut_time_abs > file_start_time
                    ):
                        cut_start_sec = max(
                            0, (start_cut_time_abs - file_start_time).total_seconds()
                        )
                        cut_end_sec = min(
                            original_duration_full,
                            (end_cut_time_abs - file_start_time).total_seconds(),
                        )

                        if cut_start_sec < cut_end_sec:
                            vessel_name = (
                                row.get("vessel_name", "Unknown")
                                if pd.notna(row.get("vessel_name", "Unknown"))
                                else "Unknown"
                            )
                            mmsi = row.get("mmsi", "Unknown")
                            distance = row.get("min_distance [m]", 0)
                            vessel_info = f"{vessel_name} - {mmsi} ({int(distance)}m)"
                            file_related_cuts.append(
                                (cut_start_sec, cut_end_sec, vessel_info)
                            )

            if not file_related_cuts:
                print(f"  No cuts found for file {file_name}")
            else:
                print(f"  Found {len(file_related_cuts)} cuts for file {file_name}")

            # Limit the number of cuts displayed to prevent clutter
            # max_cuts_to_display = 15 # Increased limit slightly
            displayed_cuts = file_related_cuts
            if len(file_related_cuts) > max_cuts_to_display:
                displayed_cuts = file_related_cuts[:max_cuts_to_display]
                # Add text indicating not all cuts are shown (optional)
                # plt.figtext(...)

            # Add vertical lines and annotations for each displayed cut
            for j, (start_sec, end_sec, vessel_info) in enumerate(displayed_cuts):
                color = plt.cm.tab10(j % 10)
                plt.axvline(
                    x=start_sec, color=color, linestyle="-", alpha=0.7, linewidth=1.5
                )
                plt.axvline(
                    x=end_sec, color=color, linestyle="-", alpha=0.7, linewidth=1.5
                )
                plt.axvspan(
                    start_sec, end_sec, color=color, alpha=0.05
                )  # Lighter alpha for span

                # Adjusted text placement logic
                text_y_pos = frequency_edges[-1] * (
                    0.95 - (j % 5) * 0.06
                )  # Cycle y-position more frequently
                text_x_pos = (start_sec + end_sec) / 2
                # Ensure text is within plot bounds
                text_x_pos = max(text_x_pos, plt.xlim()[0] * 1.05)
                text_x_pos = min(text_x_pos, plt.xlim()[1] * 0.95)

                plt.text(
                    text_x_pos,
                    text_y_pos,
                    vessel_info,
                    horizontalalignment="center",
                    verticalalignment="top",
                    color=color,
                    fontweight="bold",
                    fontsize=8,  # Slightly smaller font
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor="white",
                        edgecolor=color,
                        alpha=0.8,
                    ),
                )

            # --- Finalize Plot ---
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Add slight margin at bottom/top

            output_path = os.path.join(
                spec_output_dir, f"spec_{os.path.splitext(file_name)[0]}.png"
            )
            plt.savefig(
                output_path,
                dpi=plot_dpi,  # Use config value
                bbox_inches="tight",
                format="png",
            )
            plt.close()
            print(f"  Created time-averaged spectrogram: {output_path}")

            # Free memory explicitly
            del (
                avg_spectra_list,
                avg_spectra_all_chunks,
                chunk_time_axis,
                f,
                db_avg_spectra,
            )

            # Update cumulative time for next file using the definitive duration
            cumulative_time += original_duration_full

        except Exception as e:
            print(
                f"Error processing time-averaged spectrogram for {absolute_wav_path}: {str(e)}"
            )
            # Ensure plot is closed if error occurs mid-plot
            try:
                plt.close()
            except:
                pass
            # Update cumulative time even if error occurs to process next file correctly
            try:
                # Try to get duration again if possible
                with sf.SoundFile(absolute_wav_path) as f_err:
                    err_duration = len(f_err) / f_err.samplerate
                cumulative_time += err_duration
                print(
                    f"  Updated cumulative time by {err_duration:.2f}s despite error."
                )
            except Exception as e_dur:
                print(
                    f"  Could not get duration for errored file {wav_file} to update time: {e_dur}"
                )
                # Cannot reliably update cumulative_time, might affect subsequent files.
            continue  # Continue with next file
