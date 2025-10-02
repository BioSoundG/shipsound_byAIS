import os
import sys
import unittest
import pandas as pd
import numpy as np
import datetime
import tempfile
import matplotlib.pyplot as plt
import matplotlib
from unittest.mock import patch, MagicMock
from PIL import Image
import math
import soundfile as sf

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from visualization import plot_geolocation, plot_mother_source_spectrogram

# Try loading config for tests
try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        tomllib = None
        print(
            "Warning: tomllib or tomli not found. Using default config values in tests."
        )


def load_test_config():
    # Define defaults robustly
    default_config = {
        "visualization": {
            "chunk_duration_seconds": 600,
            "spectrogram_nperseg": 4096,
            "plot_max_freq_bins": 200,
            "plot_db_min": -80,
            "plot_db_max": -10,
            "plot_max_cuts": 15,
            "plot_dpi": 150,
        },
        "audio_processing": {"cut_margin_minutes": 1},
    }
    if tomllib:
        try:
            # Assume config.toml is in the parent directory relative to the test file
            config_file_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "config.toml")
            )
            print(f"Attempting to load test config from: {config_file_path}")
            with open(config_file_path, "rb") as f:
                loaded_config = tomllib.load(f)
                # Merge loaded config with defaults (handle potential missing keys)
                vis_conf = loaded_config.get("visualization", {})
                audio_conf = loaded_config.get("audio_processing", {})
                default_config["visualization"].update(vis_conf)
                default_config["audio_processing"].update(audio_conf)
                print(f"Test config loaded successfully from {config_file_path}")
                return default_config
        except FileNotFoundError:
            print(f"{config_file_path} not found for tests. Using default values.")
            return default_config
        except Exception as e:
            print(
                f"Error loading {config_file_path} for tests: {e}. Using default values."
            )
            return default_config
    else:
        print("tomllib/tomli not found. Using default config values.")
        return default_config


class TestVisualization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load config once for all tests in this class
        cls.config = load_test_config()
        cls.vis_config = cls.config.get("visualization", {})
        cls.audio_config = cls.config.get("audio_processing", {})
        # Ensure essential defaults are present if loading failed or key missing
        cls.vis_config.setdefault("chunk_duration_seconds", 600)
        cls.vis_config.setdefault("spectrogram_nperseg", 4096)
        cls.vis_config.setdefault("plot_max_freq_bins", 200)
        cls.vis_config.setdefault("plot_db_min", -80)
        cls.vis_config.setdefault("plot_db_max", -10)
        cls.vis_config.setdefault("plot_max_cuts", 15)
        cls.vis_config.setdefault("plot_dpi", 150)
        cls.audio_config.setdefault("cut_margin_minutes", 1)

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
        self.test_df = pd.DataFrame(
            {
                "mmsi": [123, 123, 456, 456],
                "vessel_name": ["A", "A", "B", "B"],
                "vessel_type": ["C", "C", "P", "P"],
                "latitude": [32.7, 32.71, 32.72, 32.73],
                "longitude": [129.7, 129.71, 129.72, 129.73],
                "dt_pos_utc": pd.to_datetime(
                    [
                        "2024-03-19 07:00:00",
                        "2024-03-19 07:01:00",
                        "2024-03-19 07:02:00",
                        "2024-03-19 07:03:00",
                    ]
                ),
                "length": [100, 100, 150, 150],  # Added for distance calc if needed
                "width": [20, 20, 25, 25],  # Added for distance calc if needed
            }
        )
        self.record_pos = (129.77558, 32.71161)  # Lon, Lat order for scatter
        self.original_style = plt.rcParams.copy()
        self.images = []
        # Use the class-level config by default
        self.vis_config = TestVisualization.vis_config
        # Make a copy for tests that might modify it locally
        self.local_vis_config = self.vis_config.copy()

    def tearDown(self):
        plt.rcParams.update(self.original_style)
        plt.close("all")
        for img in self.images:
            if hasattr(img, "close"):
                try:
                    img.close()
                except:
                    pass
        self.temp_dir.cleanup()

    # --- Geolocation Tests ---
    def test_plot_geolocation_output_file_existence(self):
        plot_geolocation(1, self.test_df, self.record_pos, self.output_dir)
        output_file = os.path.join(self.output_dir, "1.png")
        self.assertTrue(os.path.exists(output_file))
        self.assertGreater(os.path.getsize(output_file), 1024)

    def test_plot_geolocation_image_properties(self):
        plot_geolocation(2, self.test_df, self.record_pos, self.output_dir)
        output_file = os.path.join(self.output_dir, "2.png")
        try:
            img = Image.open(output_file)
            self.images.append(img)
            self.assertGreaterEqual(img.width, 100)
            self.assertGreaterEqual(img.height, 100)
            self.assertIn(img.mode, ["RGB", "RGBA"])
            self.assertEqual(img.format, "PNG")
        except Exception as e:
            self.fail(f"Image property test failed: {e}")

    def test_plot_geolocation_content(self):
        plot_geolocation(3, self.test_df, self.record_pos, self.output_dir)
        output_file = os.path.join(self.output_dir, "3.png")
        self.assertTrue(os.path.exists(output_file))
        img = plt.imread(output_file)
        self.assertGreater(img.shape[0], 0)
        self.assertGreater(img.shape[1], 0)
        self.assertIn(img.shape[2], [3, 4])
        self.assertFalse(np.all(img == 0))
        self.assertFalse(np.all(img == 1))

    def test_plot_geolocation_empty_df(self):
        empty_df = pd.DataFrame(columns=self.test_df.columns)
        try:
            plot_geolocation(4, empty_df, self.record_pos, self.output_dir)
            output_file = os.path.join(self.output_dir, "4.png")
            self.assertTrue(os.path.exists(output_file))
            img = Image.open(output_file)
            self.images.append(img)
            self.assertIsNotNone(img)
            img_array = plt.imread(output_file)
            self.assertGreater(img_array.size, 0)
        except Exception as e:
            self.fail(f"Empty DF test failed: {e}")

    def test_plot_geolocation_one_vessel(self):
        single_vessel_df = self.test_df[self.test_df["mmsi"] == 123].copy()
        plot_geolocation(5, single_vessel_df, self.record_pos, self.output_dir)
        output_file = os.path.join(self.output_dir, "5.png")
        self.assertTrue(os.path.exists(output_file))
        try:
            img_array = plt.imread(output_file)
            self.assertGreater(img_array.shape[0], 0)
            self.assertGreater(img_array.shape[1], 0)
            self.assertIn(img_array.shape[2], [3, 4])
            self.assertFalse(np.all(img_array == 0))
            self.assertFalse(np.all(img_array == 1))
        except Exception as e:
            self.fail(f"Single vessel test failed: {e}")

    def test_plot_geolocation_edge_case_same_coordinates(self):
        same_coord_df = self.test_df.copy()
        same_coord_df["latitude"] = 32.7
        same_coord_df["longitude"] = 129.7
        try:
            plot_geolocation(6, same_coord_df, self.record_pos, self.output_dir)
            output_file = os.path.join(self.output_dir, "6.png")
            self.assertTrue(os.path.exists(output_file))
        except Exception as e:
            self.fail(f"Same coord test failed: {e}")

    def test_plot_geolocation_extreme_coordinates(self):
        extreme_df = self.test_df.iloc[:2].copy()
        extreme_df["latitude"] = [89.9, -89.9]
        extreme_df["longitude"] = [179.9, -179.9]
        try:
            plot_geolocation(7, extreme_df, self.record_pos, self.output_dir)
            output_file = os.path.join(self.output_dir, "7.png")
            self.assertTrue(os.path.exists(output_file))
        except Exception as e:
            self.fail(f"Extreme coord test failed: {e}")

    @patch("matplotlib.pyplot.close")
    @patch("matplotlib.pyplot.savefig")
    def test_plot_geolocation_axis_labels_and_ranges(self, mock_savefig, mock_close):
        plot_geolocation(8, self.test_df, self.record_pos, self.output_dir)
        self.assertTrue(mock_savefig.called)
        fig = plt.gcf()
        ax = fig.gca()
        self.assertEqual(ax.get_xlabel(), "Longitude")
        self.assertEqual(ax.get_ylabel(), "Latitude")
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        min_lon = self.test_df["longitude"].min()
        max_lon = self.test_df["longitude"].max()
        min_lat = self.test_df["latitude"].min()
        max_lat = self.test_df["latitude"].max()
        rec_lon, rec_lat = self.record_pos
        exp_min_lon = min(min_lon, rec_lon)
        exp_max_lon = max(max_lon, rec_lon)
        exp_min_lat = min(min_lat, rec_lat)
        exp_max_lat = max(max_lat, rec_lat)
        self.assertLessEqual(xlim[0], exp_min_lon)
        self.assertGreaterEqual(xlim[1], exp_max_lon)
        self.assertLessEqual(ylim[0], exp_min_lat)
        self.assertGreaterEqual(ylim[1], exp_max_lat)
        plt.close(fig)

    def test_plot_geolocation_nan_values(self):
        df_with_nan = self.test_df.copy()
        df_with_nan.loc[1, "latitude"] = np.nan
        df_with_nan.loc[3, "longitude"] = np.nan
        try:
            plot_geolocation(9, df_with_nan, self.record_pos, self.output_dir)
            output_file = os.path.join(self.output_dir, "9.png")
            self.assertTrue(os.path.exists(output_file))
            img = Image.open(output_file)
            self.images.append(img)
            self.assertIsNotNone(img)
        except Exception as e:
            self.fail(f"NaN value test failed: {e}")

    # --- Spectrogram Tests ---

    # Patch order: bottom-up -> savefig, listdir, mock_read, mock_soundfile
    @patch("soundfile.SoundFile")
    @patch("soundfile.read")
    @patch("os.listdir")
    @patch("matplotlib.pyplot.savefig")
    def test_plot_mother_source_spectrogram(
        self, mock_savefig, mock_listdir, mock_read, mock_soundfile
    ):
        test_wav_filename = "test_basic.wav"
        record_start_time = pd.Timestamp("2024-03-19 07:00:00")
        duration_seconds = 10  # File duration
        sample_rate = 44100
        # Use chunk duration shorter than file to test chunking logic path
        local_vis_config = self.local_vis_config.copy()  # Use local copy
        local_vis_config["chunk_duration_seconds"] = 5  # Shorter than duration_seconds

        mock_listdir.return_value = [test_wav_filename]

        mock_sf_instance = MagicMock()
        mock_sf_instance.__enter__.return_value.samplerate = sample_rate
        mock_sf_instance.__enter__.return_value.__len__.return_value = int(
            duration_seconds * sample_rate
        )
        mock_soundfile.return_value = mock_sf_instance

        # Mock read to return valid data for spectrogram calculation
        nperseg = local_vis_config.get("spectrogram_nperseg", 4096)
        # Need enough data for at least one segment
        # Chunk duration is 5s, so read needs 5s * sample_rate samples
        read_len = max(
            nperseg, int(local_vis_config["chunk_duration_seconds"] * sample_rate)
        )
        mock_read.return_value = (np.random.rand(read_len), sample_rate)

        distances_df = pd.DataFrame(
            {"mmsi": [1], "min_distance_time": [record_start_time]}
        )  # Need min_distance_time
        plot_mother_source_spectrogram(
            self.output_dir,
            [distances_df],
            record_start_time,
            self.output_dir,
            local_vis_config,
        )
        # Check if savefig was called (it should be, as we mocked read with enough data)
        self.assertTrue(mock_savefig.called, "plt.savefig was not called")
        self.assertEqual(mock_savefig.call_count, 1)
        args, kwargs = mock_savefig.call_args
        filepath = args[0]
        spec_output_dir = os.path.join(self.output_dir, "spectrograms")
        expected_filename_start = os.path.join(
            spec_output_dir, f"spec_{os.path.splitext(test_wav_filename)[0]}"
        )
        self.assertTrue(filepath.startswith(expected_filename_start))
        self.assertTrue(filepath.endswith(".png"))

    def test_plot_mother_source_spectrogram_save_spectrogram(self):
        spec_output_dir = os.path.join(self.output_dir, "spectrograms")
        wav_file_path = os.path.join(self.output_dir, "test_audio.wav")
        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        tone = np.sin(2 * np.pi * 440 * t)
        sf.write(wav_file_path, tone, sample_rate)
        test_time = pd.Timestamp("2024-03-19 07:00:00")
        distances_df = pd.DataFrame(
            {
                "mmsi": [123],
                "min_distance_time": [test_time],
                "vessel_name": ["Test"],
                "vessel_type": ["Cargo"],
                "length": [100],
                "width": [20],
                "min_distance [m]": [500],
            }
        )
        record_start_time = pd.Timestamp("2024-03-19 06:59:00")
        # Call with the actual temporary directory containing the WAV
        plot_mother_source_spectrogram(
            self.output_dir,
            [distances_df],
            record_start_time,
            self.output_dir,
            self.vis_config,  # Pass class config
        )
        expected_filename = "spec_test_audio.png"
        output_path = os.path.join(spec_output_dir, expected_filename)
        self.assertTrue(os.path.exists(output_path))
        self.assertGreater(os.path.getsize(output_path), 1024)
        try:
            img = Image.open(output_path)
            self.images.append(img)
            self.assertGreaterEqual(img.width, 100)
            self.assertGreaterEqual(img.height, 100)
            self.assertIn(img.mode, ["RGB", "RGBA"])
            img_array = np.array(img)
            self.assertFalse(np.all(img_array == 0))
            self.assertGreater(np.std(img_array), 1)  # Lower threshold for averaged
        except Exception as e:
            self.fail(f"Save spectrogram test failed: {e}")

    def test_plot_mother_source_spectrogram_with_cuts(self):
        spec_output_dir = os.path.join(self.output_dir, "spectrograms")
        wav_file_path = os.path.join(self.output_dir, "test_audio_cuts.wav")
        sample_rate = 44100
        duration = 5.0  # 5 sec wav
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        tone = np.sin(2 * np.pi * 440 * t)
        sf.write(wav_file_path, tone, sample_rate)
        record_start_time = pd.Timestamp("2024-03-19 07:00:00")
        distances_df = pd.DataFrame(
            {
                "mmsi": [123, 987, 555],
                "min_distance_time": [
                    record_start_time + pd.Timedelta(seconds=1),
                    record_start_time + pd.Timedelta(seconds=2.5),
                    record_start_time + pd.Timedelta(seconds=4),
                ],
                "vessel_name": ["A", "B", "C"],
                "vessel_type": ["C", "P", "T"],
                "length": [100, 150, 80],
                "width": [20, 25, 15],
                "min_distance [m]": [500, 800, 300],
            }
        )
        plot_mother_source_spectrogram(
            self.output_dir,
            [distances_df],
            record_start_time,
            self.output_dir,
            self.vis_config,  # Pass class config
        )
        expected_filename = "spec_test_audio_cuts.png"
        output_path = os.path.join(spec_output_dir, expected_filename)
        self.assertTrue(os.path.exists(output_path))
        try:
            img = Image.open(output_path)
            self.images.append(img)
            self.assertIsNotNone(img)
            self.assertGreaterEqual(img.width, 100)
            self.assertGreaterEqual(img.height, 100)
        except Exception as e:
            self.fail(f"Spectrogram with cuts test failed: {e}")

    # Patch order: close, savefig -> args: mock_savefig, mock_close
    @patch("matplotlib.pyplot.close")
    @patch("matplotlib.pyplot.savefig")
    def test_plot_mother_source_spectrogram_actual_time_axis(
        self, mock_savefig, mock_close
    ):
        wav_file_path = os.path.join(self.output_dir, "test_time_axis.wav")
        sample_rate = 44100
        duration = 5.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        tone = np.sin(2 * np.pi * 440 * t)
        sf.write(wav_file_path, tone, sample_rate)
        distances_df = pd.DataFrame(
            {"mmsi": [1], "min_distance_time": [pd.Timestamp("2024-03-19 07:00:01")]}
        )
        record_start_time = pd.Timestamp("2024-03-19 07:00:00")

        # Mocks needed for read/SoundFile inside the function
        # Patch order: SoundFile, read -> args: mock_read, mock_soundfile
        with patch("soundfile.SoundFile") as mock_soundfile, patch(
            "soundfile.read", return_value=(tone, sample_rate)
        ):

            mock_sf_instance = MagicMock()
            mock_sf_instance.__enter__.return_value.samplerate = sample_rate
            mock_sf_instance.__enter__.return_value.__len__.return_value = int(
                duration * sample_rate
            )
            mock_soundfile.return_value = mock_sf_instance

            plot_mother_source_spectrogram(
                self.output_dir,
                [distances_df],
                record_start_time,
                self.output_dir,
                self.vis_config,
            )

        self.assertTrue(mock_savefig.called)
        fig = plt.gcf()
        ax = fig.gca()
        xlim = ax.get_xlim()
        self.assertAlmostEqual(
            xlim[1],
            duration,
            delta=duration * 0.05,
            msg=f"X-axis limit ({xlim[1]}) not close to duration ({duration})",
        )
        plt.close(fig)

    # Patch order: close, savefig -> args: mock_savefig, mock_close
    @patch("matplotlib.pyplot.close")
    @patch("matplotlib.pyplot.savefig")
    def test_plot_mother_source_spectrogram_axis_labels_and_ranges(
        self, mock_savefig, mock_close
    ):
        wav_file_path = os.path.join(self.output_dir, "test_axis_labels.wav")
        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        tone = np.sin(2 * np.pi * 440 * t)
        sf.write(wav_file_path, tone, sample_rate)
        distances_df = pd.DataFrame(
            {"mmsi": [1], "min_distance_time": [pd.Timestamp("2024-03-19 07:00:00.5")]}
        )
        record_start_time = pd.Timestamp("2024-03-19 07:00:00")

        # Patch order: SoundFile, read -> args: mock_read, mock_soundfile
        with patch("soundfile.SoundFile") as mock_soundfile, patch(
            "soundfile.read", return_value=(tone, sample_rate)
        ):

            mock_sf_instance = MagicMock()
            mock_sf_instance.__enter__.return_value.samplerate = sample_rate
            mock_sf_instance.__enter__.return_value.__len__.return_value = int(
                duration * sample_rate
            )
            mock_soundfile.return_value = mock_sf_instance

            plot_mother_source_spectrogram(
                self.output_dir,
                [distances_df],
                record_start_time,
                self.output_dir,
                self.vis_config,
            )

        self.assertTrue(mock_savefig.called)
        fig = plt.gcf()
        ax = fig.gca()

        expected_xlabel = f"Time (averaged over {self.vis_config['chunk_duration_seconds']}s intervals)"
        self.assertEqual(ax.get_xlabel(), expected_xlabel)
        self.assertEqual(ax.get_ylabel(), "Frequency [Hz]")

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        self.assertAlmostEqual(xlim[0], 0.0, delta=0.1)
        self.assertAlmostEqual(xlim[1], duration, delta=duration * 0.05)
        nyquist = sample_rate / 2.0
        self.assertAlmostEqual(ylim[0], 0.0, delta=10)
        self.assertAlmostEqual(
            ylim[1], nyquist, delta=nyquist * 0.05
        )  # Relaxed delta slightly
        self.assertGreaterEqual(ylim[1], nyquist * 0.5)
        plt.close(fig)

    # Patch order: savefig -> args: mock_savefig
    @patch("matplotlib.pyplot.savefig")
    def test_plot_mother_source_spectrogram_tick_labels(self, mock_savefig):
        wav_file_path = os.path.join(self.output_dir, "test_tick_labels.wav")
        sample_rate = 44100
        duration = 5.0
        t_wav = np.linspace(0, duration, int(sample_rate * duration), False)
        tone = np.sin(2 * np.pi * 440 * t_wav)
        sf.write(wav_file_path, tone, sample_rate)
        distances_df = pd.DataFrame(
            {"mmsi": [1], "min_distance_time": [pd.Timestamp("2024-03-20 10:00:02")]}
        )
        record_start_time = pd.Timestamp("2024-03-20 10:00:00")
        file_start_time = record_start_time
        time_format = "%Y-%m-%d %H:%M:%S"
        captured_ticks = []
        captured_labels = []

        def capture_state(*args, **kwargs):
            ax = plt.gca()
            plt.gcf().canvas.draw()  # Finalize layout
            captured_ticks.extend(ax.get_xticks())
            captured_labels.extend([label.get_text() for label in ax.get_xticklabels()])

        mock_savefig.side_effect = capture_state

        # Patch order: SoundFile, read -> args: mock_read, mock_soundfile
        with patch("soundfile.SoundFile") as mock_soundfile, patch(
            "soundfile.read", return_value=(tone, sample_rate)
        ):

            mock_sf_instance = MagicMock()
            mock_sf_instance.__enter__.return_value.samplerate = sample_rate
            mock_sf_instance.__enter__.return_value.__len__.return_value = int(
                duration * sample_rate
            )
            mock_soundfile.return_value = mock_sf_instance

            plot_mother_source_spectrogram(
                self.output_dir,
                [distances_df],
                record_start_time,
                self.output_dir,
                self.vis_config,
            )

        self.assertTrue(mock_savefig.called)
        self.assertGreater(len(captured_ticks), 0, "No ticks captured")
        if len(captured_ticks) > 0:
            self.assertEqual(len(captured_ticks), len(captured_labels))
            first_tick_val = captured_ticks[0]
            first_label = captured_labels[0]
            exp_first_dt = file_start_time + pd.Timedelta(seconds=first_tick_val)
            exp_first_str = exp_first_dt.strftime(time_format)
            self.assertEqual(
                first_label,
                exp_first_str,
                f"First Label: Got '{first_label}', Exp '{exp_first_str}'",
            )

            last_tick_val = captured_ticks[-1]
            last_label = captured_labels[-1]
            exp_last_dt = file_start_time + pd.Timedelta(seconds=last_tick_val)
            exp_last_str = exp_last_dt.strftime(time_format)
            self.assertEqual(
                last_label,
                exp_last_str,
                f"Last Label: Got '{last_label}', Exp '{exp_last_str}'",
            )

            tick_range = last_tick_val - first_tick_val
            self.assertAlmostEqual(
                tick_range, duration, delta=duration * 0.15, msg="Tick range mismatch"
            )
        plt.close(plt.gcf())

    # Patch order: savefig, listdir, read, soundfile
    @patch("soundfile.SoundFile")
    @patch("soundfile.read")
    @patch("os.listdir")
    @patch("matplotlib.pyplot.savefig")
    def test_plot_mother_source_spectrogram_realistic_duration_ticks(
        self, mock_savefig, mock_listdir, mock_read, mock_soundfile
    ):
        test_wav_filename = "date_crossing.wav"
        record_start_time = pd.Timestamp("2024-03-19 23:55:00")
        duration_minutes = 15
        duration_seconds = duration_minutes * 60
        sample_rate = 44100
        time_format = "%Y-%m-%d %H:%M:%S"

        mock_listdir.return_value = [test_wav_filename]

        mock_sf_instance = MagicMock()
        mock_sf_instance.__enter__.return_value.samplerate = sample_rate
        mock_sf_instance.__enter__.return_value.__len__.return_value = int(
            duration_seconds * sample_rate
        )
        mock_soundfile.return_value = mock_sf_instance

        # Mock read: return data for one chunk calculation
        read_chunk_len = int(self.vis_config["chunk_duration_seconds"] * sample_rate)
        nperseg = self.vis_config.get("spectrogram_nperseg", 4096)
        mock_read_data = np.random.rand(
            max(read_chunk_len, nperseg)
        )  # Ensure enough for one spec calc

        # Make read callable to simulate reading different chunks (or just return same data for simplicity)
        def mock_read_side_effect(*args, **kwargs):
            # Return the same chunk data regardless of seek/read size for simplicity
            return (mock_read_data, sample_rate)

        mock_read.side_effect = mock_read_side_effect

        captured_ticks = []
        captured_labels = []

        def capture_state(*args, **kwargs):
            ax = plt.gca()
            plt.gcf().canvas.draw()
            captured_ticks.extend(ax.get_xticks())
            captured_labels.extend([label.get_text() for label in ax.get_xticklabels()])

        mock_savefig.side_effect = capture_state

        distances_df = pd.DataFrame(
            {
                "mmsi": [1],
                "min_distance_time": [record_start_time + pd.Timedelta(minutes=5)],
            }
        )
        plot_mother_source_spectrogram(
            self.output_dir,
            [distances_df],
            record_start_time,
            self.output_dir,
            self.vis_config,
        )

        self.assertTrue(mock_savefig.called)
        self.assertGreater(len(captured_ticks), 1)
        if len(captured_ticks) > 0:
            self.assertEqual(len(captured_ticks), len(captured_labels))
            first_tick_val = captured_ticks[0]
            first_label = captured_labels[0]
            exp_first_dt = record_start_time + pd.Timedelta(seconds=first_tick_val)
            exp_first_str = exp_first_dt.strftime(time_format)
            self.assertEqual(first_label, exp_first_str)
            self.assertAlmostEqual(first_tick_val, 0.0, delta=duration_seconds * 0.1)

            last_tick_val = captured_ticks[-1]
            last_label = captured_labels[-1]
            exp_last_dt = record_start_time + pd.Timedelta(seconds=last_tick_val)
            exp_last_str = exp_last_dt.strftime(time_format)
            self.assertAlmostEqual(
                last_tick_val, duration_seconds, delta=duration_seconds * 0.1
            )
            try:
                time_diff = abs(
                    (exp_last_dt - pd.Timestamp(last_label)).total_seconds()
                )
                self.assertLess(
                    time_diff,
                    self.vis_config["chunk_duration_seconds"],
                    "Last label time difference too large",
                )
            except ValueError:
                self.fail(f"Could not parse last label '{last_label}' as timestamp")

            tick_range = last_tick_val - first_tick_val
            self.assertAlmostEqual(
                tick_range, duration_seconds, delta=duration_seconds * 0.15
            )

            exp_start_date = record_start_time.strftime("%Y-%m-%d")
            exp_end_date = (
                record_start_time + pd.Timedelta(seconds=duration_seconds)
            ).strftime("%Y-%m-%d")
            label_dates = set(
                label.split(" ")[0] for label in captured_labels if " " in label
            )
            if exp_start_date != exp_end_date:
                self.assertIn(exp_start_date, label_dates)
                self.assertIn(exp_end_date, label_dates)
            else:
                self.assertTrue(all(d == exp_start_date for d in label_dates))
        plt.close(plt.gcf())


if __name__ == "__main__":
    unittest.main()
