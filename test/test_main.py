import os
import sys
import unittest
import pandas as pd
import numpy as np
import tempfile
import shutil
import glob
from unittest.mock import patch, MagicMock

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import main  # メインモジュールをインポート


class TestMain(unittest.TestCase):
    def setUp(self):
        # テスト用の一時ディレクトリを作成
        self.temp_dir = tempfile.mkdtemp()

        # テスト用のAISディレクトリを作成
        self.ais_dir = os.path.join(self.temp_dir, "ais_data")
        os.makedirs(self.ais_dir, exist_ok=True)

        # テスト用のWAVディレクトリを作成
        self.wav_dir = os.path.join(self.temp_dir, "wav_data")
        os.makedirs(self.wav_dir, exist_ok=True)

        # テスト用のAISデータを作成
        self.create_test_ais_data()

        # テスト用のWAVファイルを作成
        self.create_test_wav_files()

        # テスト用のTOMLファイルを作成
        self.toml_path = os.path.join(self.temp_dir, "metadata.toml")
        self.create_test_toml_file()

        # テスト用の開始時間
        self.record_start_time = "2024-03-19T06:53:00"

    def tearDown(self):
        # 一時ディレクトリを削除
        shutil.rmtree(self.temp_dir)

    def create_test_ais_data(self):
        """テスト用のAISデータを作成"""
        test_data = pd.DataFrame(
            {
                "mmsi": [123456789, 123456789, 987654321, 987654321],
                "vessel_name": ["Vessel A", "Vessel A", "Vessel B", "Vessel B"],
                "vessel_type": ["Cargo", "Cargo", "Passenger", "Passenger"],
                "length": [100, 100, 150, 150],
                "width": [20, 20, 30, 30],
                "latitude": [32.7, 32.71, 32.72, 32.73],
                "longitude": [129.7, 129.71, 129.72, 129.73],
                "dt_pos_utc": [
                    "2024-03-19 07:00:00",
                    "2024-03-19 07:01:00",
                    "2024-03-19 07:02:00",
                    "2024-03-19 07:03:00",
                ],
            }
        )

        # CSVファイルとして保存
        file_path = os.path.join(self.ais_dir, "test_ais_data.csv")
        test_data.to_csv(file_path, index=False)

        # 複数のCSVファイルをテストするためにコピーを作成
        for i in range(1, 3):
            copy_path = os.path.join(self.ais_dir, f"test_ais_data_{i}.csv")
            test_data.to_csv(copy_path, index=False)

    def create_test_wav_files(self):
        """テスト用のWAVファイル（空ファイル）を作成"""
        for i in range(3):
            with open(os.path.join(self.wav_dir, f"test_{i}.WAV"), "w") as f:
                f.write("dummy wav content")

    def create_test_toml_file(self):
        """テスト用のTOMLファイルを作成"""
        toml_content = """
[observation_info.date_info]
time_zone = "JST"
start_date = 2024-03-19T06:53:00

[observation_info.location_info]
record_env = 1
position = [32.71161, 129.77558]
installation_depth = 100.0
water_depth = 150.0

[observation_info.machine_info]
machine = "hydrophone"
maker = "aqua-sound"
product_name = "AUSOMS-mini"
item_num = "AQM-302"

[observation_info.record_info]
file_format = "WAV"
fs = 44100
bit = 16
channel_num = 1
amp = 10
highpass_filter = 0
lowpass_filter = 0

[source_info]
cut_source_name = "cut.wav"
category = 1
sound_source = "Ship"
reliability = 2
precipitation = 0.0
wind_speed = 0.0
condition = ""
appendix = ""
"""
        with open(self.toml_path, "w") as f:
            f.write(toml_content)

    @patch("main.plot_geolocation")
    @patch("main.cut_wav_and_make_metadata")
    @patch("main.complement_trajectory")
    @patch("main.read_ais")
    def test_main_with_fig_flag_true(
        self,
        mock_read_ais,
        mock_complement_trajectory,
        mock_cut_wav,
        mock_plot_geolocation,
    ):
        """fig_flagがTrueの場合のテスト"""
        # モック関数の設定
        mock_ais_df = pd.DataFrame(
            {
                "mmsi": [123456789],
                "vessel_name": ["Vessel A"],
                "vessel_type": ["Cargo"],
                "length": [100],
                "width": [20],
                "latitude": [32.7],
                "longitude": [129.7],
                "dt_pos_utc": [pd.Timestamp("2024-03-19 07:00:00")],
            }
        )
        mock_read_ais.return_value = mock_ais_df

        mock_comp_df = mock_ais_df.copy()
        mock_complement_trajectory.return_value = mock_comp_df

        mock_distances = [
            {"mmsi": 123456789, "vessel_name": "Vessel A", "length": 100, "width": 20}
        ]
        mock_complement_trajectory.return_value = pd.DataFrame(
            {
                "mmsi": [123456789],
                "vessel_name": ["Vessel A"],
                "vessel_type": ["Cargo"],
                "length": [100],
                "width": [20],
                "latitude": [32.7],
                "longitude": [129.7],
                "dt_pos_utc": [pd.Timestamp("2024-03-19 07:00:00")],
            }
        )

        # 関数実行
        main.main(
            self.ais_dir,
            self.wav_dir,
            self.toml_path,
            self.record_start_time,
            True,  # fig_flag
            False,  # movie_flag
            False,  # csv_flag
        )

        # plot_geolocation関数が呼ばれたことを確認
        self.assertTrue(mock_plot_geolocation.called)

        # 各AISファイルに対して1回ずつ呼ばれたことを確認
        self.assertEqual(
            mock_plot_geolocation.call_count, len(glob.glob(f"{self.ais_dir}/*.csv"))
        )

    @patch("main.cut_wav_and_make_metadata")
    @patch("main.complement_trajectory")
    @patch("main.read_ais")
    @patch("pandas.DataFrame.to_csv")
    def test_main_with_csv_flag_true(
        self, mock_to_csv, mock_read_ais, mock_complement_trajectory, mock_cut_wav
    ):
        """csv_flagがTrueの場合のテスト"""
        # モック関数の設定
        mock_ais_df = pd.DataFrame(
            {
                "mmsi": [123456789],
                "vessel_name": ["Vessel A"],
                "vessel_type": ["Cargo"],
                "length": [100],
                "width": [20],
                "latitude": [32.7],
                "longitude": [129.7],
                "dt_pos_utc": [pd.Timestamp("2024-03-19 07:00:00")],
            }
        )
        mock_read_ais.return_value = mock_ais_df

        mock_comp_df = mock_ais_df.copy()
        mock_complement_trajectory.return_value = mock_comp_df

        # 関数実行
        main.main(
            self.ais_dir,
            self.wav_dir,
            self.toml_path,
            self.record_start_time,
            False,  # fig_flag
            False,  # movie_flag
            True,  # csv_flag
        )

        # DataFrame.to_csvが呼ばれたことを確認
        self.assertTrue(mock_to_csv.called)

        # 各AISファイルに対して1回ずつ呼ばれたことを確認
        self.assertEqual(
            mock_to_csv.call_count, len(glob.glob(f"{self.ais_dir}/*.csv"))
        )

    @patch("main.plot_geolocation")
    @patch("main.cut_wav_and_make_metadata")
    @patch("main.complement_trajectory")
    @patch("main.read_ais")
    @patch("pandas.DataFrame.to_csv")
    def test_main_with_all_flags_true(
        self,
        mock_to_csv,
        mock_read_ais,
        mock_complement_trajectory,
        mock_cut_wav,
        mock_plot_geolocation,
    ):
        """すべてのフラグがTrueの場合のテスト"""
        # モック関数の設定
        mock_ais_df = pd.DataFrame(
            {
                "mmsi": [123456789],
                "vessel_name": ["Vessel A"],
                "vessel_type": ["Cargo"],
                "length": [100],
                "width": [20],
                "latitude": [32.7],
                "longitude": [129.7],
                "dt_pos_utc": [pd.Timestamp("2024-03-19 07:00:00")],
            }
        )
        mock_read_ais.return_value = mock_ais_df

        mock_comp_df = mock_ais_df.copy()
        mock_complement_trajectory.return_value = mock_comp_df

        # 関数実行
        main.main(
            self.ais_dir,
            self.wav_dir,
            self.toml_path,
            self.record_start_time,
            True,  # fig_flag
            True,  # movie_flag (現在は使用されていないが、将来的に実装される可能性がある)
            True,  # csv_flag
        )

        # plot_geolocation関数が呼ばれたことを確認
        self.assertTrue(mock_plot_geolocation.called)

        # DataFrame.to_csvが呼ばれたことを確認
        self.assertTrue(mock_to_csv.called)

        # 各AISファイルに対して処理が行われたことを確認
        expected_call_count = len(glob.glob(f"{self.ais_dir}/*.csv"))
        self.assertEqual(mock_plot_geolocation.call_count, expected_call_count)
        self.assertEqual(mock_to_csv.call_count, expected_call_count)


if __name__ == "__main__":
    unittest.main()
