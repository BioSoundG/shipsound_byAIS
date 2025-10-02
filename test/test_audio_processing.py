import os
import sys
import unittest
import pandas as pd
import numpy as np
import soundfile as sf
import shutil
import datetime
import tomli
import tempfile
from wav_index import WavFileIndex

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from audio_processing import cut_wav_file, cut_wav_and_make_metadata


class TestAudioProcessing(unittest.TestCase):
    def setUp(self):
        """テスト用の一時ディレクトリとWAVファイルを作成"""
        self.test_dir = tempfile.mkdtemp()
        self.wav_dir = os.path.join(self.test_dir, "wav")
        os.makedirs(self.wav_dir, exist_ok=True)

        # テスト用のWAVファイルを作成
        self.sample_rate = 44100
        self.duration = 10  # 10秒
        self.wav_files = []

        # 3つの連続したWAVファイルを作成
        for i in range(3):
            # 10秒のサイン波を生成
            t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
            data = np.sin(2 * np.pi * 440 * t) * 32767  # 440Hzのサイン波
            data = data.astype(np.int16)

            wav_path = os.path.join(self.wav_dir, f"test_{i}.wav")
            sf.write(wav_path, data, self.sample_rate)
            self.wav_files.append(wav_path)

    def tearDown(self):
        """テスト用の一時ディレクトリを削除"""
        shutil.rmtree(self.test_dir)

    def test_wav_index_creation(self):
        """WavFileIndexの作成と基本的な機能をテスト"""
        start_time = pd.Timestamp("2024-03-19 06:53:00")
        wav_index = WavFileIndex(self.wav_files, start_time)

        # 時間範囲の確認
        self.assertEqual(len(wav_index.time_ranges), 3)
        self.assertEqual(wav_index.time_ranges[0]["start"], start_time)
        self.assertEqual(
            wav_index.time_ranges[0]["end"], start_time + datetime.timedelta(seconds=10)
        )

        # サンプル数の確認
        expected_samples = self.sample_rate * self.duration
        self.assertEqual(wav_index.time_ranges[0]["samples"], expected_samples)

    def test_wav_index_find_wav_index(self):
        """WavFileIndexのfind_wav_indexメソッドをテスト"""
        start_time = pd.Timestamp("2024-03-19 06:53:00")
        wav_index = WavFileIndex(self.wav_files, start_time)

        # 各WAVファイルの時間範囲内の時刻でテスト
        test_times = [
            (start_time + datetime.timedelta(seconds=5), 0),  # 1つ目のファイル
            (start_time + datetime.timedelta(seconds=15), 1),  # 2つ目のファイル
            (start_time + datetime.timedelta(seconds=25), 2),  # 3つ目のファイル
        ]

        for target_time, expected_index in test_times:
            found_index = wav_index.find_wav_index(target_time)
            self.assertEqual(found_index, expected_index)

    def test_cut_wav_and_make_metadata(self):
        """改善されたcut_wav_and_make_metadata関数をテスト"""
        # テスト用のメタデータ
        meta_data = {
            "observation_info": {
                "record_info": {"channel_num": 1},
                "date_info": {"start_date": "2024-03-19T06:53:00"},
            },
            "source_info": {},
        }

        # テスト用の距離データ
        start_time = pd.Timestamp("2024-03-19 06:53:00")
        # min_distance_timeを7秒後、cut_margin_minutesを0.1（6秒）にして必ず範囲内に
        distances = pd.DataFrame(
            {
                "mmsi": [123456789],
                "vessel_name": ["Test Vessel"],
                "vessel_type": ["Cargo"],
                "min_distance [m]": [100.0],
                "min_distance_time": [start_time + pd.Timedelta(seconds=7)],
                "length": [100],
                "width": [20],
            }
        )

        # 空の距離リスト
        distance_list = pd.DataFrame()

        # 出力ディレクトリ
        output_dir = os.path.join(self.test_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        # 録音位置
        record_pos = (35.0, 139.0)

        # オーディオ設定
        audio_config = {
            "cut_margin_minutes": 0.1,  # 6秒
            "max_cut_distance": 200.0,
            "check_other_vessels": False,
        }

        # 関数を実行
        cut_wav_and_make_metadata(
            self.wav_files,
            meta_data,
            start_time.strftime("%Y-%m-%d %H:%M:%S"),
            distances,
            distance_list,
            output_dir,
            record_pos,
            audio_config,
        )

        # 出力ファイルの確認
        wav_output_dir = os.path.join(output_dir, "wav")
        self.assertTrue(os.path.exists(wav_output_dir))

        # WAVファイルとTOMLファイルが生成されていることを確認
        wav_files = [f for f in os.listdir(wav_output_dir) if f.endswith(".wav")]
        toml_files = [f for f in os.listdir(wav_output_dir) if f.endswith(".toml")]

        self.assertEqual(len(wav_files), 1)
        self.assertEqual(len(toml_files), 1)

        # TOMLファイルの内容を確認
        with open(os.path.join(wav_output_dir, toml_files[0]), "rb") as f:
            toml_data = tomli.load(f)

        # メタデータの内容を確認
        self.assertEqual(toml_data["source_info"]["sound_source"], "Cargo(Test Vessel)")
        self.assertEqual(toml_data["source_info"]["category"], 1)
        self.assertEqual(toml_data["source_info"]["reliability"], 2)
        self.assertIn("Ship distance: 100.00m", toml_data["source_info"]["condition"])


if __name__ == "__main__":
    unittest.main()
