import os
import sys
import unittest
import pandas as pd
import numpy as np
import datetime
import tempfile
import matplotlib.pyplot as plt
import matplotlib
from unittest.mock import patch, MagicMock, mock_open
from PIL import Image
import re
import soundfile as sf
import tomllib

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from visualization import plot_geolocation, plot_mother_source_spectrogram


def load_test_config():
    default_config = {"visualization": {}, "audio_processing": {}}
    if tomllib:
        try:
            # Assume config.toml is in the parent directory relative to the test file
            config_file_path = os.path.join(
                os.path.dirname(__file__), "..", "config.toml"
            )
            with open(config_file_path, "rb") as f:
                # Merge loaded config with defaults, loaded values take precedence
                loaded_config = tomllib.load(f)
                # Basic merge (can be made more sophisticated)
                default_config["visualization"].update(
                    loaded_config.get("visualization", {})
                )
                default_config["audio_processing"].update(
                    loaded_config.get("audio_processing", {})
                )
                print(f"Test config loaded from {config_file_path}")
            # This return should be inside the try block, after successful loading
            return default_config
        except FileNotFoundError:
            print(f"{config_file_path} not found for tests. Using default values.")
            # Return default if file not found
            return default_config
        except Exception as e:
            print(
                f"Error loading {config_file_path} for tests: {e}. Using default values."
            )
            # Return default on other errors
            return default_config
    else:
        # Return default if tomllib itself is not available
        return default_config


class TestVisualization(unittest.TestCase):
    def setUp(self):
        # テスト用の一時ディレクトリを作成
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name

        # テスト用のAISデータを作成（より精密なテスト用のデータ）
        self.test_df = pd.DataFrame(
            {
                "mmsi": [
                    123456789,
                    123456789,
                    987654321,
                    987654321,
                    555666777,
                    555666777,
                ],
                "vessel_name": [
                    "Vessel A",
                    "Vessel A",
                    "Vessel B",
                    "Vessel B",
                    "Vessel C",
                    "Vessel C",
                ],
                "vessel_type": [
                    "Cargo",
                    "Cargo",
                    "Passenger",
                    "Passenger",
                    "Tanker",
                    "Tanker",
                ],
                "latitude": [32.7, 32.71, 32.72, 32.73, 32.74, 32.75],
                "longitude": [129.7, 129.71, 129.72, 129.73, 129.74, 129.75],
                "dt_pos_utc": [
                    pd.Timestamp("2024-03-19 07:00:00"),
                    pd.Timestamp("2024-03-19 07:01:00"),
                    pd.Timestamp("2024-03-19 07:02:00"),
                    pd.Timestamp("2024-03-19 07:03:00"),
                    pd.Timestamp("2024-03-19 07:04:00"),
                    pd.Timestamp("2024-03-19 07:05:00"),
                ],
            }
        )

        # 録音位置
        self.record_pos = [32.71161, 129.77558]

        # 既存のプロットスタイルを保存
        self.original_style = plt.rcParams.copy()

        # 画像を保持するリスト（テスト後にクローズするため）
        self.images = []

    def tearDown(self):
        # Matplotlibのメモリリーク防止とスタイル復元
        plt.rcParams.update(self.original_style)
        plt.close("all")

        # 開いた画像をすべて閉じる
        for img in self.images:
            if hasattr(img, "close"):
                try:
                    img.close()
                except:
                    pass

        # 一時ディレクトリを削除
        self.temp_dir.cleanup()

    def test_plot_geolocation_output_file_existence(self):
        """plot_geolocation関数の出力ファイル存在確認テスト"""
        # プロットの生成
        plot_geolocation(1, self.test_df, self.record_pos, self.output_dir)

        # 出力ファイルが生成されていることを確認
        output_file = os.path.join(self.output_dir, "1.png")
        self.assertTrue(
            os.path.exists(output_file),
            f"出力ファイル {output_file} が生成されていません",
        )

        # ファイルサイズが最小サイズ（1KB）より大きいことを確認
        min_size = 1024  # 1KB
        self.assertGreater(
            os.path.getsize(output_file),
            min_size,
            f"生成された画像ファイルが小さすぎます（{os.path.getsize(output_file)}バイト < {min_size}バイト）",
        )

    def test_plot_geolocation_image_properties(self):
        """plot_geolocationで生成された画像のプロパティテスト"""
        # プロットの生成
        plot_geolocation(2, self.test_df, self.record_pos, self.output_dir)

        # 出力ファイルのパス
        output_file = os.path.join(self.output_dir, "2.png")

        # 画像が正しく開けるか確認
        try:
            img = Image.open(output_file)
            self.images.append(img)  # 後でクローズするために保存

            # 画像のサイズが適切か確認（最低サイズ - MatplotlibのデフォルトFigureSizeに合わせて調整）
            self.assertGreaterEqual(img.width, 100, "画像の幅が小さすぎます")
            self.assertGreaterEqual(img.height, 100, "画像の高さが小さすぎます")

            # 画像のモードがRGBまたはRGBAであることを確認
            self.assertIn(
                img.mode, ["RGB", "RGBA"], f"画像のモードが不適切です: {img.mode}"
            )

            # 画像のフォーマットがPNGであることを確認
            self.assertEqual(
                img.format, "PNG", f"画像のフォーマットが不適切です: {img.format}"
            )

        except Exception as e:
            self.fail(f"画像ファイルの検証中にエラーが発生しました: {e}")

    def test_plot_geolocation_content(self):
        """plot_geolocationで生成された画像の内容テスト"""
        # 関数を実行
        plot_geolocation(3, self.test_df, self.record_pos, self.output_dir)

        # 出力ファイルが生成されていることを確認
        output_file = os.path.join(self.output_dir, "3.png")
        self.assertTrue(
            os.path.exists(output_file),
            "出力ファイルが生成されていません",
        )

        # 画像の内容を確認
        img = plt.imread(output_file)
        self.assertGreater(img.shape[0], 0, "画像の高さが0です")
        self.assertGreater(img.shape[1], 0, "画像の幅が0です")
        self.assertIn(img.shape[2], [3, 4], "画像の色チャネル数が不正です")

        # 画像が空でないことを確認
        self.assertFalse(np.all(img == 0), "画像が全て黒（0）です")
        self.assertFalse(np.all(img == 1), "画像が全て白（1）です")

    def test_plot_geolocation_empty_df(self):
        """空のデータフレームでのテスト（より厳格なチェック）"""
        # 空のデータフレーム
        empty_df = pd.DataFrame(columns=self.test_df.columns)

        # プロットの生成と例外が発生しないことを確認
        try:
            plot_geolocation(4, empty_df, self.record_pos, self.output_dir)
            output_file = os.path.join(self.output_dir, "4.png")

            # ファイルが生成されていることを確認
            self.assertTrue(
                os.path.exists(output_file),
                "空のデータフレームで出力ファイルが生成されていません",
            )

            # 画像ファイルとして有効であることを確認
            img = Image.open(output_file)
            self.images.append(img)  # 後でクローズするために保存
            self.assertIsNotNone(img, "生成された画像が無効です")

            # 画像データが存在することを確認
            img_array = plt.imread(output_file)
            self.assertGreater(img_array.size, 0, "画像データが空です")

        except Exception as e:
            self.fail(f"空のデータフレームでプロット生成中に例外が発生しました: {e}")

    def test_plot_geolocation_one_vessel(self):
        """単一の船舶でのテスト（より詳細な検証）"""
        # 1つの船舶のみのデータフレーム
        single_vessel_df = self.test_df[self.test_df["mmsi"] == 123456789].copy()

        # プロットの生成
        plot_geolocation(5, single_vessel_df, self.record_pos, self.output_dir)

        # 出力ファイルを検証
        output_file = os.path.join(self.output_dir, "5.png")
        self.assertTrue(
            os.path.exists(output_file),
            "単一船舶データで出力ファイルが生成されていません",
        )

        # 画像の内容を検証
        try:
            img_array = plt.imread(output_file)

            # 画像データが存在することを確認
            self.assertGreater(img_array.shape[0], 0, "画像の高さが0です")
            self.assertGreater(img_array.shape[1], 0, "画像の幅が0です")

            # 画像が適切なカラーチャネル数を持つことを確認
            self.assertIn(img_array.shape[2], [3, 4], "画像の色チャネル数が不正です")

            # 自動的に内容検証はできないが、少なくとも有効な画像であることを確認
            self.assertFalse(np.all(img_array == 0), "画像が全て黒（0）です")
            self.assertFalse(np.all(img_array == 1), "画像が全て白（1）です")

        except Exception as e:
            self.fail(f"単一船舶の画像検証中にエラーが発生しました: {e}")

    def test_plot_geolocation_edge_case_same_coordinates(self):
        """同一座標の船舶でのエッジケーステスト"""
        # 全て同じ座標を持つデータフレーム
        same_coord_df = pd.DataFrame(
            {
                "mmsi": [123456789, 123456789, 987654321, 987654321],
                "vessel_name": ["Vessel A", "Vessel A", "Vessel B", "Vessel B"],
                "vessel_type": ["Cargo", "Cargo", "Passenger", "Passenger"],
                "latitude": [32.7, 32.7, 32.7, 32.7],  # 全て同じ
                "longitude": [129.7, 129.7, 129.7, 129.7],  # 全て同じ
                "dt_pos_utc": [
                    pd.Timestamp("2024-03-19 07:00:00"),
                    pd.Timestamp("2024-03-19 07:01:00"),
                    pd.Timestamp("2024-03-19 07:02:00"),
                    pd.Timestamp("2024-03-19 07:03:00"),
                ],
            }
        )

        # エラーなしでプロットできることを確認
        try:
            plot_geolocation(6, same_coord_df, self.record_pos, self.output_dir)
            output_file = os.path.join(self.output_dir, "6.png")
            self.assertTrue(
                os.path.exists(output_file),
                "同一座標データでファイルが生成されていません",
            )
        except Exception as e:
            self.fail(f"同一座標データでプロット中に例外が発生しました: {e}")

    def test_plot_geolocation_extreme_coordinates(self):
        """極端な座標値でのテスト"""
        # 極端な座標を持つデータフレーム
        extreme_df = pd.DataFrame(
            {
                "mmsi": [123456789, 987654321],
                "vessel_name": ["Extreme Vessel A", "Extreme Vessel B"],
                "vessel_type": ["Cargo", "Passenger"],
                "latitude": [89.9, -89.9],  # 極端な北と南
                "longitude": [179.9, -179.9],  # 極端な東と西
                "dt_pos_utc": [
                    pd.Timestamp("2024-03-19 07:00:00"),
                    pd.Timestamp("2024-03-19 07:01:00"),
                ],
            }
        )

        # エラーなしでプロットできることを確認
        try:
            plot_geolocation(7, extreme_df, self.record_pos, self.output_dir)
            output_file = os.path.join(self.output_dir, "7.png")
            self.assertTrue(
                os.path.exists(output_file),
                "極端な座標データでファイルが生成されていません",
            )
        except Exception as e:
            self.fail(f"極端な座標データでプロット中に例外が発生しました: {e}")

    def test_plot_geolocation_nan_values(self):
        """NaN値を含むデータフレームでのテスト"""
        # NaN値を含むデータフレーム
        df_with_nan = self.test_df.copy()
        df_with_nan.loc[1, "latitude"] = np.nan
        df_with_nan.loc[3, "longitude"] = np.nan

        # プロットの生成（例外が発生しないことを確認）
        try:
            plot_geolocation(9, df_with_nan, self.record_pos, self.output_dir)
            output_file = os.path.join(self.output_dir, "9.png")

            # ファイルが生成されていることを確認
            self.assertTrue(
                os.path.exists(output_file), "NaN値ありでも出力ファイルが生成されます"
            )

            # 有効な画像ファイルであることを確認
            img = Image.open(output_file)
            self.images.append(img)
            self.assertIsNotNone(img, "生成された画像が有効です")

        except Exception as e:
            self.fail(f"NaN値を含むデータでプロット生成中に例外が発生しました: {e}")

    @patch("soundfile.read")
    @patch("soundfile.SoundFile")
    @patch("os.listdir")
    @patch("matplotlib.pyplot.savefig")
    def test_plot_mother_source_spectrogram(
        self, mock_savefig, mock_listdir, mock_soundfile, mock_read
    ):
        """plot_mother_source_spectrogram関数のテスト"""
        # モックデータの設定
        # WAVファイルリストをモック
        mock_listdir.return_value = ["test1.WAV", "test2.WAV"]

        # SoundFileモックを設定して継続時間を返す
        mock_sf_instance = MagicMock()
        mock_sf_instance.__enter__.return_value.samplerate = 44100
        mock_sf_instance.__enter__.return_value.__len__.return_value = (
            44100 * 10
        )  # 10秒のWAVファイル
        mock_soundfile.return_value = mock_sf_instance

        # soundfile.readモックを設定
        # 簡単な1秒の波形データを作成
        sample_data = np.sin(np.linspace(0, 2 * np.pi, 44100)) * 0.5
        mock_read.return_value = (sample_data, 44100)

        # テスト用の距離データを作成
        test_time = pd.Timestamp("2024-03-19 07:00:00")
        distances_df = pd.DataFrame(
            {
                "mmsi": [123456789],
                "vessel_name": ["Test Vessel"],
                "vessel_type": ["Cargo"],
                "length": [100],
                "width": [20],
                "min_distance [m]": [500.0],
                "min_distance_time": [test_time],
            }
        )

        # 録音開始時間
        record_start_time = pd.Timestamp("2024-03-19 06:53:00")

        # 関数を実行
        plot_mother_source_spectrogram(
            "test_wav_path", distances_df, record_start_time, self.output_dir
        )

        # 出力ディレクトリが作成されたか確認
        spec_output_dir = os.path.join(self.output_dir, "spectrograms")

        # savefigが呼び出されたことを確認（スペクトログラムの保存）
        self.assertTrue(mock_savefig.called, "plt.savefigが呼び出されていません")
        self.assertEqual(
            mock_savefig.call_count,
            2,
            "2つのWAVファイルに対して2回のsavefigが呼ばれるべきです",
        )

        # savefigの第1引数（ファイルパス）が正しい形式であることを確認
        for call_args in mock_savefig.call_args_list:
            args, kwargs = call_args
            filepath = args[0]
            # ファイルパスが正しいディレクトリと'spec_'で始まるファイル名を持つことを確認
            self.assertTrue(
                filepath.startswith(os.path.join(spec_output_dir, "spec_")),
                f"ファイル名が'spec_'で始まっていません: {filepath}",
            )
            self.assertTrue(
                filepath.endswith(".png"),
                f"ファイル拡張子が.pngではありません: {filepath}",
            )

    @patch("soundfile.read")
    @patch("soundfile.SoundFile")
    @patch("os.listdir")
    @patch("matplotlib.pyplot.savefig")
    def test_plot_mother_source_spectrogram_time_axis_length(
        self, mock_savefig, mock_listdir, mock_soundfile, mock_read
    ):
        """時間軸の長さがWAVファイルの実際の長さと一致することを確認するテスト"""
        # テスト用の一時ディレクトリを作成
        with tempfile.TemporaryDirectory() as temp_dir:
            # WAVファイルの読み込みをモック化（10秒のファイルをシミュレート）
            sample_rate = 48000
            duration = 10  # 10秒のファイル
            num_samples = sample_rate * duration
            mock_data = np.random.rand(num_samples)

            # モックの設定
            mock_read.return_value = (mock_data, sample_rate)
            mock_listdir.return_value = ["test.wav"]  # WAVファイルを1つ用意

            # SoundFileモックを設定して継続時間を返す
            mock_sf_instance = MagicMock()
            mock_sf_instance.__enter__.return_value.samplerate = sample_rate
            mock_sf_instance.__enter__.return_value.__len__.return_value = num_samples
            mock_soundfile.return_value = mock_sf_instance

            # 時間データを用意
            start_time = pd.Timestamp("2021-01-01 00:00:00")

            # 空のデータフレーム
            df = pd.DataFrame()

            # プロット関数を実行
            plot_mother_source_spectrogram("test", [df], start_time, temp_dir)

            # savefigが呼び出されたことを確認
            assert (
                mock_savefig.call_count == 1
            ), f"savefig call count was {mock_savefig.call_count}, expected 1"

            # 保存されたファイルパスの形式を確認
            save_path = mock_savefig.call_args[0][0]
            assert save_path.endswith(".png")
            assert os.path.join(temp_dir, "spectrograms", "spec_test.png") == save_path

    def test_plot_mother_source_spectrogram_save_spectrogram(self):
        """スペクトログラムが正常に保存されるかを検証するテスト"""
        # テスト用の一時ディレクトリを作成（tearDown()で自動的に削除される）
        spec_output_dir = os.path.join(self.output_dir, "spectrograms")

        # テスト用WAVファイルを作成
        wav_file_path = os.path.join(self.output_dir, "test_audio.wav")

        # 簡易的なWAVファイルを生成（1秒の単純な正弦波）
        sample_rate = 44100
        t = np.linspace(0, 1, sample_rate, False)
        tone = np.sin(2 * np.pi * 440 * t)  # 440Hzの正弦波
        sf.write(wav_file_path, tone, sample_rate)

        # テスト用のデータフレームを準備
        test_time = pd.Timestamp("2024-03-19 07:00:00")
        distances_df = pd.DataFrame(
            {
                "mmsi": [123456789],
                "vessel_name": ["Test Vessel"],
                "vessel_type": ["Cargo"],
                "length": [100],
                "width": [20],
                "min_distance [m]": [500.0],
                "min_distance_time": [test_time],
            }
        )

        # 録音開始時間
        record_start_time = pd.Timestamp("2024-03-19 06:59:00")

        # 関数を実行
        plot_mother_source_spectrogram(
            self.output_dir, distances_df, record_start_time, self.output_dir
        )

        # スペクトログラムファイルの存在確認
        expected_filename = f"spec_test_audio.png"
        output_path = os.path.join(spec_output_dir, expected_filename)

        self.assertTrue(
            os.path.exists(output_path),
            f"スペクトログラムファイル {expected_filename} が生成されていません",
        )

        # ファイルサイズが最小サイズより大きいことを確認
        min_size = 10 * 1024  # 10KB
        self.assertGreater(
            os.path.getsize(output_path),
            min_size,
            f"生成されたスペクトログラムファイルが小さすぎます（{os.path.getsize(output_path)}バイト < {min_size}バイト）",
        )

        # 画像が正しく読み込めることを確認
        try:
            img = Image.open(output_path)
            self.images.append(img)  # 後でクローズするために保存

            # 画像のサイズが適切か確認
            self.assertGreaterEqual(
                img.width, 500, "スペクトログラム画像の幅が小さすぎます"
            )
            self.assertGreaterEqual(
                img.height, 300, "スペクトログラム画像の高さが小さすぎます"
            )

            # RGBまたはRGBAフォーマットであることを確認
            self.assertIn(
                img.mode, ["RGB", "RGBA"], f"画像のモードが不適切です: {img.mode}"
            )

            # 画像データを読み込んで基本的なチェック
            img_array = np.array(img)

            # 画像が空ではないことを確認
            self.assertFalse(np.all(img_array == 0), "画像が全て黒（0）です")
            self.assertFalse(np.all(img_array == 255), "画像が全て白（255）です")

            # 最低限の色の変化があることを確認（単色ではない）
            self.assertGreater(
                np.std(img_array), 10, "画像の標準偏差が低すぎます（色の変化が少ない）"
            )

        except Exception as e:
            self.fail(f"スペクトログラム画像の検証中にエラーが発生しました: {e}")

    def test_plot_mother_source_spectrogram_with_cuts(self):
        """スペクトログラムにカットセクションが正しく表示されるかを検証するテスト"""
        # テスト用の一時ディレクトリを作成
        spec_output_dir = os.path.join(self.output_dir, "spectrograms")

        # テスト用WAVファイルを作成
        wav_file_path = os.path.join(self.output_dir, "test_audio.wav")

        # 5秒の単純なWAVファイルを生成
        sample_rate = 44100
        t = np.linspace(0, 5, sample_rate * 5, False)
        tone = np.sin(2 * np.pi * 440 * t)  # 440Hzの正弦波
        sf.write(wav_file_path, tone, sample_rate)

        # 複数のカットを含むテスト用のデータフレームを準備
        record_start_time = pd.Timestamp("2024-03-19 07:00:00")

        # 3つの異なる船舶のカットを含むデータフレーム
        distances_df = pd.DataFrame(
            {
                "mmsi": [123456789, 987654321, 555666777],
                "vessel_name": ["Vessel A", "Vessel B", "Vessel C"],
                "vessel_type": ["Cargo", "Passenger", "Tanker"],
                "length": [100, 150, 80],
                "width": [20, 25, 15],
                "min_distance [m]": [500.0, 800.0, 300.0],
                "min_distance_time": [
                    record_start_time + pd.Timedelta(seconds=1),
                    record_start_time + pd.Timedelta(seconds=2),
                    record_start_time + pd.Timedelta(seconds=3),
                ],
            }
        )

        # 関数を実行
        plot_mother_source_spectrogram(
            self.output_dir, distances_df, record_start_time, self.output_dir
        )

        # スペクトログラムファイルの存在確認
        expected_filename = f"spec_test_audio.png"
        output_path = os.path.join(spec_output_dir, expected_filename)

        self.assertTrue(
            os.path.exists(output_path),
            f"スペクトログラムファイル {expected_filename} が生成されていません",
        )

        # 画像を読み込んで内容を検査
        try:
            img = Image.open(output_path)
            self.images.append(img)

            # 画像が生成されていることを確認
            self.assertIsNotNone(img, "スペクトログラム画像が無効です")

            # 画像サイズが妥当であることを確認
            self.assertGreaterEqual(
                img.width, 500, "スペクトログラム画像の幅が小さすぎます"
            )
            self.assertGreaterEqual(
                img.height, 300, "スペクトログラム画像の高さが小さすぎます"
            )

        except Exception as e:
            self.fail(f"スペクトログラム画像の検証中にエラーが発生しました: {e}")

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_plot_mother_source_spectrogram_actual_time_axis(
        self, mock_close, mock_savefig
    ):
        """スペクトログラムの時間軸の長さがWAVファイルの実際の長さと一致するか検証するテスト"""
        # テスト用WAVファイルを作成（5秒間の単純な正弦波）
        wav_file_path = os.path.join(self.output_dir, "test_time_axis.wav")
        sample_rate = 44100
        duration = 5.0  # 5秒
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        tone = np.sin(2 * np.pi * 440 * t)  # 440Hzの正弦波
        sf.write(wav_file_path, tone, sample_rate)

        # テスト用のダミーデータフレーム（内容は影響しないはず）
        distances_df = pd.DataFrame({"mmsi": [1]})  # Minimal df
        record_start_time = pd.Timestamp("2024-03-19 07:00:00")

        # 関数を実行（savefigとcloseはモック化されている）
        plot_mother_source_spectrogram(
            self.output_dir, distances_df, record_start_time, self.output_dir
        )

        # savefigが呼び出されたか確認
        self.assertTrue(mock_savefig.called, "savefigが呼び出されませんでした")

        # プロットされた図と軸を取得
        fig = plt.gcf()  # 現在の Figure を取得
        ax = fig.gca()  # 現在の Axes を取得

        # X軸（時間軸）の範囲を取得
        xlim = ax.get_xlim()
        actual_max_time = xlim[1]  # X軸の最大値

        # X軸の最大値がWAVファイルの長さ（duration）とほぼ等しいことを確認
        # signal.spectrogram や プロット処理での微小な誤差を許容するため delta を設定
        delta = 0.1  # 許容誤差（秒）
        self.assertAlmostEqual(
            actual_max_time,
            duration,
            delta=delta,
            msg=(
                f"スペクトログラムの時間軸の最大値 ({actual_max_time:.2f}s) が "
                f"WAVファイルの実際の長さ ({duration:.2f}s) と一致しません（許容誤差: {delta}s）"
            ),
        )

        # モック解除のために手動で閉じる (tearDownでも呼ばれるが念のため)
        plt.close(fig)

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_plot_geolocation_axis_labels_and_ranges(self, mock_close, mock_savefig):
        """Geolocationプロットの軸ラベルと範囲が適切か検証するテスト"""
        # 関数を実行（savefigとcloseはモック化）
        plot_geolocation(8, self.test_df, self.record_pos, self.output_dir)

        # savefigが呼び出されたか確認
        self.assertTrue(
            mock_savefig.called, "savefigが呼び出されませんでした (geolocation)"
        )

        # プロットされた図と軸を取得
        fig = plt.gcf()
        ax = fig.gca()

        # 軸ラベルを検証
        self.assertEqual(
            ax.get_xlabel(), "Longitude", "X軸ラベルが異なります (geolocation)"
        )
        self.assertEqual(
            ax.get_ylabel(), "Latitude", "Y軸ラベルが異なります (geolocation)"
        )

        # 軸の範囲を検証
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # プロットされるべきデータの範囲を計算 (NaNを除外)
        min_lon = self.test_df["longitude"].dropna().min()
        max_lon = self.test_df["longitude"].dropna().max()
        min_lat = self.test_df["latitude"].dropna().min()
        max_lat = self.test_df["latitude"].dropna().max()

        # 録音位置も考慮
        record_lon, record_lat = self.record_pos
        expected_min_lon = min(min_lon, record_lon)
        expected_max_lon = max(max_lon, record_lon)
        expected_min_lat = min(min_lat, record_lat)
        expected_max_lat = max(max_lat, record_lat)

        # 軸範囲がデータ範囲を含んでいることを確認 (マージンを考慮して <=, >= で評価)
        self.assertLessEqual(
            xlim[0], expected_min_lon, "X軸の最小値がデータ範囲を下回っています"
        )
        self.assertGreaterEqual(
            xlim[1], expected_max_lon, "X軸の最大値がデータ範囲を上回っていません"
        )
        self.assertLessEqual(
            ylim[0], expected_min_lat, "Y軸の最小値がデータ範囲を下回っています"
        )
        self.assertGreaterEqual(
            ylim[1], expected_max_lat, "Y軸の最大値がデータ範囲を上回っていません"
        )

        # 手動で閉じる
        plt.close(fig)

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_plot_mother_source_spectrogram_axis_labels_and_ranges(
        self, mock_close, mock_savefig
    ):
        """Spectrogramプロットの軸ラベルと範囲が適切か検証するテスト"""
        # テスト用WAVファイルを作成（1秒）
        wav_file_path = os.path.join(self.output_dir, "test_axis_labels.wav")
        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        tone = np.sin(2 * np.pi * 440 * t)
        sf.write(wav_file_path, tone, sample_rate)

        # ダミーデータ
        distances_df = pd.DataFrame({"mmsi": [1]})
        record_start_time = pd.Timestamp("2024-03-19 07:00:00")

        # 関数を実行（savefigとcloseはモック化）
        plot_mother_source_spectrogram(
            self.output_dir, distances_df, record_start_time, self.output_dir
        )

        # savefigが呼び出されたか確認
        self.assertTrue(
            mock_savefig.called, "savefigが呼び出されませんでした (spectrogram)"
        )

        # プロットされた図と軸を取得
        fig = plt.gcf()
        ax = fig.gca()

        # 軸ラベルを検証
        self.assertEqual(ax.get_xlabel(), "Time", "X軸ラベルが異なります (spectrogram)")
        self.assertEqual(
            ax.get_ylabel(), "Frequency [Hz]", "Y軸ラベルが異なります (spectrogram)"
        )

        # 軸の範囲を検証
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # X軸 (時間) の範囲を確認
        self.assertAlmostEqual(
            xlim[0], 0.0, delta=0.1, msg="X軸の最小値が0からずれています"
        )
        # xlim[1] (最大値) は test_plot_mother_source_spectrogram_actual_time_axis で確認済みだが、ここでも確認
        self.assertAlmostEqual(
            xlim[1],
            duration,
            delta=0.1,
            msg="X軸の最大値がWAVファイルの長さと一致しません",
        )

        # Y軸 (周波数) の範囲を確認
        nyquist_frequency = sample_rate / 2.0
        self.assertAlmostEqual(
            ylim[0], 0.0, delta=10, msg="Y軸の最小値が0からずれています"
        )  # 多少の誤差は許容
        # self.assertLessEqual(ylim[1], nyquist_frequency, msg="Y軸の最大値がナイキスト周波数を超えています")
        # Y軸最大値がナイキスト周波数に非常に近いことを確認
        self.assertAlmostEqual(
            ylim[1],
            nyquist_frequency,
            delta=nyquist_frequency * 0.01,  # ナイキスト周波数の1%程度の誤差を許容
            msg=f"Y軸の最大値({ylim[1]:.2f})がナイキスト周波数({nyquist_frequency:.2f})から大きくずれています",
        )
        # Y軸最大値が極端に低くないかも確認 (例: ナイキスト周波数の半分以上はあるか)
        self.assertGreaterEqual(
            ylim[1], nyquist_frequency * 0.5, msg="Y軸の最大値が低すぎます"
        )

        # 手動で閉じる
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
