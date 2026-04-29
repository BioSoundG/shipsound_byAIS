import os
import sys
import unittest
import numpy as np
import tempfile
import soundfile as sf

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from acoustic_features import (
    calculate_third_octave_sel,
    calculate_spectral_centroid,
    calculate_kurtosis_and_crest_factor,
    detect_tonals_and_calculate_level,
    extract_audio_segment,
)


class TestAcousticFeatures(unittest.TestCase):
    def setUp(self):
        """テスト用の音声データを作成"""
        self.sr = 44100  # Sample rate
        self.duration = 2.0  # seconds
        self.n_samples = int(self.sr * self.duration)
        
        # テスト用の正弦波（1000 Hz）
        t = np.linspace(0, self.duration, self.n_samples, endpoint=False)
        self.test_signal = np.sin(2 * np.pi * 1000 * t)
        
        # テスト用のWAVファイルを作成
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_wav_path = os.path.join(self.temp_dir.name, "test_audio.wav")
        sf.write(self.test_wav_path, self.test_signal, self.sr)

    def tearDown(self):
        """一時ディレクトリを削除"""
        self.temp_dir.cleanup()

    def test_calculate_third_octave_sel(self):
        """1/3オクターブバンドSELの計算をテスト"""
        sel_values = calculate_third_octave_sel(self.test_signal, self.sr, center_freqs=[1000])
        
        # 1000 HzのSELが計算されていることを確認
        self.assertIn("1000Hz", sel_values)
        self.assertIsInstance(sel_values["1000Hz"], (int, float))
        self.assertGreater(sel_values["1000Hz"], -np.inf)

    def test_calculate_spectral_centroid(self):
        """スペクトル重心の計算をテスト"""
        centroid = calculate_spectral_centroid(self.test_signal, self.sr, freq_range=(50, 5000))
        
        # スペクトル重心が計算されていることを確認
        self.assertIsInstance(centroid, (int, float))
        self.assertGreater(centroid, 0)
        # 1000 Hz正弦波なので、重心は1000 Hz付近のはず
        self.assertGreater(centroid, 500)
        self.assertLess(centroid, 2000)

    def test_calculate_kurtosis_and_crest_factor(self):
        """クルトシスとクレストファクタの計算をテスト"""
        result = calculate_kurtosis_and_crest_factor(self.test_signal)
        
        # 両方の値が計算されていることを確認
        self.assertIn("kurtosis", result)
        self.assertIn("crest_factor_db", result)
        self.assertIsInstance(result["kurtosis"], (int, float))
        self.assertIsInstance(result["crest_factor_db"], (int, float))
        
        # 正弦波のクルトシスは負の値（正規分布より尖っていない）
        self.assertLess(result["kurtosis"], 0)
        
        # クレストファクタは有限値
        self.assertNotEqual(result["crest_factor_db"], np.inf)

    def test_detect_tonals_and_calculate_level(self):
        """トーナル検出とレベル計算をテスト"""
        result = detect_tonals_and_calculate_level(self.test_signal, self.sr, n_harmonics=5)
        
        # 必要なキーが存在することを確認
        self.assertIn("tonal_level_db", result)
        self.assertIn("num_tonals", result)
        self.assertIn("fundamental_freq", result)
        
        # トーナルが検出されていることを確認
        self.assertGreater(result["num_tonals"], 0)
        self.assertGreater(result["tonal_level_db"], -np.inf)
        
        # 基音周波数が1000 Hz付近であることを確認
        self.assertGreater(result["fundamental_freq"], 800)
        self.assertLess(result["fundamental_freq"], 1200)

    def test_extract_audio_segment(self):
        """音声セグメント抽出をテスト"""
        # 0.5秒から1.0秒間抽出
        audio_data, sr = extract_audio_segment(self.test_wav_path, start_time=0.5, duration=1.0)
        
        # データが抽出されていることを確認
        self.assertGreater(len(audio_data), 0)
        self.assertEqual(sr, self.sr)
        
        # 抽出されたデータの長さが約1秒分であることを確認
        expected_samples = int(self.sr * 1.0)
        self.assertAlmostEqual(len(audio_data), expected_samples, delta=100)

    def test_extract_audio_segment_boundary(self):
        """音声セグメント抽出の境界条件をテスト"""
        # ファイルの範囲外を指定した場合
        audio_data, sr = extract_audio_segment(self.test_wav_path, start_time=10.0, duration=1.0)
        
        # ファイル範囲外の場合、最小限のデータ（またはほぼ空）が返されることを確認
        # start_frameがlen(f)-1に制限されるため、1サンプルが返される
        self.assertLessEqual(len(audio_data), 10)  # ほぼ空（10サンプル以下）

    def test_third_octave_sel_multiple_bands(self):
        """複数の1/3オクターブバンドSELの計算をテスト"""
        center_freqs = [63, 125, 200, 500, 1000, 2000]
        sel_values = calculate_third_octave_sel(self.test_signal, self.sr, center_freqs=center_freqs)
        
        # 全ての周波数バンドでSELが計算されていることを確認
        for fc in center_freqs:
            key = f"{fc}Hz"
            self.assertIn(key, sel_values)
            self.assertIsInstance(sel_values[key], (int, float))
        
        # 1000 HzバンドのSELが他のバンドより高いことを確認（1000 Hz正弦波なので）
        self.assertGreater(sel_values["1000Hz"], sel_values["63Hz"])
        self.assertGreater(sel_values["1000Hz"], sel_values["2000Hz"])


if __name__ == "__main__":
    unittest.main()

