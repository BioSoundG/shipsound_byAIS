import os
import sys
import unittest
import pandas as pd
import numpy as np
import datetime
import tempfile

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_processing import read_ais, complement_trajectory


class TestDataProcessing(unittest.TestCase):
    def setUp(self):
        # テスト用の一時ディレクトリを作成
        self.temp_dir = tempfile.TemporaryDirectory()

        # テスト用のAISデータを作成
        self.test_ais_data = pd.DataFrame(
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

        # テスト用のCSVファイルを作成
        self.test_csv_path = os.path.join(self.temp_dir.name, "test_ais_data.csv")
        self.test_ais_data.to_csv(self.test_csv_path, index=False)

    def tearDown(self):
        # 一時ディレクトリを削除
        self.temp_dir.cleanup()

    def test_read_ais(self):
        """read_ais関数のテスト"""
        # read_ais関数を呼び出し
        result_df = read_ais(self.test_csv_path)

        # 結果がDataFrameであることを確認
        self.assertIsInstance(result_df, pd.DataFrame)

        # dt_pos_utcがdatetime型に変換されていることを確認
        self.assertEqual(result_df["dt_pos_utc"].dtype, "datetime64[ns]")

        # MMSIでソートされていることを確認
        mmsi_sorted = result_df["mmsi"].tolist()
        self.assertEqual(mmsi_sorted, sorted(self.test_ais_data["mmsi"].tolist()))

        # 各船舶のデータが時間順にソートされていることを確認
        for mmsi in result_df["mmsi"].unique():
            vessel_df = result_df[result_df["mmsi"] == mmsi]
            times = vessel_df["dt_pos_utc"].tolist()
            self.assertEqual(times, sorted(times))

        # 元のデータと行数が同じであることを確認
        self.assertEqual(len(result_df), len(self.test_ais_data))

    def test_complement_trajectory(self):
        """complement_trajectory関数のテスト"""
        # complement_trajectory関数を呼び出し
        result_df = complement_trajectory(self.test_csv_path)

        # 結果がDataFrameであることを確認
        self.assertIsInstance(result_df, pd.DataFrame)

        # 補完により行数が増えていることを確認（1秒間隔で補完されるため）
        self.assertGreater(len(result_df), len(self.test_ais_data))

        # dt_pos_utcがインデックスでなく、通常の列として存在することを確認
        self.assertIn("dt_pos_utc", result_df.columns)

        # 各船舶のデータが1秒間隔で補完されていることをチェック
        for mmsi in result_df["mmsi"].unique():
            vessel_df = result_df[result_df["mmsi"] == mmsi].sort_values("dt_pos_utc")
            # 時間差が1秒であることをチェック（最初と最後の行を除く）
            for i in range(len(vessel_df) - 1):
                time_diff = (
                    vessel_df.iloc[i + 1]["dt_pos_utc"]
                    - vessel_df.iloc[i]["dt_pos_utc"]
                ).total_seconds()
                self.assertEqual(time_diff, 1.0)

        # depth列が追加されていることを確認
        self.assertIn("depth", result_df.columns)
        self.assertTrue((result_df["depth"] == 0).all())

        # 補完により新しく作られた行の値が線形補間されていることを確認
        # 例：VesselAの最初と最後の緯度値間の中間値がデータに含まれているか
        vessel_a_df = result_df[result_df["mmsi"] == 123456789].sort_values(
            "dt_pos_utc"
        )
        first_lat = vessel_a_df.iloc[0]["latitude"]
        last_lat = vessel_a_df.iloc[-1]["latitude"]
        mid_lat = (first_lat + last_lat) / 2
        # 中間値付近の値がデータに含まれていることを確認
        self.assertTrue(
            any(abs(lat - mid_lat) < 0.005 for lat in vessel_a_df["latitude"])
        )


if __name__ == "__main__":
    unittest.main()
