import os
import sys
import unittest
import pandas as pd
import numpy as np
import datetime

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from distance_calculation import haversine, calculate_shortest_distance


class TestDistanceCalculation(unittest.TestCase):
    def setUp(self):
        # テスト用のデータフレームを作成
        self.test_df = pd.DataFrame(
            {
                "mmsi": [123456789, 123456789, 987654321, 987654321],
                "vessel_name": ["Vessel A", "Vessel A", "Vessel B", "Vessel B"],
                "vessel_type": ["Cargo", "Cargo", "Passenger", "Passenger"],
                "length": [100, 100, 150, 150],
                "width": [20, 20, 30, 30],
                "latitude": [32.7, 32.71, 32.72, 32.73],
                "longitude": [129.7, 129.71, 129.72, 129.73],
                "dt_pos_utc": [
                    pd.Timestamp("2024-03-19 07:00:00"),
                    pd.Timestamp("2024-03-19 07:01:00"),
                    pd.Timestamp("2024-03-19 07:02:00"),
                    pd.Timestamp("2024-03-19 07:03:00"),
                ],
            }
        )

        # 録音位置と深度
        self.record_pos = [32.71161, 129.77558]
        self.record_depth = 100.0

    def test_haversine(self):
        """haversine関数のテスト"""
        # 既知の座標間の距離を計算
        lat1, lon1 = 35.6812, 139.7671  # 東京駅の緯度経度
        lat2, lon2 = 34.6723, 135.4983  # 大阪駅の緯度経度

        # haversine関数で距離を計算
        distance = haversine(lat1, lon1, lat2, lon2)

        # 東京-大阪間の距離は約400km前後
        self.assertGreater(distance, 390000)  # 390km以上
        self.assertLess(distance, 410000)  # 410km以下

        # 同じ座標の場合は距離0
        self.assertEqual(haversine(lat1, lon1, lat1, lon1), 0)

        # 緯度が増加する場合の距離チェック
        lat3 = lat1 + 1.0
        dist_north = haversine(lat1, lon1, lat3, lon1)
        self.assertGreater(dist_north, 110000)  # 緯度1度は約111km
        self.assertLess(dist_north, 112000)

    def test_calculate_shortest_distance(self):
        """calculate_shortest_distance関数のテスト"""
        # 距離計算を実行
        distances = calculate_shortest_distance(
            self.test_df, self.record_pos, self.record_depth
        )

        # 結果がリストであることを確認
        self.assertIsInstance(distances, list)

        # 結果のリストの長さが2（ユニークなmmsiの数）であることを確認
        self.assertEqual(len(distances), 2)

        # 各船舶の結果を確認
        for dist_info in distances:
            # 必要なキーが全て含まれていることを確認
            required_keys = [
                "mmsi",
                "vessel_name",
                "vessel_type",
                "length",
                "width",
                "min_distance_idx",
                "min_distance [m]",
                "min_distance_pos",
                "min_distance_time",
            ]
            for key in required_keys:
                self.assertIn(key, dist_info)

            # mmsiに基づいて正しい船舶名がセットされていることを確認
            if dist_info["mmsi"] == 123456789:
                self.assertEqual(dist_info["vessel_name"], "Vessel A")
                self.assertEqual(dist_info["vessel_type"], "Cargo")
                self.assertEqual(dist_info["length"], 100)
                self.assertEqual(dist_info["width"], 20)
            elif dist_info["mmsi"] == 987654321:
                self.assertEqual(dist_info["vessel_name"], "Vessel B")
                self.assertEqual(dist_info["vessel_type"], "Passenger")
                self.assertEqual(dist_info["length"], 150)
                self.assertEqual(dist_info["width"], 30)

            # 計算された距離が正の値であることを確認
            self.assertGreater(dist_info["min_distance [m]"], 0)

    def test_calculate_shortest_distance_missing_columns(self):
        """必須カラムが欠けている場合のテスト"""
        # widthカラムを削除したデータフレーム
        df_missing_width = self.test_df.drop(columns=["width"])

        # ValueErrorが発生することを確認
        with self.assertRaises(ValueError) as context:
            calculate_shortest_distance(
                df_missing_width, self.record_pos, self.record_depth
            )

        # エラーメッセージに欠けているカラム名が含まれていることを確認
        self.assertIn("width", str(context.exception))

        # lengthカラムを削除したデータフレーム
        df_missing_length = self.test_df.drop(columns=["length"])

        # ValueErrorが発生することを確認
        with self.assertRaises(ValueError) as context:
            calculate_shortest_distance(
                df_missing_length, self.record_pos, self.record_depth
            )

        # エラーメッセージに欠けているカラム名が含まれていることを確認
        self.assertIn("length", str(context.exception))


if __name__ == "__main__":
    unittest.main()
