import os
import sys
import unittest
import pandas as pd
import numpy as np
import datetime
import tempfile

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_processing import read_ais, read_all_ais, complement_trajectory


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

    def test_complement_trajectory_plots_before_after(self):
        """plot_before_after=True で補完前後の図が生成されることを確認"""
        # 出力ディレクトリ
        output_dir = os.path.join(self.temp_dir.name, "plots_out")
        os.makedirs(output_dir, exist_ok=True)

        # 記録位置（適当な座標）
        record_pos = (32.71161, 129.77558)

        # 関数実行（補完前後の図を出力）
        _ = complement_trajectory(
            self.test_csv_path,
            record_pos=record_pos,
            output_dir=output_dir,
            plot_before_after=True,
        )

        # 期待ファイルの存在確認
        before_png = os.path.join(output_dir, "traj_before.png")
        after_png = os.path.join(output_dir, "traj_after.png")
        self.assertTrue(os.path.exists(before_png))
        self.assertTrue(os.path.exists(after_png))

    def test_complement_trajectory_no_plots_when_flag_false(self):
        """plot_before_after=False の場合は図が生成されないことを確認"""
        output_dir = os.path.join(self.temp_dir.name, "plots_out2")
        os.makedirs(output_dir, exist_ok=True)
        record_pos = (32.71161, 129.77558)

        _ = complement_trajectory(
            self.test_csv_path,
            record_pos=record_pos,
            output_dir=output_dir,
            plot_before_after=False,
        )

        before_png = os.path.join(output_dir, "traj_before.png")
        after_png = os.path.join(output_dir, "traj_after.png")
        self.assertFalse(os.path.exists(before_png))
        self.assertFalse(os.path.exists(after_png))

    def test_linear_interpolation_lat_lon(self):
        """緯度・経度が線形補間されることを時刻を指定して厳密に検証"""
        # 合成データ: 1船, 0秒と10秒の2点を与え、間を補間
        t0 = pd.Timestamp("2024-01-01 00:00:00")
        t1 = pd.Timestamp("2024-01-01 00:00:10")
        df = pd.DataFrame(
            {
                "mmsi": [111111111, 111111111],
                "vessel_name": ["V1", "V1"],
                "vessel_type": ["Cargo", "Cargo"],
                "length": [100, 100],
                "width": [20, 20],
                "latitude": [10.0, 11.0],
                "longitude": [20.0, 22.0],
                "dt_pos_utc": [t0, t1],
            }
        )
        tmp_csv = os.path.join(self.temp_dir.name, "lin_int.csv")
        df.to_csv(tmp_csv, index=False)

        result = complement_trajectory(tmp_csv)
        # 0〜10秒の11点（両端含む）が生成される
        v = result[result["mmsi"] == 111111111].sort_values("dt_pos_utc")
        self.assertEqual(len(v), 11)
        # 元点は保持
        self.assertAlmostEqual(v.iloc[0]["latitude"], 10.0)
        self.assertAlmostEqual(v.iloc[0]["longitude"], 20.0)
        self.assertAlmostEqual(v.iloc[-1]["latitude"], 11.0)
        self.assertAlmostEqual(v.iloc[-1]["longitude"], 22.0)
        # 4秒後の点を検証（線形補間）: lat=10.4, lon=20.8
        t_mid = t0 + pd.Timedelta(seconds=4)
        row_mid = v[v["dt_pos_utc"] == t_mid].iloc[0]
        self.assertAlmostEqual(row_mid["latitude"], 10.4, places=6)
        self.assertAlmostEqual(row_mid["longitude"], 20.8, places=6)
        # 連続1秒刻みで単調増加
        diffs = v["dt_pos_utc"].diff().dropna().dt.total_seconds().values
        self.assertTrue((diffs == 1.0).all())

    def test_interpolation_is_per_mmsi_and_no_cross_talk(self):
        """MMSIごとに独立に補間され、範囲外の時刻が混入しないことを検証"""
        ta0 = pd.Timestamp("2024-01-01 00:00:00")
        ta1 = pd.Timestamp("2024-01-01 00:00:05")
        tb0 = pd.Timestamp("2024-01-01 00:01:00")
        tb1 = pd.Timestamp("2024-01-01 00:01:03")
        df = pd.DataFrame(
            {
                "mmsi": [111, 111, 222, 222],
                "vessel_name": ["A", "A", "B", "B"],
                "vessel_type": ["Cargo", "Cargo", "Cargo", "Cargo"],
                "length": [100, 100, 100, 100],
                "width": [20, 20, 20, 20],
                "latitude": [0.0, 0.5, 1.0, 1.2],
                "longitude": [0.0, 0.5, 2.0, 2.3],
                "dt_pos_utc": [ta0, ta1, tb0, tb1],
            }
        )
        tmp_csv = os.path.join(self.temp_dir.name, "two_vessels.csv")
        df.to_csv(tmp_csv, index=False)

        result = complement_trajectory(tmp_csv)
        a = result[result["mmsi"] == 111].sort_values("dt_pos_utc")
        b = result[result["mmsi"] == 222].sort_values("dt_pos_utc")
        # Aは0〜5秒で6点、Bは0〜3秒で4点（それぞれ自身の範囲のみ）
        self.assertEqual(len(a), 6)
        self.assertEqual(len(b), 4)
        self.assertGreaterEqual(a["dt_pos_utc"].min(), ta0)
        self.assertLessEqual(a["dt_pos_utc"].max(), ta1)
        self.assertGreaterEqual(b["dt_pos_utc"].min(), tb0)
        self.assertLessEqual(b["dt_pos_utc"].max(), tb1)
        # いずれも1秒刻み
        self.assertTrue(
            (a["dt_pos_utc"].diff().dropna().dt.total_seconds() == 1.0).all()
        )
        self.assertTrue(
            (b["dt_pos_utc"].diff().dropna().dt.total_seconds() == 1.0).all()
        )

    def test_read_all_ais_combines_same_mmsi(self):
        """同一MMSIが別CSVに分かれていても結合され、ソートされることを検証"""
        t0 = pd.Timestamp("2024-01-01 00:00:00")
        t1 = pd.Timestamp("2024-01-01 00:00:05")
        t2 = pd.Timestamp("2024-01-01 00:00:10")
        df_a = pd.DataFrame(
            {
                "mmsi": [999, 999],
                "vessel_name": ["X", "X"],
                "vessel_type": ["Cargo", "Cargo"],
                "length": [100, 100],
                "width": [20, 20],
                "latitude": [0.0, 0.1],
                "longitude": [0.0, 0.1],
                "dt_pos_utc": [t0, t1],
            }
        )
        df_b = pd.DataFrame(
            {
                "mmsi": [999],
                "vessel_name": ["X"],
                "vessel_type": ["Cargo"],
                "length": [100],
                "width": [20],
                "latitude": [0.2],
                "longitude": [0.2],
                "dt_pos_utc": [t2],
            }
        )
        p_a = os.path.join(self.temp_dir.name, "ais_a.csv")
        p_b = os.path.join(self.temp_dir.name, "ais_b.csv")
        df_a.to_csv(p_a, index=False)
        df_b.to_csv(p_b, index=False)

        combined = read_all_ais([p_a, p_b])
        self.assertEqual(len(combined), 3)
        self.assertTrue((combined["mmsi"].unique() == [999]).all())
        # 時系列が昇順
        times = combined["dt_pos_utc"].tolist()
        self.assertEqual(times, sorted(times))

    def test_complement_trajectory_accepts_dataframe(self):
        """DataFrame入力を受け取り、全範囲で1秒補間されることを検証"""
        t0 = pd.Timestamp("2024-01-01 00:00:00")
        t2 = pd.Timestamp("2024-01-01 00:00:02")
        df = pd.DataFrame(
            {
                "mmsi": [777, 777],
                "vessel_name": ["Y", "Y"],
                "vessel_type": ["Cargo", "Cargo"],
                "length": [100, 100],
                "width": [20, 20],
                "latitude": [1.0, 1.2],
                "longitude": [2.0, 2.2],
                "dt_pos_utc": [t0, t2],
            }
        )
        result = complement_trajectory(df)
        v = result[result["mmsi"] == 777].sort_values("dt_pos_utc")
        # 0,1,2秒の3点
        self.assertEqual(len(v), 3)
        # 中央は線形補間
        self.assertAlmostEqual(v.iloc[1]["latitude"], 1.1, places=6)
        self.assertAlmostEqual(v.iloc[1]["longitude"], 2.1, places=6)

    def test_complement_trajectory_drops_nat_timestamps(self):
        """Rows with missing timestamps are ignored before resampling."""
        t0 = pd.Timestamp("2024-01-01 00:00:00")
        t2 = pd.Timestamp("2024-01-01 00:00:02")
        df = pd.DataFrame(
            {
                "mmsi": [111, 111, 111, 222, 222],
                "length": [100, 100, 100, 80, 80],
                "width": [20, 20, 20, 15, 15],
                "latitude": [1.0, 9.9, 1.2, 2.0, 2.1],
                "longitude": [2.0, 9.9, 2.2, 3.0, 3.1],
                "dt_pos_utc": [t0, pd.NaT, t2, pd.NaT, pd.NaT],
            }
        )

        result = complement_trajectory(df)

        self.assertFalse(result["dt_pos_utc"].isna().any())
        self.assertEqual(set(result["mmsi"].unique()), {111})
        v = result[result["mmsi"] == 111].sort_values("dt_pos_utc")
        self.assertEqual(len(v), 3)
        self.assertAlmostEqual(v.iloc[1]["latitude"], 1.1, places=6)
        self.assertAlmostEqual(v.iloc[1]["longitude"], 2.1, places=6)

    def test_complement_trajectory_returns_empty_when_all_timestamps_nat(self):
        """All-invalid timestamp input does not reach pandas resample."""
        df = pd.DataFrame(
            {
                "mmsi": [333, 333],
                "latitude": [1.0, 1.1],
                "longitude": [2.0, 2.1],
                "dt_pos_utc": [pd.NaT, pd.NaT],
            }
        )

        result = complement_trajectory(df)

        self.assertTrue(result.empty)
        self.assertIn("depth", result.columns)

    def test_read_ais_speed_column_normalization(self):
        """speedカラムが様々なヘッダー名から正しく読み込まれることを確認"""
        # テスト1: "speed"ヘッダー
        df1 = pd.DataFrame(
            {
                "mmsi": [111111111],
                "latitude": [32.7],
                "longitude": [129.7],
                "dt_pos_utc": ["2024-03-19 07:00:00"],
                "speed": [12.5],
            }
        )
        path1 = os.path.join(self.temp_dir.name, "test_speed1.csv")
        df1.to_csv(path1, index=False)
        result1 = read_ais(path1)
        self.assertIn("speed", result1.columns)
        self.assertAlmostEqual(result1["speed"].iloc[0], 12.5)

        # テスト2: "SOG"ヘッダー（Speed Over Ground）
        df2 = pd.DataFrame(
            {
                "MMSI": [222222222],
                "LATITUDE": [32.8],
                "LONGITUDE": [129.8],
                "dt_pos_utc": ["2024-03-19 08:00:00"],
                "SOG": [8.3],
            }
        )
        path2 = os.path.join(self.temp_dir.name, "test_speed2.csv")
        df2.to_csv(path2, index=False)
        result2 = read_ais(path2)
        self.assertIn("speed", result2.columns)
        self.assertAlmostEqual(result2["speed"].iloc[0], 8.3)

        # テスト3: "Speed"ヘッダー（大文字始まり）
        df3 = pd.DataFrame(
            {
                "mmsi": [333333333],
                "lat": [32.9],
                "lon": [129.9],
                "dt_pos_utc": ["2024-03-19 09:00:00"],
                "Speed": [15.0],
            }
        )
        path3 = os.path.join(self.temp_dir.name, "test_speed3.csv")
        df3.to_csv(path3, index=False)
        result3 = read_ais(path3)
        self.assertIn("speed", result3.columns)
        self.assertAlmostEqual(result3["speed"].iloc[0], 15.0)

    def test_read_ais_no_speed_column(self):
        """speedカラムがない場合でもエラーにならないことを確認"""
        df = pd.DataFrame(
            {
                "mmsi": [444444444],
                "latitude": [33.0],
                "longitude": [130.0],
                "dt_pos_utc": ["2024-03-19 10:00:00"],
            }
        )
        path = os.path.join(self.temp_dir.name, "test_no_speed.csv")
        df.to_csv(path, index=False)
        result = read_ais(path)
        # speedカラムがない場合は存在しないだけ（エラーにならない）
        self.assertNotIn("speed", result.columns)


if __name__ == "__main__":
    unittest.main()
