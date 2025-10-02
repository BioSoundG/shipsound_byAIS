import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from visualization import plot_geolocation
import os

# テスト用ディレクトリ作成
output_dir = "./test_output"
os.makedirs(output_dir, exist_ok=True)

# テストデータ作成 - 各船舶に複数の位置を持たせる
test_df = pd.DataFrame(
    {
        "mmsi": [
            123456789,
            123456789,
            123456789,
            987654321,
            987654321,
            987654321,
            555666777,
            555666777,
        ],
        "vessel_name": [
            "Vessel A",
            "Vessel A",
            "Vessel A",
            "Vessel B",
            "Vessel B",
            "Vessel B",
            "Vessel C",
            "Vessel C",
        ],
        "vessel_type": [
            "Cargo",
            "Cargo",
            "Cargo",
            "Passenger",
            "Passenger",
            "Passenger",
            "Tanker",
            "Tanker",
        ],
        "latitude": [32.7, 32.72, 32.74, 32.71, 32.73, 32.75, 32.69, 32.71],
        "longitude": [129.7, 129.72, 129.74, 129.71, 129.73, 129.75, 129.69, 129.71],
        "dt_pos_utc": [
            pd.Timestamp("2024-03-19 07:00:00"),
            pd.Timestamp("2024-03-19 07:02:00"),
            pd.Timestamp("2024-03-19 07:04:00"),
            pd.Timestamp("2024-03-19 07:01:00"),
            pd.Timestamp("2024-03-19 07:03:00"),
            pd.Timestamp("2024-03-19 07:05:00"),
            pd.Timestamp("2024-03-19 07:00:30"),
            pd.Timestamp("2024-03-19 07:02:30"),
        ],
    }
)

# 録音位置設定
record_pos = [32.71161, 129.77558]

# プロット生成
print("元のテストデータの概要:")
print(test_df.groupby("mmsi").count()[["dt_pos_utc"]])
print("\nユニークなMMSI:", test_df["mmsi"].unique())

# プロット生成
plot_geolocation(1, test_df, record_pos, output_dir)
print("プロットが生成されました:", os.path.join(output_dir, "1.png"))
