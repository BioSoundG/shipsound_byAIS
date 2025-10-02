import pandas as pd
import os
import sys

# CSVファイルを読み込み
csv_path = "g:/20240319_nagasaki/IHI_JAMSTEC_032024/IHI_JAMSTEC_032024/Spire_20240319090000_20240319210000.csv"
df = pd.read_csv(csv_path)

# 列名の一覧を表示
print("CSV Columns:")
print(df.columns.tolist())

# length列とwidth列が存在するか確認
if "length" in df.columns and "width" in df.columns:
    print("\nLENGTH and WIDTH columns exist!")
    print("\nSample data (first 3 rows):")
    print(df[["mmsi", "vessel_name", "vessel_type", "length", "width"]].head(3))
else:
    missing = []
    if "length" not in df.columns:
        missing.append("length")
    if "width" not in df.columns:
        missing.append("width")
    print(f"\nMissing columns: {missing}")

# 現在のディレクトリに移動
sys.path.append(os.getcwd())

# distance_calculationモジュールをインポート
try:
    from distance_calculation import calculate_shortest_distance
    from data_processing import complement_trajectory

    # テスト用のデータフレームを作成
    test_df = complement_trajectory(csv_path)

    # 計算実行前のlength/width確認
    print("\nAfter complement_trajectory - 最初の3行:")
    print(test_df[["mmsi", "vessel_name", "vessel_type", "length", "width"]].head(3))

    # 距離計算
    record_pos = [32.71161, 129.77558]
    record_depth = 100.0
    distances = calculate_shortest_distance(test_df, record_pos, record_depth)

    # 距離計算後の結果を確認
    print("\nAfter distance calculation - 最初の3行:")
    for i, dist in enumerate(distances[:3]):
        print(f"Row {i}:")
        for key, value in dist.items():
            print(f"  {key}: {value}")

    # lengh, widthがあるか確認
    for i, dist in enumerate(distances[:3]):
        if "length" in dist:
            print(f"\nFound length in distance result {i}: {dist['length']}")
        else:
            print(f"\nNo length found in distance result {i}")

        if "width" in dist:
            print(f"Found width in distance result {i}: {dist['width']}")
        else:
            print(f"No width found in distance result {i}")

except Exception as e:
    print(f"Error: {e}")
