import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from visualization import plot_geolocation
import os
import matplotlib

# デバッグ目的で対話モードを無効化
matplotlib.use("Agg")

# テスト用ディレクトリ作成
output_dir = "./test_output_debug"
os.makedirs(output_dir, exist_ok=True)

# ===== テストケース1: 意図的に船舶間の座標を離す =====
print("テストケース1: 船舶間の座標を大きく離す")
test_df1 = pd.DataFrame(
    {
        "mmsi": [123456789, 123456789, 123456789, 987654321, 987654321, 987654321],
        "vessel_name": [
            "Vessel A",
            "Vessel A",
            "Vessel A",
            "Vessel B",
            "Vessel B",
            "Vessel B",
        ],
        "vessel_type": [
            "Cargo",
            "Cargo",
            "Cargo",
            "Passenger",
            "Passenger",
            "Passenger",
        ],
        "latitude": [32.5, 32.6, 32.7, 33.1, 33.2, 33.3],  # 明確に離れた位置
        "longitude": [129.5, 129.6, 129.7, 130.1, 130.2, 130.3],
        "dt_pos_utc": [
            pd.Timestamp("2024-03-19 07:00:00"),
            pd.Timestamp("2024-03-19 07:02:00"),
            pd.Timestamp("2024-03-19 07:04:00"),
            pd.Timestamp("2024-03-19 07:01:00"),
            pd.Timestamp("2024-03-19 07:03:00"),
            pd.Timestamp("2024-03-19 07:05:00"),
        ],
    }
)

# 録音位置設定
record_pos = [129.77558, 32.71161]

# 時間順ソートの検証
print("\nVessel A (123456789)のデータ:")
vessel_a = test_df1[test_df1["mmsi"] == 123456789].sort_values("dt_pos_utc")
print(vessel_a[["dt_pos_utc", "latitude", "longitude"]])

print("\nVessel B (987654321)のデータ:")
vessel_b = test_df1[test_df1["mmsi"] == 987654321].sort_values("dt_pos_utc")
print(vessel_b[["dt_pos_utc", "latitude", "longitude"]])

# 通常のプロット生成
plot_geolocation(1, test_df1, record_pos, output_dir)
print("プロット1が生成されました:", os.path.join(output_dir, "1.png"))

# ===== テストケース2: sort_valuesを明示的に確認 =====
print("\nテストケース2: sort_valuesの挙動を確認")

# vessel_dfのソートを直接確認
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

# 録音位置プロット
ax.scatter(record_pos[0], record_pos[1], c="blue", marker="*", s=100, label="rec_pos")

# 各船舶の軌跡を明示的にプロット
colors = ["red", "green"]
for i, mmsi in enumerate([123456789, 987654321]):
    # 特定のMMSIの船舶データを抽出
    vessel_df = test_df1[test_df1["mmsi"] == mmsi]
    vessel_name = vessel_df["vessel_name"].iloc[0]

    # 時間順にソート
    vessel_df_sorted = vessel_df.sort_values("dt_pos_utc")

    # ソート前後のデータ出力
    print(f"\n{vessel_name} (MMSI: {mmsi}):")
    print("ソート前:")
    print(vessel_df[["dt_pos_utc", "latitude", "longitude"]])
    print("ソート後:")
    print(vessel_df_sorted[["dt_pos_utc", "latitude", "longitude"]])

    # 船の位置をプロット (各ポイント)
    ax.scatter(
        vessel_df_sorted["longitude"],
        vessel_df_sorted["latitude"],
        color=colors[i],
        label=vessel_name,
        s=50,
    )

    # 明示的に軌跡を線でつなぐ
    ax.plot(
        vessel_df_sorted["longitude"],
        vessel_df_sorted["latitude"],
        color=colors[i],
        linestyle="-",
        linewidth=2,
    )

    # 各点に時間を表示
    for idx, row in vessel_df_sorted.iterrows():
        ax.text(
            row["longitude"],
            row["latitude"],
            row["dt_pos_utc"].strftime("%H:%M:%S"),
            fontsize=9,
            ha="left",
            va="bottom",
            color=colors[i],
        )

ax.legend()
ax.grid(True, linestyle="--", alpha=0.6)

# 保存
plt.savefig(
    os.path.join(output_dir, "debug_explicit_sort.png"), bbox_inches="tight", dpi=150
)
plt.close(fig)
print(
    "デバッグプロットが生成されました:",
    os.path.join(output_dir, "debug_explicit_sort.png"),
)
