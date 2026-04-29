# メモリ最適化ガイド

## 概要
大量のAISデータを処理する際にメモリ不足エラーが発生する場合の対処方法をまとめています。

## メモリ不足が発生する原因

1. **全CSVファイルの同時読み込み**: `combine_all_ais = true`の場合、全てのAIS CSVファイルをメモリに読み込みます
2. **1秒間隔への補間**: 元データが疎でも、1秒間隔にリサンプリングすることでデータ量が大幅に増加
3. **複数のDataFrameコピー**: 処理の各段階で複数のコピーが作成される

## 対策方法

### 方法1: 低メモリモードの使用（推奨）

`config.toml`で以下の設定を有効化:

```toml
[data_processing]
combine_all_ais = true
low_memory_mode = true  # ← これを追加
```

**効果**:
- MMSI単位でデータを処理するため、メモリ使用量が削減
- 処理時間は増加するが、大規模データセットでも動作可能

**処理の流れ**:
1. 各CSVファイルを順次読み込み
2. MMSI別にグループ化してメモリに保持
3. 各MMSI単位でデータを結合・重複削除
4. 最終的に全MMSIを統合

### 方法2: CSVファイルの分割処理

非常に大規模なデータセットの場合、以下のように複数回に分けて処理:

```bash
# 第1バッチ: ファイル1-100
python main.py -a ./ais_data_batch1/ -w ./wav_data/ -m metadata.toml -t 2024-03-19T06:53:00 -c config.toml

# 第2バッチ: ファイル101-200
python main.py -a ./ais_data_batch2/ -w ./wav_data/ -m metadata.toml -t 2024-03-19T06:53:00 -c config.toml
```

### 方法3: システム設定の調整

#### Windowsの場合
仮想メモリ（ページファイル）のサイズを増やす:
1. システムのプロパティ → 詳細設定 → パフォーマンス設定
2. 詳細設定タブ → 仮想メモリ → 変更
3. カスタムサイズを設定（推奨: RAM容量の1.5-3倍）

#### Linuxの場合
スワップ領域を増やす:
```bash
# 現在のスワップ確認
free -h

# スワップファイルの作成（例: 16GB）
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### 方法4: データの前処理

不要な船舶データを事前にフィルタリング:

```python
# 例: 録音位置から一定距離以上離れた船舶を除外
# data_processing.py の read_ais() で距離計算を追加
```

## メモリ使用量の見積もり

### 標準モード (`low_memory_mode = false`)

**概算式**:
```
メモリ使用量 (MB) ≈ CSV合計サイズ (MB) × 3-5
```

**例**:
- CSVファイル合計: 2 GB
- 推定メモリ使用量: 6-10 GB

### 低メモリモード (`low_memory_mode = true`)

**概算式**:
```
メモリ使用量 (MB) ≈ (最大MMSI単位のデータサイズ × 3) + (全MMSI数 × 0.1 MB)
```

**例**:
- 最大の船舶データ: 50 MB
- 船舶数: 1000隻
- 推定メモリ使用量: 150 MB + 100 MB = 250 MB

## トラブルシューティング

### エラー: `MemoryError`

**対策**:
1. `low_memory_mode = true`を設定
2. 不要なアプリケーションを終了
3. CSVファイルを分割して処理

### エラー: `pandas.errors.OutOfMemoryError`

**対策**:
```toml
[data_processing]
combine_all_ais = false  # CSVごとに処理（重複切り出しの可能性あり）
```

### 処理が非常に遅い

**対策**:
1. `low_memory_mode = false`に戻す（メモリが十分な場合）
2. SSDを使用（HDDより高速）
3. データの前処理でフィルタリング

## パフォーマンス比較

| 設定 | メモリ使用量 | 処理時間 | 推奨ケース |
|------|------------|---------|-----------|
| `combine_all_ais=false` | 低 | 中 | 小規模データ（<500MB） |
| `combine_all_ais=true, low_memory=false` | 高 | 短 | 中規模データ（500MB-5GB）、十分なRAM |
| `combine_all_ais=true, low_memory=true` | 中 | 長 | 大規模データ（>5GB）、RAM制限あり |

## モニタリング

処理中のメモリ使用量を確認:

### Windowsの場合
```powershell
# タスクマネージャーを開く
taskmgr

# PowerShellでメモリ監視
Get-Process python | Select-Object Name, WorkingSet
```

### Linuxの場合
```bash
# メモリ使用量をリアルタイム監視
watch -n 1 free -h

# Pythonプロセスのメモリ確認
ps aux | grep python
```

## まとめ

- **小規模データ（<500MB）**: デフォルト設定で問題なし
- **中規模データ（500MB-5GB）**: `combine_all_ais = true`を推奨
- **大規模データ（>5GB）**: `low_memory_mode = true`を有効化
- **超大規模データ（>50GB）**: データを分割して複数回処理

メモリエラーが解決しない場合は、データの前処理やシステムのアップグレードを検討してください。

