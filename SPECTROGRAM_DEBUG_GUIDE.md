# スペクトログラムが-80dBになる問題のデバッグガイド

## 問題の症状

スペクトログラムの全てのビンが-80dB（最小値）になり、音響情報が表示されない。

---

## 考えられる原因と対処法

### **1. 音声データが無音またはゼロ**

#### **症状**
```
[DEBUG] Data range: [0.000000, 0.000000]
[DEBUG] Non-zero samples: 0/13230000
[WARNING] Audio data appears to be silent or nearly zero!
```

#### **原因**
- WAVファイルが実際に無音
- WAVファイルが破損している
- 録音が失敗している

#### **対処法**
1. **別の音声再生ソフトでWAVファイルを確認**
   ```bash
   # Windowsの場合
   start your_wav_file.wav
   
   # 音が聞こえるか確認
   ```

2. **WAVファイルのメタデータを確認**
   ```python
   import soundfile as sf
   with sf.SoundFile('your_file.wav') as f:
       print(f"Sample rate: {f.samplerate}")
       print(f"Channels: {f.channels}")
       print(f"Subtype: {f.subtype}")
       print(f"Format: {f.format}")
       
       # 最初の1秒を読み込んで確認
       data = f.read(f.samplerate)
       print(f"Data range: [{data.min()}, {data.max()}]")
   ```

---

### **2. 整数型WAVファイルの正規化問題**

#### **症状**
```
[DEBUG] File format: subtype=PCM_16, format=WAV
[DEBUG] Data range: [0.000000, 0.000031]  # 非常に小さい値
[DEBUG] Sxx range: [1.23e-20, 4.56e-18]   # 極小のパワー
[DEBUG] dB range: [-100.00, -80.00]
```

#### **原因**
- WAVファイルが16ビット整数（PCM_16）で保存されている
- `soundfile`は`dtype='float32'`指定時に自動正規化するが、何らかの理由で正規化が不十分
- ビット深度が24ビットや32ビット整数の場合も同様の問題が発生

#### **対処法**

**方法1: 明示的な正規化を追加**

```python
# visualization.pyの該当箇所を修正
data = f_soundfile.read(num_samples_to_read, dtype="float32", always_2d=False)

# ステレオ→モノラル変換
if len(data.shape) > 1 and data.shape[1] > 1:
    data = data[:, 0]

# 明示的な正規化（追加）
if f_soundfile.subtype in ['PCM_16', 'PCM_24', 'PCM_32']:
    # 整数型の場合、正規化されているか確認
    if np.max(np.abs(data)) < 1e-3:
        print(f"    [WARNING] Data appears unnormalized. Max value: {np.max(np.abs(data))}")
        # 16ビットの場合は32768で、24ビットの場合は8388608で除算
        bit_depth_map = {'PCM_16': 32768.0, 'PCM_24': 8388608.0, 'PCM_32': 2147483648.0}
        normalization_factor = bit_depth_map.get(f_soundfile.subtype, 32768.0)
        # Note: soundfileは通常自動正規化するので、この問題は稀
```

**方法2: 異なる読み込み方法**

```python
# dtype指定なしで読み込み（soundfileがデフォルトで正規化）
data = f_soundfile.read(num_samples_to_read, always_2d=False)
```

---

### **3. サンプリングレートの不一致**

#### **症状**
```
[DEBUG] Sample rate: 96000Hz  # 高いサンプリングレート
[DEBUG] nperseg: 4096
# npersegが小さすぎて周波数分解能が不足
```

#### **原因**
- 高いサンプリングレート（96kHz、192kHzなど）のファイル
- `nperseg=4096`では窓長が約43ms（96kHzの場合）と短すぎる
- 低周波の情報が不足

#### **対処法**

`config.toml`で`nperseg`を調整：

```toml
[visualization]
spectrogram_nperseg = 8192  # または 16384（サンプリングレートに応じて）
```

**推奨値**:
- 44.1kHz: nperseg = 4096（93ms窓）
- 48kHz: nperseg = 4096（85ms窓）
- 96kHz: nperseg = 8192（85ms窓）
- 192kHz: nperseg = 16384（85ms窓）

---

### **4. チャンク長が短すぎる**

#### **症状**
```
[DEBUG] Spectrogram shape: (2049, 0)  # 時間軸が0
Error calculating spectrogram for chunk 1: ...
```

#### **原因**
- `chunk_duration_seconds`が小さすぎる
- WAVファイルの継続時間よりチャンク長が大きい
- `num_samples_to_read < nperseg`

#### **対処法**

`config.toml`で調整：

```toml
[visualization]
chunk_duration_seconds = 300  # 最小60秒以上を推奨
```

---

### **5. dBレンジの設定問題**

#### **症状**
```
[DEBUG] dB range: [-100.00, -85.00]  # 実際の値
# しかし設定は db_min=-80, db_max=-10
# 全ての値が-80dB以下なので、表示上は全て-80dBに見える
```

#### **原因**
- 実際のdB値が設定範囲外
- 音声が非常に小さい
- 背景ノイズレベルが低い環境

#### **対処法**

`config.toml`でdBレンジを調整：

```toml
[visualization]
plot_db_min = -120  # より低い値に設定
plot_db_max = -40   # より低い値に設定
```

または、デバッグ出力から実際のdBレンジを確認して調整。

---

## デバッグ手順

### **ステップ1: デバッグ情報の確認**

プログラムを実行すると、以下のような出力が表示されます：

```
Processing time-averaged spectrogram for your_file.wav...
[DEBUG] File format: subtype=PCM_16, format=WAV
[DEBUG] Channels: 1, Frames: 158760000
  Total duration: 3600.00s, Sample rate: 44100Hz
  Processing in 12 chunks of ~300s...
    Processing chunk 1/12 (0.0s - 300.0s)...
    [DEBUG] Data shape: (13230000,)
    [DEBUG] Data dtype: float32
    [DEBUG] Data range: [-0.123456, 0.234567]  # ← ここを確認
    [DEBUG] Data mean: 0.000123
    [DEBUG] Data std: 0.045678
    [DEBUG] Non-zero samples: 13230000/13230000  # ← ほぼ全て非ゼロなら正常
    [DEBUG] Sample values: [-0.01234 0.02345 -0.00123 0.01456 -0.00987]
    [DEBUG] Spectrogram shape: (2049, 6510)
    [DEBUG] Sxx range: [1.23e-08, 4.56e-04]  # ← パワーの範囲
    [DEBUG] Sxx mean: 2.34e-06
    [DEBUG] Avg spectrum range: [3.45e-08, 1.23e-04]
    [DEBUG] dB range: [-74.62, -39.10]  # ← 実際のdB値
```

### **ステップ2: 問題の特定**

#### **ケース A: データが全てゼロ**
```
[DEBUG] Data range: [0.000000, 0.000000]
[DEBUG] Non-zero samples: 0/13230000
```
→ WAVファイルが無音または破損。原因1を確認。

#### **ケース B: データが非常に小さい**
```
[DEBUG] Data range: [-0.000031, 0.000031]
[DEBUG] dB range: [-100.00, -85.00]
```
→ 正規化の問題。原因2を確認。

#### **ケース C: dB値が設定範囲外**
```
[DEBUG] dB range: [-95.00, -85.00]
# 設定: db_min=-80, db_max=-10
```
→ dBレンジの調整が必要。原因5を確認。

### **ステップ3: 修正と再実行**

1. デバッグ出力から問題を特定
2. 上記の対処法を適用
3. プログラムを再実行
4. スペクトログラムが正しく表示されることを確認

---

## よくある質問

### **Q: 正常なスペクトログラムの例は？**

正常な場合のデバッグ出力：
```
[DEBUG] Data range: [-0.5, 0.6]         # -1.0〜1.0の範囲内
[DEBUG] Non-zero samples: 13230000/13230000  # ほぼ全て非ゼロ
[DEBUG] Sxx range: [1e-10, 1e-02]       # パワーが広範囲
[DEBUG] dB range: [-60.00, -20.00]      # 実用的なdB範囲
```

### **Q: 複数のWAVファイルで同じ問題が発生する場合は？**

- 全て同じファイル形式を使用している可能性が高い
- 録音設定（ビット深度、サンプリングレート）を確認
- 1つの代表的なファイルで問題を解決してから、他のファイルに適用

### **Q: デバッグ出力を無効にしたい**

デバッグコードの該当箇所（`if chunk_idx == 0:`）をコメントアウト：

```python
# Debug: Check data statistics
# if chunk_idx == 0:
#     print(f"    [DEBUG] Data shape: {data.shape}")
#     ...
```

---

## まとめ

1. **まずデバッグ出力を確認**して問題の種類を特定
2. **最も可能性が高いのは原因2（正規化問題）または原因5（dBレンジ）**
3. **実際のWAVファイルを音声再生ソフトで確認**して、音が含まれていることを確認
4. **段階的に対処**：まず1つのファイルで修正し、その後全ファイルに適用

問題が解決しない場合は、デバッグ出力の全文を保存して、詳細な調査を依頼してください。

