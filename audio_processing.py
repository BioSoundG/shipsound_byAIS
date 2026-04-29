import soundfile as sf
import os
import datetime
import json
import tomli_w  # TOMLファイルの書き込み用ライブラリを追加
import pandas as pd
import numpy as np
from wav_index import WavFileIndex


def cut_wav_file(
    wav_file0,
    wav_file1,
    record_type,
    record_start_time,
    start_time,
    end_time,
    data_sample_num,
    output_dir,
    metadata_for_dis,
    record_pos,
    wav_durations,
    current_wav_index,
):
    data0, samplerate = sf.read(wav_file0, dtype="int16")
    data1, _ = sf.read(wav_file1, dtype="int16")
    if record_type == 2:
        data0 = data0[:, 0]
        data1 = data1[:, 0]
    data = np.concatenate([data0, data1], axis=0)

    start_sample = int((start_time - record_start_time).total_seconds() * samplerate)
    end_sample = int((end_time - record_start_time).total_seconds() * samplerate)

    # デバッグ出力をファイルに追記
    with open("debug_log.txt", "a") as dbg:
        dbg.write(
            f"[DEBUG] start_sample={start_sample}, end_sample={end_sample}, data_sample_num={data_sample_num}, len(data)={len(data)}\n"
        )

    if data_sample_num < start_sample and data_sample_num + len(data) > end_sample:
        print(f"cutting.....")
        cut_data = data[
            (start_sample - data_sample_num) : (end_sample - data_sample_num)
        ]
        wav_name = f"cut_{start_time.strftime('%Y%m%d_%H%M%S')}_{record_pos[0]}_{record_pos[1]}"
        # 出力先をoutput_dir/wavに修正
        wav_output_dir = os.path.join(output_dir, "wav")
        os.makedirs(wav_output_dir, exist_ok=True)
        cut_file = os.path.join(wav_output_dir, wav_name + ".wav")
        sf.write(cut_file, cut_data, samplerate)

        # 使用するWAVファイルを決定
        mother_source_idx = current_wav_index
        wav_file0_samples = len(data0)  # 1つ目のWAVファイルのサンプル数
        mid_point = data_sample_num + wav_file0_samples  # 2つのWAVファイルの境界点

        print(
            f"DEBUG: data_sample_num = {data_sample_num}, wav_file0_samples = {wav_file0_samples}"
        )
        print(f"DEBUG: boundary = {mid_point}, start_sample = {start_sample}")

        # 切り出しが主に2つ目のファイルに含まれるかどうかを判断
        # 切り出しの開始位置が1つ目のファイルの終了位置より後ろにある場合、2つ目のファイルを使用
        if start_sample >= mid_point:
            mother_source = os.path.basename(wav_file1)
            mother_source_idx = current_wav_index + 1
            # 2つ目のファイルの開始位置からの相対位置を計算
            cut_start_in_mother = start_sample - mid_point
            print(
                f"DEBUG: 母ファイル選択: wav_file1 = {mother_source}, start_sample = {start_sample}, mid_point = {mid_point}"
            )
            print(
                f"DEBUG: cut_start_in_mother = start_sample - mid_point = {start_sample} - {mid_point} = {cut_start_in_mother}"
            )
        else:
            mother_source = os.path.basename(wav_file0)
            cut_start_in_mother = start_sample - data_sample_num
            print(
                f"DEBUG: 母ファイル選択: wav_file0 = {mother_source}, cut_start_in_mother = {cut_start_in_mother}"
            )

        metadata_for_dis["source_info"]["mother_source_name"] = mother_source

        # start_dateからmother_source_nameのwavファイルが始まるまでの時間を計算
        # 元のstart_dateから母ファイルが始まるまでの経過時間を計算（前のWAVファイルの長さを累積）
        total_previous_duration_seconds = 0
        for i in range(mother_source_idx):
            total_previous_duration_seconds += wav_durations[i]

        # 母ファイルの開始時間を計算
        mother_start_time = record_start_time + datetime.timedelta(
            seconds=total_previous_duration_seconds
        )
        metadata_for_dis["source_info"]["mother_source_start_time"] = (
            mother_start_time.strftime("%Y%m%d_%H%M%S")
        )

        # 母ファイル内での切り出し開始時間を計算
        # start_sampleは記録開始からの位置、data_sample_numは現在処理中のファイルの開始位置
        # 現在のファイルの開始位置からの相対位置を計算し、それに基づいてcut_source_start_time_in_mother_sourceを設定
        cut_seconds_in_mother = cut_start_in_mother / samplerate  # 秒数に変換

        # 秒数を時:分:秒形式に変換
        hours, remainder = divmod(cut_seconds_in_mother, 3600)
        minutes, seconds = divmod(remainder, 60)
        # 秒も2桁でフォーマット（整数部分が1桁の場合は0パディング）
        seconds_int = int(seconds)
        seconds_frac = seconds - seconds_int
        time_format = f"{int(hours):02d}:{int(minutes):02d}:{seconds_int:02d}{seconds_frac:.2f}".replace(
            "0.", "."
        )

        metadata_for_dis["source_info"][
            "cut_source_start_time_in_mother_source"
        ] = f"{cut_start_in_mother} samples ({time_format})"

        # Set the cut_source_name to match the actual cut WAV file name
        cut_wav_filename = f"cut_{start_time.strftime('%Y%m%d_%H%M%S')}.wav"
        metadata_for_dis["source_info"]["cut_source_name"] = cut_wav_filename

        return len(data0), True, metadata_for_dis, wav_name
    else:
        print(f"out of target range. next wav.....")
        return len(data0), False, metadata_for_dis, None


def update_metadata_and_save_toml(meta_d, distance, wav_name, output_dir):
    """
    メタデータを更新し、TOMLファイルを保存する

    Args:
        meta_d (dict): 更新するメタデータ
        distance (pd.Series): 船舶の距離情報
        wav_name (str): 生成されたWAVファイル名
        output_dir (str): 出力ディレクトリ
    """
    # Set the vessel sound source information
    # Category for vessel sounds is Anthrophony (1)
    meta_d["source_info"]["category"] = 1

    # Set the vessel name as the sound source using the format: vessel_type(vessel_name)
    vessel_type = (
        distance["vessel_type"] if not pd.isna(distance["vessel_type"]) else "Unknown"
    )
    vessel_name = (
        distance["vessel_name"] if not pd.isna(distance["vessel_name"]) else "Unknown"
    )
    meta_d["source_info"]["sound_source"] = f"{vessel_type}({vessel_name})"

    # Reliability is 2 since we confirmed the vessel via AIS data
    meta_d["source_info"]["reliability"] = 2

    # Add condition information
    meta_d["source_info"][
        "condition"
    ] = f"Ship distance: {distance['min_distance [m]']:.2f}m"

    # Get ship length and width from AIS data if available
    ship_length = (
        distance["length"]
        if "length" in distance.index and not pd.isna(distance["length"])
        else "Unknown"
    )
    ship_width = (
        distance["width"]
        if "width" in distance.index and not pd.isna(distance["width"])
        else "Unknown"
    )

    # Format vessel details for the appendix field including length and width
    vessel_info_str = (
        f"mmsi: {distance['mmsi']}, "
        f"vessel_type: {vessel_type}, "
        f"length: {ship_length}, "
        f"width: {ship_width}, "
        f"min_distance_m: {distance['min_distance [m]']}, "
        f"min_distance_time: {distance['min_distance_time'].strftime('%Y-%m-%dT%H:%M:%S')}"
    )

    # Set appendix with vessel information
    meta_d["source_info"]["appendix"] = vessel_info_str

    # TOMLファイルを生成して保存
    wav_output_dir = os.path.join(output_dir, "wav")
    os.makedirs(wav_output_dir, exist_ok=True)
    with open(os.path.join(wav_output_dir, f"{wav_name}.toml"), "wb") as f:
        tomli_w.dump(meta_d, f)


def cut_wav_and_make_metadata(
    wav_list,
    meta_data,
    start_tim,
    distances,
    distance_list,
    output_dir,
    record_pos,
    audio_config,
):
    """
    Cuts the WAV file based on the shortest distance information and generates corresponding metadata.

    Args:
        wav_list (list): List of WAV files.
        meta_data (dict): Metadata from a TOML file.
        start_tim (str): Start time for the recording.
        distances (DataFrame): DataFrame of shortest distances between vessels and the recording position.
        distance_list (DataFrame): DataFrame of distances between the recording position and other vessels.
        output_dir (str): Path to the output directory where the files will be saved.
        record_pos (tuple): The recording position (latitude, longitude).
        audio_config (dict): Dictionary containing audio processing parameters from config.toml.
    
    Returns:
        int: Number of WAV files actually cut and saved.
    """
    # 出力ディレクトリを作成
    wav_output_dir = os.path.join(output_dir, "wav")
    os.makedirs(wav_output_dir, exist_ok=True)

    # Load parameters from config
    cut_margin_minutes = audio_config.get("cut_margin_minutes", 1)  # Default 1 minute
    max_cut_distance = audio_config.get(
        "max_cut_distance", float("inf")
    )  # デフォルトは無限大（制限なし）
    check_other_vessels = audio_config.get(
        "check_other_vessels", False
    )  # デフォルトはFalse

    # WAVファイルのインデックスを構築
    wav_index = WavFileIndex(wav_list, pd.to_datetime(start_tim))
    
    # WAV時刻範囲をデバッグ出力
    wav_start_time = wav_index.record_start_time
    wav_durations = wav_index.get_wav_durations()
    total_duration = sum(wav_durations)
    wav_end_time = wav_start_time + datetime.timedelta(seconds=total_duration)
    print(f"\n[DEBUG] WAVファイル情報:")
    print(f"  ファイル数: {len(wav_list)}")
    print(f"  録音開始時刻: {wav_start_time}")
    print(f"  録音終了時刻: {wav_end_time}")
    print(f"  総録音時間: {total_duration/3600:.2f} 時間")

    # 最接近時刻でソート
    sorted_distances = distances.sort_values("min_distance_time")

    # 前回の探索位置を記録
    last_processed_wav_index = 0
    
    # 実際に切り出されたWAVファイル数をカウント
    cut_count = 0
    
    # デバッグ用：スキップ理由のカウント
    skip_reasons = {
        "distance_exceeded": 0,
        "not_closest_vessel": 0,
        "wav_not_found": 0,
        "wav_index_out_of_range": 0,
        "cut_failed": 0,
    }

    print(f"\n[DEBUG] 切り出し処理開始: 対象船舶数 = {len(sorted_distances)}")

    for vessel_idx, distance in sorted_distances.iterrows():
        mmsi = distance['mmsi']
        vessel_name = distance.get('vessel_name', 'Unknown')
        min_dist = distance["min_distance [m]"]
        min_dist_time = distance["min_distance_time"]
        
        print(f"\n[DEBUG] 船舶 {vessel_idx+1}/{len(sorted_distances)}: {vessel_name} (MMSI: {mmsi})")
        print(f"        最短距離: {min_dist:.2f}m, 時刻: {min_dist_time}")
        
        # 条件1: 最短距離が設定した距離以下かチェック
        if min_dist > max_cut_distance:
            print(f"        → スキップ: 距離超過 ({min_dist:.2f}m > {max_cut_distance:.2f}m)")
            skip_reasons["distance_exceeded"] += 1
            continue

        # 条件2: 他の船舶との距離比較をチェック（必要な場合）
        if check_other_vessels and not distance_list.empty:
            print(f"        他船比較チェック実施中...")
            target_mmsi = mmsi
            is_closest_vessel = True

            for other_mmsi in distances["mmsi"].unique():
                if other_mmsi == target_mmsi:
                    continue

                other_vessel_data = distance_list[distance_list["mmsi"] == other_mmsi]
                if other_vessel_data.empty:
                    continue

                other_vessel_data = other_vessel_data.copy()
                other_vessel_data["time_diff"] = abs(
                    other_vessel_data["dt_pos_utc"] - min_dist_time
                )
                closest_record = other_vessel_data.loc[
                    other_vessel_data["time_diff"].idxmin()
                ]

                if closest_record["distance [m]"] < min_dist:
                    print(
                        f"        → スキップ: 他船がより近い (他船MMSI: {other_mmsi}, "
                        f"{closest_record['distance [m]']:.2f}m < {min_dist:.2f}m)"
                    )
                    is_closest_vessel = False
                    break

            if not is_closest_vessel:
                skip_reasons["not_closest_vessel"] += 1
                continue
            else:
                print(f"        他船比較: OK (最も近い船舶)")

        # 切り出し時刻の計算
        margin_delta = datetime.timedelta(minutes=cut_margin_minutes)
        start_time = min_dist_time - margin_delta
        end_time = min_dist_time + margin_delta
        
        print(f"        切り出し時刻: {start_time} ~ {end_time} (マージン: {cut_margin_minutes}分)")

        # 前回の探索位置から開始
        wav_idx = wav_index.find_wav_index(start_time, last_processed_wav_index)
        print(f"        WAVファイル検索: wav_idx={wav_idx}, last_processed={last_processed_wav_index}")
        
        if wav_idx is None:
            print(f"        → スキップ: WAVファイルが見つからない (時刻範囲外)")
            skip_reasons["wav_not_found"] += 1
            continue

        # 切り出し処理
        if wav_idx < len(wav_list) - 1:
            print(f"        切り出し実行中: {os.path.basename(wav_list[wav_idx])} & {os.path.basename(wav_list[wav_idx + 1])}")
            # サンプルオフセットを取得
            data_sample_num = wav_index.get_sample_offset(wav_idx)

            # メタデータのコピーを作成
            metadata_for_dis = meta_data.copy()

            # 切り出し開始時刻をTOMLファイルのstart_dateに設定
            metadata_for_dis["observation_info"]["date_info"]["start_date"] = (
                start_time.strftime("%Y-%m-%dT%H:%M:%S")
            )

            # 切り出し処理
            sample_num, flag, meta_d, wav_name = cut_wav_file(
                wav_list[wav_idx],
                wav_list[wav_idx + 1],
                meta_data["observation_info"]["record_info"]["channel_num"],
                wav_index.record_start_time,
                start_time,
                end_time,
                data_sample_num,
                output_dir,
                metadata_for_dis,
                record_pos,
                wav_index.get_wav_durations(),
                wav_idx,
            )

            if flag:
                # メタデータの更新とTOMLファイルの生成
                update_metadata_and_save_toml(meta_d, distance, wav_name, output_dir)
                last_processed_wav_index = wav_idx
                cut_count += 1
                print(f"        ✓ 切り出し成功: {wav_name}.wav")
            else:
                print(f"        → スキップ: 切り出し失敗 (flag=False)")
                skip_reasons["cut_failed"] += 1
        else:
            print(f"        → スキップ: WAVインデックス範囲外 (wav_idx={wav_idx} >= {len(wav_list)-1})")
            skip_reasons["wav_index_out_of_range"] += 1
    
    # デバッグサマリーを出力
    print("\n" + "="*60)
    print("[DEBUG] 切り出し処理完了サマリー:")
    print("="*60)
    print(f"対象船舶数:             {len(sorted_distances):>6} 隻")
    print(f"切り出し成功:           {cut_count:>6} 個")
    print(f"スキップ合計:           {sum(skip_reasons.values()):>6} 隻")
    print("-"*60)
    print("スキップ内訳:")
    print(f"  - 距離超過:           {skip_reasons['distance_exceeded']:>6} 隻")
    print(f"  - 他船がより近い:     {skip_reasons['not_closest_vessel']:>6} 隻")
    print(f"  - WAV見つからず:      {skip_reasons['wav_not_found']:>6} 隻")
    print(f"  - WAVインデックス範囲外: {skip_reasons['wav_index_out_of_range']:>6} 隻")
    print(f"  - 切り出し失敗:       {skip_reasons['cut_failed']:>6} 隻")
    print("="*60)
    
    return cut_count
