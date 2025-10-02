import soundfile as sf
import datetime
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple


class WavFileIndex:
    def __init__(self, wav_list: List[str], record_start_time: pd.Timestamp):
        """
        WAVファイルのインデックスを管理するクラス

        Args:
            wav_list (List[str]): WAVファイルのパスのリスト
            record_start_time (pd.Timestamp): 録音開始時刻
        """
        self.wav_list = wav_list
        self.record_start_time = record_start_time
        self.time_ranges = []
        self.total_samples = 0
        self._build_index()
        self._build_time_index()

    def _build_index(self) -> None:
        """WAVファイルの時間範囲とサンプル数を事前計算"""
        current_time = self.record_start_time
        for wav_file in self.wav_list:
            with sf.SoundFile(wav_file) as f:
                duration = len(f) / f.samplerate
                self.time_ranges.append(
                    {
                        "start": current_time,
                        "end": current_time + datetime.timedelta(seconds=duration),
                        "start_sample": self.total_samples,
                        "samples": len(f),
                        "samplerate": f.samplerate,
                    }
                )
                self.total_samples += len(f)
                current_time += datetime.timedelta(seconds=duration)

    def _build_time_index(self) -> None:
        """時間範囲のインデックスを構築（バイナリサーチ用）"""
        self.time_index = []
        for i, time_range in enumerate(self.time_ranges):
            self.time_index.append((time_range["start"], i))
        self.time_index.sort(key=lambda x: x[0])

    def find_wav_index(
        self, target_time: pd.Timestamp, start_index: int = 0
    ) -> Optional[int]:
        """
        指定時刻を含むWAVファイルのインデックスを探索

        Args:
            target_time (pd.Timestamp): 探索対象の時刻
            start_index (int): 探索開始インデックス

        Returns:
            Optional[int]: WAVファイルのインデックス。見つからない場合はNone
        """
        # 開始インデックス以降のデータのみを対象
        search_index = [x for x in self.time_index if x[1] >= start_index]

        # バイナリサーチ
        left, right = 0, len(search_index) - 1
        while left <= right:
            mid = (left + right) // 2
            current_index = search_index[mid][1]
            if (
                self.time_ranges[current_index]["start"]
                <= target_time
                <= self.time_ranges[current_index]["end"]
            ):
                return current_index
            elif self.time_ranges[current_index]["start"] > target_time:
                right = mid - 1
            else:
                left = mid + 1
        return None

    def get_sample_offset(self, wav_index: int) -> int:
        """
        指定WAVファイルの開始サンプル位置を取得

        Args:
            wav_index (int): WAVファイルのインデックス

        Returns:
            int: 開始サンプル位置
        """
        return (
            self.time_ranges[wav_index]["start_sample"]
            if wav_index < len(self.time_ranges)
            else 0
        )

    def get_wav_durations(self) -> List[float]:
        """
        各WAVファイルの継続時間（秒）を取得

        Returns:
            List[float]: 継続時間のリスト
        """
        return [
            time_range["samples"] / time_range["samplerate"]
            for time_range in self.time_ranges
        ]

    def get_time_range(self, wav_index: int) -> Dict[str, pd.Timestamp]:
        """
        指定WAVファイルの時間範囲を取得

        Args:
            wav_index (int): WAVファイルのインデックス

        Returns:
            Dict[str, pd.Timestamp]: 開始時刻と終了時刻を含む辞書
        """
        if 0 <= wav_index < len(self.time_ranges):
            return {
                "start": self.time_ranges[wav_index]["start"],
                "end": self.time_ranges[wav_index]["end"],
            }
        return None

    def get_total_duration(self) -> datetime.timedelta:
        """
        全WAVファイルの合計継続時間を取得

        Returns:
            datetime.timedelta: 合計継続時間
        """
        if not self.time_ranges:
            return datetime.timedelta(0)
        return self.time_ranges[-1]["end"] - self.time_ranges[0]["start"]
