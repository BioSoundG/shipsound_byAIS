import numpy as np
from math import radians, sin, cos, sqrt, atan2
import pandas as pd


def haversine(lat1, lon1, lat2, lon2):
    """
    Calculates the Haversine distance between two geographic coordinates.

    Args:
        lat1 (float): Latitude of the first point.
        lon1 (float): Longitude of the first point.
        lat2 (float): Latitude of the second point.
        lon2 (float): Longitude of the second point.

    Returns:
        float: The Haversine distance in meters.
    """
    R = 6371.0  # Earth radius in kilometers
    lat1_rad, lon1_rad = map(radians, [lat1, lon1])
    lat2_rad, lon2_rad = map(radians, [lat2, lon2])
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = sin(dlat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c * 1000  # distance in meters


def calculate_shortest_distance(df, record_pos, record_depth):
    """
    Calculates the shortest distance between vessels and the recording position.

    Args:
        df (DataFrame): DataFrame containing vessel data (latitude, longitude, MMSI, etc.).
        record_pos (tuple): The recording position (latitude, longitude).
        record_depth (float): The depth of the recording device.

    Returns:
        list: List of dictionaries containing information on the shortest distances between vessels and the recording position.

    Raises:
        ValueError: If 'length' or 'width' columns are not available in the dataframe.
    """
    # 最初にカラムの存在確認
    required_columns = ["length", "width"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Required columns missing from dataframe: {missing_columns}")

    distances = []
    for mmsi in df["mmsi"].unique():
        vessel_df = df[df["mmsi"] == mmsi]
        vessel_pos = vessel_df[["latitude", "longitude"]].values
        vessel_type = vessel_df["vessel_type"].values[0]
        vessel_name = vessel_df["vessel_name"].values[0]
        # Get length and width
        length = vessel_df["length"].values[0]
        width = vessel_df["width"].values[0]

        record_pos_arr = np.tile(record_pos, (len(vessel_pos), 1))
        dist = np.array(
            [
                haversine(coord1[0], coord1[1], coord2[0], coord2[1])
                for coord1, coord2 in zip(vessel_pos, record_pos_arr)
            ]
        )
        min_dist_idx = np.argmin(dist)
        min_dist = np.sqrt(dist[min_dist_idx] ** 2 + record_depth**2)
        min_dist_pos = vessel_pos[min_dist_idx]
        min_dist_time = vessel_df.iloc[min_dist_idx]["dt_pos_utc"]
        distances.append(
            {
                "mmsi": mmsi,
                "vessel_name": vessel_name,
                "vessel_type": vessel_type,
                "length": length,
                "width": width,
                "min_distance_idx": min_dist_idx,
                "min_distance [m]": min_dist,
                "min_distance_pos": min_dist_pos,
                "min_distance_time": min_dist_time,
            }
        )
    return distances
