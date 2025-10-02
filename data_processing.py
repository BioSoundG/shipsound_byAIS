import pandas as pd


def read_ais(ais_data):
    """
    Reads AIS data from a CSV file and converts the datetime format.

    Args:
        ais_data (str): Path to the AIS data CSV file.

    Returns:
        DataFrame: AIS data with sorted MMSI and converted datetime column.
    """

    def convert_datetime(df):
        df["dt_pos_utc"] = pd.to_datetime(df["dt_pos_utc"])
        return df

    ais_df = pd.read_csv(ais_data)
    ais_df.sort_values(["mmsi", "dt_pos_utc"], inplace=True)
    return convert_datetime(ais_df)


def complement_trajectory(data):
    """
    Complements vessel trajectories by resampling and interpolating data to fill in missing points.

    Args:
        data (str): Path to the CSV file containing vessel data.

    Returns:
        DataFrame: Resampled and interpolated vessel data.
    """
    data = pd.read_csv(data)
    data["dt_pos_utc"] = pd.to_datetime(data["dt_pos_utc"])
    data.drop_duplicates(subset=["mmsi", "dt_pos_utc"], inplace=True)
    resampled_data_list = []

    for _, group in data.groupby("mmsi"):
        group.set_index("dt_pos_utc", inplace=True)
        resampled = group.resample("1s")
        group_resampled = resampled.interpolate().ffill()
        resampled_data_list.append(group_resampled)

    resampled_data = pd.concat(resampled_data_list).reset_index()
    resampled_data["depth"] = 0
    return resampled_data
