import numpy as np
import pandas as pd
import xarray as xr
import xdas
import os

from scipy.stats import pearsonr, spearmanr, skew, kurtosis
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.feature_selection import mutual_info_regression


hdf5_file_path = 'C:\\magisterka\\febus\\Strain*.h5'

data = xdas.open_mfdataarray(
    hdf5_file_path,
    engine="febus",
    tolerance=np.timedelta64(15, "ms"),
)

def select_data(data, start, end, dist_start, dist_end):
    return data.sel(
        time=slice(start, end),
        distance=slice(dist_start, dist_end)
    )

subset = select_data(
    data,
    start="2025-08-06T11:35:00",
    end="2025-08-06T11:45:00",
    dist_start=300,
    dist_end=305   # start with few channels
)

def get_second_boundaries(time_array):
    """
    Returns indices where timestamp crosses integer second.
    """
    times = pd.to_datetime(time_array.values)
    seconds = times.floor("S")
    mask = (times == seconds)
    return np.where(mask)[0]

def compute_window_features(window):
    """
    window: 1D numpy array (1-second data)
    Returns feature vector dictionary.
    """
    features = {}

    features["mean"] = np.mean(window)
    features["std"] = np.std(window)
    features["rms"] = np.sqrt(np.mean(window**2))
    features["skew"] = skew(window)
    features["kurtosis"] = kurtosis(window)
    features["max"] = np.max(window)
    features["min"] = np.min(window)
    features["range"] = features["max"] - features["min"]

    # Linear trend (slope)
    x = np.arange(len(window))
    coeff = np.polyfit(x, window, 1)
    features["slope"] = coeff[0]

    # Total variation
    features["total_variation"] = np.sum(np.abs(np.diff(window)))

    # Lag-1 autocorrelation
    if len(window) > 1:
        features["lag1_corr"] = np.corrcoef(window[:-1], window[1:])[0, 1]
    else:
        features["lag1_corr"] = np.nan

    return features

def build_feature_dataset(data_array):
    """
    data_array: xarray DataArray (time x distance)
    Returns dataframe with features and anomaly amplitude.
    """

    fs = 1 / np.median(np.diff(data_array.time.values).astype('timedelta64[ns]').astype(float) * 1e-9)
    samples_per_second = int(round(fs))

    results = []

    for ch in data_array.distance.values:
        signal = data_array.sel(distance=ch).values
        time_array = data_array.time

        second_indices = get_second_boundaries(time_array)

        for idx in second_indices:
            if idx < samples_per_second:
                continue  # skip first second

            window = signal[idx - samples_per_second:idx]

            anomaly_value = signal[idx]
            anomaly_jump = signal[idx] - signal[idx - 1]

            features = compute_window_features(window)
            features["anomaly_value"] = anomaly_value
            features["anomaly_jump"] = anomaly_jump
            features["channel"] = float(ch)

            results.append(features)

    df = pd.DataFrame(results)
    df = df.dropna()

    return df


df = build_feature_dataset(subset)
df.head()