import numpy as np
import xdas

def find_second_crossings(data):
    """
    Finds the indices of the data that correspond to the crossing of a second.
    """
    times = data.time.values
    # Get the second component of the timestamps
    seconds = times.astype('datetime64[s]')
    # Find where the second changes
    diffs = np.diff(seconds)
    # Get the indices where the second changes
    crossings = np.where(diffs > np.timedelta64(0, 's'))[0] + 1
    return crossings

def run_test(data):
    # Test the find_second_crossings function
    subset = data.sel(
        time=slice("2025-08-06T11:35:00", "2025-08-06T11:36:00.0"),
        distance=slice(300, 450)
    )
    crossings = find_second_crossings(subset)
    print(f"Found {len(crossings)} second crossings at indices: {crossings}")
    if len(crossings) > 0:
        print("Timestamps of crossings:")
        print(subset.time.values[crossings])
