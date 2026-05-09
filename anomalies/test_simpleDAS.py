"""
Example use of simpleDASreader.

The objective of this file is to illustrate basic reading and investigation
of OptoDAS measurements.
"""

import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.dates as mdates
import scipy.signal as sps

import simpledas


# %% Find input files from experiment input folder and time interval

input_folder = './data'

start = datetime.datetime(2025, 8, 6, 11, 35, 25)
duration = datetime.timedelta(seconds=30)

# Request a subset of channels. The function below will inspect the file and
# determine which channels exist in the file, returned as the chIndex variable
channels = np.arange(300, 450)

file_names, chIndex, samples = simpledas.find_DAS_files(
    input_folder, start, duration, load_file_from_start=False
)
print(f"Found {len(file_names)} files with {len(chIndex)  if chIndex else 0} channels and {samples} samples per channel.")

# %% Load the data files for the channels requested and found
chIndex = slice(300,400) # slice is the array slicing operator
samples = slice(0,1000)
dfdas = simpledas.load_DAS_files('data/Strain_DS_2025-08-06_11-30-26_UTC.hdf5', chIndex, samples)
# %% Show the first five columns to check the names of the series and data type

print(dfdas.head(5))

# %% Plot the data magntitude for time and channel

dt = dfdas.meta['dt']
Nt = len(dfdas)
plt.figure(1, clear=True)
plt.imshow(
    np.abs(dfdas),
    norm=colors.LogNorm(vmin=1e-9),
    extent=[dfdas.columns[0], dfdas.columns[-1], len(dfdas), 0],
)
plt.colorbar()

# %% Compare the time series data for few channels
plt.figure(2, clear=True)
for ch in range(7500, 9000, 500):
    plt.plot(dfdas[ch], label=str(ch))
plt.legend()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%M:%S"))
# Alternative

plt.figure(2, clear=True)
ax = plt.gca()
chs = np.arange(7500, 9000, 500)
plt.plot(dfdas.loc[: start + datetime.timedelta(seconds=3.0), chs])
plt.legend(['Ch %d' % ch for ch in chs])
plt.xlabel('Time')
plt.ylabel('Signal [%s]' % dfdas.meta['unit'])
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%M:%S"))

# %% Compare the PSD for the same channels

plt.figure(3, clear=True)
for ch in range(7500, 9000, 500):
    f, Pxx = sps.welch(dfdas[ch], fs=1 / dfdas.meta['dt'], axis=0)
    plt.loglog(f, np.sqrt(Pxx), label=f'Ch {ch}')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude spectral density [%s/√Hz]' % dfdas.meta['unit'])
plt.legend()
plt.tight_layout()
# %% Export the first 3 seconds of channel data to a csv file

dfdas[: start + datetime.timedelta(seconds=3.0)].to_csv('das_data.csv')