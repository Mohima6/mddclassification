import os
import numpy as np
import mne
from scipy.signal import hilbert
from tqdm import tqdm
DATASET_PATH = r"C:\Users\mohimaCHAKRABORTY\mddclassify"
CLEAN_EPOCHS_DIR = os.path.join(DATASET_PATH, "processed_data_epochs")
OUTPUT_BASE = os.path.join(DATASET_PATH, "band_decomposition")
FREQUENCY_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 22),
    "gamma": (22, 30)
}
for band_name in FREQUENCY_BANDS.keys():
    os.makedirs(os.path.join(OUTPUT_BASE, band_name), exist_ok=True)
epoch_files = []
for root, dirs, files in os.walk(CLEAN_EPOCHS_DIR):
    for file in files:
        if file.endswith("_epochs.npy"):
            epoch_files.append(os.path.join(root, file))
print(f"Found {len(epoch_files)} clean epoch files\n")
def bandpass_filter_epochs(epochs_data, sfreq, l_freq, h_freq):
    """
    Apply zero‑phase bandpass filter to epochs data.
    epochs_data : numpy array (n_epochs, n_channels, n_times)
    sfreq : sampling frequency (Hz)
    l_freq, h_freq : band edges (Hz)
    Returns filtered data (same shape).
    """
    n_epochs, n_channels, n_times = epochs_data.shape
    filtered = np.zeros_like(epochs_data, dtype=np.float32)
    for ep_idx in range(n_epochs):
        info = mne.create_info(ch_names=[f"ch_{i}" for i in range(n_channels)], sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(epochs_data[ep_idx], info, verbose=False)
        raw.filter(l_freq, h_freq, fir_design='firwin', verbose=False)
        filtered[ep_idx] = raw.get_data()
    return filtered
def compute_envelope(data):
    """Compute analytic amplitude (envelope) using Hilbert transform."""
    return np.abs(hilbert(data, axis=-1))
edf_files = []
for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        if file.lower().endswith(".edf"):
            edf_files.append(os.path.join(root, file))
            break
    if edf_files:
        break
if not edf_files:
    raise RuntimeError("No .edf file found to determine sampling frequency. Please set SFREQ manually.")
raw_demo = mne.io.read_raw_edf(edf_files[0], preload=False, verbose=False)
SFREQ = raw_demo.info['sfreq']
print(f"Sampling frequency: {SFREQ} Hz\n")
for epoch_file in tqdm(epoch_files, desc="Band decomposition"):
    epochs_data = np.load(epoch_file).astype(np.float64)
    print(f"\nProcessing {os.path.basename(epoch_file)}: shape {epochs_data.shape}")
    base_name = os.path.splitext(os.path.basename(epoch_file))[0].replace("_epochs", "")
    for band_name, (l_freq, h_freq) in FREQUENCY_BANDS.items():
        band_data = bandpass_filter_epochs(epochs_data, SFREQ, l_freq, h_freq)
        band_data = band_data.astype(np.float16)
        out_file = os.path.join(OUTPUT_BASE, band_name, f"{base_name}_{band_name}.npy")
        np.save(out_file, band_data)
    print(f"   Saved 5 band files for {base_name}")
print(f"\n All band decomposition completed. Results saved in:\n {OUTPUT_BASE}")