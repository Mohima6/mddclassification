import os
import numpy as np
import mne
from tqdm import tqdm
from scipy.stats import zscore
DATASET_PATH = r"C:\Users\mohimaCHAKRABORTY\mddclassify"
OUTPUT_FOLDER = os.path.join(DATASET_PATH, "processed_data_epochs")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
LOW_FREQ = 0.1
HIGH_FREQ = 70
NOTCH_FREQ = 50
EPOCH_LENGTH = 5
OVERLAP = 0.5
edf_files = []
for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        if file.lower().endswith(".edf"):
            edf_files.append(os.path.join(root, file))
print(f"Found {len(edf_files)} EDF files\n")
def apply_ica(raw, n_components=0.99, random_state=42):
    """Fit FastICA and remove EOG artifacts if an EOG channel exists."""
    eog_channels = [ch for ch in raw.ch_names if 'eog' in ch.lower() or 'eye' in ch.lower()]
    if not eog_channels:
        print("   No EOG channel found. ICA will be fitted but no components will be excluded.")
    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method='fastica',
        random_state=random_state,
        max_iter='auto'
    )
    ica.fit(raw)
    if eog_channels:
        eog_idx, scores = ica.find_bads_eog(raw, ch_name=eog_channels[0])
        ica.exclude = eog_idx
        print(f"   Excluding {len(eog_idx)} EOG components")
    else:
        ica.exclude = []
    cleaned = raw.copy()
    ica.apply(cleaned)
    return cleaned
def preprocess_file(file_path):
    file_name = os.path.basename(file_path)
    print(f"\nProcessing: {file_name}")
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    raw.pick('eeg')
    print(f"   EEG channels: {len(raw.ch_names)}")
    raw.filter(LOW_FREQ, HIGH_FREQ, fir_design='firwin', verbose=False)
    raw.notch_filter(NOTCH_FREQ, verbose=False)
    raw = apply_ica(raw)
    step = EPOCH_LENGTH * (1 - OVERLAP)
    overlap = EPOCH_LENGTH - step
    epochs = mne.make_fixed_length_epochs(
        raw,
        duration=EPOCH_LENGTH,
        overlap=overlap,
        preload=True,
        verbose=False
    )
    epoch_data = epochs.get_data()
    print(f"   Epochs generated: {epoch_data.shape[0]}")
    epoch_data = zscore(epoch_data, axis=-1)
    epoch_data = np.nan_to_num(epoch_data)
    base_name = os.path.splitext(file_name)[0].lower()
    if base_name.startswith('h'):
        label = 0
    elif base_name.startswith('mdd'):
        label = 1
    else:
        print(f"   Unknown label in filename: {file_name}, skipping.")
        return
    labels = np.full(len(epoch_data), label, dtype=np.uint8)
    epoch_data = epoch_data.astype(np.float16)
    out_prefix = os.path.join(OUTPUT_FOLDER, os.path.splitext(file_name)[0])
    np.save(f"{out_prefix}_epochs.npy", epoch_data)
    np.save(f"{out_prefix}_labels.npy", labels)
    print(f"   Saved: {out_prefix}_epochs.npy (shape {epoch_data.shape}, dtype float16)")
    print(f"   Saved: {out_prefix}_labels.npy (shape {labels.shape}, dtype uint8)")
for fpath in tqdm(edf_files, desc="Overall progress"):
    preprocess_file(fpath)
print(f"\n All processing finished. Clean epochs saved in:\n {OUTPUT_FOLDER}")