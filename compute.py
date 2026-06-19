import os
import numpy as np
from scipy.signal import hilbert
from scipy.stats import shapiro, ttest_ind, mannwhitneyu
from statsmodels.stats.multitest import fdrcorrection
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
DATASET_PATH = r"C:\Users\mohimaCHAKRABORTY\mddclassify"
CLEAN_EPOCHS_DIR = os.path.join(DATASET_PATH, "processed_data_epochs")
BAND_DECOMP_DIR = os.path.join(DATASET_PATH, "band_decomposition")
OUTPUT_RESULTS = os.path.join(DATASET_PATH, "connectivity_results")
os.makedirs(OUTPUT_RESULTS, exist_ok=True)
FREQUENCY_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 22),
    "gamma": (22, 30)
}
def compute_plv_pli_wpli(signal):
    """
    Compute PLV, PLI, and wPLI for a multi‑channel EEG segment.

    Parameters
    ----------
    signal : np.ndarray, shape (n_channels, n_times)
        Band‑passed EEG data (one epoch).

    Returns
    -------
    plv : np.ndarray, shape (n_channels, n_channels)
        Phase Locking Value (symmetric, diagonal = 0)
    pli : np.ndarray, shape (n_channels, n_channels)
        Phase Lag Index (symmetric, diagonal = 0)
    wpli : np.ndarray, shape (n_channels, n_channels)
        weighted Phase Lag Index (symmetric, diagonal = 0)
    """
    n_ch, n_t = signal.shape
    # Analytic signal via Hilbert transform
    analytic = hilbert(signal, axis=-1)
    phase = np.angle(analytic)

    plv = np.zeros((n_ch, n_ch))
    pli = np.zeros((n_ch, n_ch))
    wpli = np.zeros((n_ch, n_ch))

    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            # phase difference
            dphi = phase[i] - phase[j]
            sin_dphi = np.sin(dphi)

            plv_val = np.abs(np.mean(np.exp(1j * dphi)))

            pli_val = np.abs(np.mean(np.sign(sin_dphi)))

            # wPLI
            imag_cross = np.imag(analytic[i] * np.conj(analytic[j]))
            abs_imag = np.abs(imag_cross)
            if np.sum(abs_imag) > 0:
                wpli_val = np.abs(np.mean(abs_imag * np.sign(imag_cross))) / np.mean(abs_imag)
            else:
                wpli_val = 0.0

            plv[i, j] = plv[j, i] = plv_val
            pli[i, j] = pli[j, i] = pli_val
            wpli[i, j] = wpli[j, i] = wpli_val

    return plv, pli, wpli

def average_connectivity_over_epochs(epochs_data):
    """
    Compute connectivity matrices for each epoch and average.
    epochs_data : np.ndarray, shape (n_epochs, n_channels, n_times)
    Returns average PLV, PLI, wPLI matrices (n_channels, n_channels).
    """
    n_epochs = epochs_data.shape[0]
    plv_sum = None
    pli_sum = None
    wpli_sum = None

    for ep in range(n_epochs):
        plv, pli, wpli = compute_plv_pli_wpli(epochs_data[ep])
        if plv_sum is None:
            plv_sum = plv
            pli_sum = pli
            wpli_sum = wpli
        else:
            plv_sum += plv
            pli_sum += pli
            wpli_sum += wpli

    n_epochs = float(n_epochs)
    return plv_sum / n_epochs, pli_sum / n_epochs, wpli_sum / n_epochs

epoch_files = [f for f in os.listdir(CLEAN_EPOCHS_DIR) if f.endswith('_epochs.npy')]
subjects = [f.replace('_epochs.npy', '') for f in epoch_files]
print(f"Found {len(subjects)} subjects.")

labels = {}
for sub in subjects:
    label_file = os.path.join(CLEAN_EPOCHS_DIR, f"{sub}_labels.npy")
    if os.path.exists(label_file):
        lbl = np.load(label_file)
        labels[sub] = int(lbl[0])
    else:
        if sub.lower().startswith('h'):
            labels[sub] = 0
        elif sub.lower().startswith('mdd'):
            labels[sub] = 1
        else:
            print(f"Warning: no label for {sub}, skipping.")
            labels[sub] = None

subjects = [s for s in subjects if labels.get(s) is not None]
connectivity = {}
for band in FREQUENCY_BANDS:
    connectivity[band] = {
        'plv': {},
        'pli': {},
        'wpli': {}
    }

for sub in tqdm(subjects, desc="Processing subjects"):
    sample_band = list(FREQUENCY_BANDS.keys())[0]
    sample_file = os.path.join(BAND_DECOMP_DIR, sample_band, f"{sub}_{sample_band}.npy")
    if not os.path.exists(sample_file):
        print(f"Missing data for {sub}, skipping.")
        continue

    for band in FREQUENCY_BANDS:
        band_file = os.path.join(BAND_DECOMP_DIR, band, f"{sub}_{band}.npy")
        if not os.path.exists(band_file):
            print(f"Missing band {band} for {sub}, skipping band.")
            continue
        epochs = np.load(band_file)
        if epochs.ndim != 3:
            print(f"Unexpected shape for {sub} {band}: {epochs.shape}, skipping.")
            continue

        plv_avg, pli_avg, wpli_avg = average_connectivity_over_epochs(epochs)
        connectivity[band]['plv'][sub] = plv_avg
        connectivity[band]['pli'][sub] = pli_avg
        connectivity[band]['wpli'][sub] = wpli_avg

band0 = list(FREQUENCY_BANDS.keys())[0]
metric0 = 'plv'
channel_counts = {}
for sub in connectivity[band0][metric0]:
    n_ch = connectivity[band0][metric0][sub].shape[0]
    channel_counts[n_ch] = channel_counts.get(n_ch, 0) + 1
most_common_ch = max(channel_counts, key=channel_counts.get)
print(f"\nMost common channel count: {most_common_ch} (appears in {channel_counts[most_common_ch]} subjects)")
for band in FREQUENCY_BANDS:
    for metric in ['plv', 'pli', 'wpli']:
        connectivity[band][metric] = {
            sub: mat for sub, mat in connectivity[band][metric].items()
            if mat.shape[0] == most_common_ch
        }
retained_subjects = list(connectivity[band0][metric0].keys())
print(f"Retained {len(retained_subjects)} subjects with consistent channel count.\n")
def test_connectivity_pair(group0_vals, group1_vals, alpha=0.05):
    """
    Perform normality test, choose t-test or Mann‑Whitney, return p‑value and effect size.
    Returns: (test_name, p_raw, effect_size, is_normal)
    """
    # Shapiro‑Wilk
    normal0 = shapiro(group0_vals).pvalue > alpha if len(group0_vals) >= 3 else True
    normal1 = shapiro(group1_vals).pvalue > alpha if len(group1_vals) >= 3 else True
    both_normal = normal0 and normal1
    if both_normal:
        # Independent t‑test (Welch's)
        t_stat, p = ttest_ind(group0_vals, group1_vals, equal_var=False)
        # Cohen's d
        n0, n1 = len(group0_vals), len(group1_vals)
        var0, var1 = np.var(group0_vals, ddof=1), np.var(group1_vals, ddof=1)
        pooled_std = np.sqrt(((n0 - 1) * var0 + (n1 - 1) * var1) / (n0 + n1 - 2))
        d = (np.mean(group0_vals) - np.mean(group1_vals)) / pooled_std if pooled_std > 0 else 0.0
        return 't-test', p, d, True
    else:
        # Mann‑Whitney U
        u, p = mannwhitneyu(group0_vals, group1_vals, alternative='two-sided')
        # biserial correlation as effect size: r = |Z|/sqrt(N)
        n0, n1 = len(group0_vals), len(group1_vals)
        N = n0 + n1
        mu = n0 * n1 / 2.0
        sigma = np.sqrt(n0 * n1 * (N + 1) / 12.0)
        Z = (u - mu) / sigma if sigma > 0 else 0.0
        r = abs(Z) / np.sqrt(N)
        return 'Mann-Whitney', p, r, False
results = {}
for band in FREQUENCY_BANDS:
    results[band] = {}
    for metric in ['plv', 'pli', 'wpli']:
        print(f"\nTesting {band} - {metric}")
        sub_list = list(connectivity[band][metric].keys())
        if not sub_list:
            continue
        n_ch = connectivity[band][metric][sub_list[0]].shape[0]
        pair_indices = [(i, j) for i in range(n_ch) for j in range(i + 1, n_ch)]
        all_pairs_data = []
        for i, j in pair_indices:
            vals0 = []
            vals1 = []
            for sub in sub_list:
                val = connectivity[band][metric][sub][i, j]
                if labels[sub] == 0:
                    vals0.append(val)
                else:
                    vals1.append(val)
            all_pairs_data.append((i, j, np.array(vals0), np.array(vals1)))
        raw_p = []
        effect_sizes = []
        test_names = []
        normality_flags = []
        for i, j, v0, v1 in all_pairs_data:
            if len(v0) < 2 or len(v1) < 2:
                raw_p.append(1.0)
                effect_sizes.append(0.0)
                test_names.append('insufficient')
                normality_flags.append(False)
                continue
            test_name, p, es, normal = test_connectivity_pair(v0, v1)
            raw_p.append(p)
            effect_sizes.append(es)
            test_names.append(test_name)
            normality_flags.append(normal)

        # FDR correction (Benjamini‑Hochberg)
        reject, p_corr = fdrcorrection(raw_p, alpha=0.05)
        results[band][metric] = {
            'pair_indices': pair_indices,
            'raw_p': raw_p,
            'p_corr': p_corr,
            'reject': reject,
            'effect_sizes': effect_sizes,
            'test_names': test_names,
            'normality': normality_flags
        }
for band in FREQUENCY_BANDS:
    for metric in ['plv', 'pli', 'wpli']:
        if band not in results or metric not in results[band]:
            continue
        res = results[band][metric]
        out_file = os.path.join(OUTPUT_RESULTS, f"{band}_{metric}_stats.txt")
        with open(out_file, 'w') as f:
            f.write(f"Band: {band}, Metric: {metric}\n")
            f.write("Pair (i,j)\tRaw p\tFDR p\tReject\tTest\tEffect size\tNormal?\n")
            for idx, (i, j) in enumerate(res['pair_indices']):
                line = f"{i}-{j}\t{res['raw_p'][idx]:.6f}\t{res['p_corr'][idx]:.6f}\t"
                line += f"{res['reject'][idx]}\t{res['test_names'][idx]}\t"
                line += f"{res['effect_sizes'][idx]:.4f}\t{res['normality'][idx]}\n"
                f.write(line)
        print(f"Saved: {out_file}")

# summary: number of significant pairs per band/metric
summary_file = os.path.join(OUTPUT_RESULTS, "significant_summary.txt")
with open(summary_file, 'w') as f:
    f.write("Band\tMetric\tSignificant pairs (FDR < 0.05)\tTotal pairs\n")
    for band in FREQUENCY_BANDS:
        for metric in ['plv', 'pli', 'wpli']:
            if band in results and metric in results[band]:
                sig = np.sum(results[band][metric]['reject'])
                total = len(results[band][metric]['reject'])
                f.write(f"{band}\t{metric}\t{sig}\t{total}\n")
print(f"\nSummary saved: {summary_file}")

print("\nAll done!  'connectivity_results' output files.")