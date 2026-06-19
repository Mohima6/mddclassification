import os
import numpy as np
import pandas as pd
from scipy.signal import hilbert
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score)
from xgboost import XGBClassifier
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------
#  Paths (adjust to your folder structure)
# ----------------------------------------------------------------------
DATASET_PATH = r"C:\Users\mohimaCHAKRABORTY\mddclassify"
CLEAN_EPOCHS_DIR = os.path.join(DATASET_PATH, "processed_data_epochs")
BAND_DECOMP_DIR = os.path.join(DATASET_PATH, "band_decomposition")
OUTPUT_RESULTS = os.path.join(DATASET_PATH, "ml_results")
os.makedirs(OUTPUT_RESULTS, exist_ok=True)

FREQUENCY_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 22),
    "gamma": (22, 30)
}


# ----------------------------------------------------------------------
#  Connectivity computation functions (same as before)
# ----------------------------------------------------------------------
def compute_plv_pli_wpli(signal):
    n_ch, n_t = signal.shape
    analytic = hilbert(signal, axis=-1)
    phase = np.angle(analytic)
    plv = np.zeros((n_ch, n_ch))
    pli = np.zeros((n_ch, n_ch))
    wpli = np.zeros((n_ch, n_ch))
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            dphi = phase[i] - phase[j]
            sin_dphi = np.sin(dphi)
            plv[i, j] = plv[j, i] = np.abs(np.mean(np.exp(1j * dphi)))
            pli[i, j] = pli[j, i] = np.abs(np.mean(np.sign(sin_dphi)))
            imag_cross = np.imag(analytic[i] * np.conj(analytic[j]))
            abs_imag = np.abs(imag_cross)
            if np.sum(abs_imag) > 0:
                wpli[i, j] = wpli[j, i] = np.abs(np.mean(abs_imag * np.sign(imag_cross))) / np.mean(abs_imag)
            else:
                wpli[i, j] = wpli[j, i] = 0.0
    return plv, pli, wpli


def average_connectivity_over_epochs(epochs_data):
    n_epochs = epochs_data.shape[0]
    plv_sum = pli_sum = wpli_sum = None
    for ep in range(n_epochs):
        plv, pli, wpli = compute_plv_pli_wpli(epochs_data[ep])
        if plv_sum is None:
            plv_sum, pli_sum, wpli_sum = plv, pli, wpli
        else:
            plv_sum += plv
            pli_sum += pli
            wpli_sum += wpli
    return plv_sum / n_epochs, pli_sum / n_epochs, wpli_sum / n_epochs


# ----------------------------------------------------------------------
#  Identify most common channel count (across subjects)
# ----------------------------------------------------------------------
def get_subject_channel_count(sub):
    """Read a single band file to determine number of channels."""
    sample_band = list(FREQUENCY_BANDS.keys())[0]
    band_file = os.path.join(BAND_DECOMP_DIR, sample_band, f"{sub}_{sample_band}.npy")
    if not os.path.exists(band_file):
        return None
    data = np.load(band_file)
    if data.ndim == 3:
        return data.shape[1]  # (epochs, channels, time)
    else:
        return None


# Gather all subjects
epoch_files = [f for f in os.listdir(CLEAN_EPOCHS_DIR) if f.endswith('_epochs.npy')]
all_subjects = [f.replace('_epochs.npy', '') for f in epoch_files]

# Load labels
labels = {}
for sub in all_subjects:
    label_file = os.path.join(CLEAN_EPOCHS_DIR, f"{sub}_labels.npy")
    if os.path.exists(label_file):
        labels[sub] = int(np.load(label_file)[0])
    else:
        if sub.lower().startswith('h'):
            labels[sub] = 0
        elif sub.lower().startswith('mdd'):
            labels[sub] = 1
        else:
            labels[sub] = None
all_subjects = [s for s in all_subjects if labels.get(s) is not None]
print(f"Total subjects with labels: {len(all_subjects)}")

# Count channels per subject (by reading a single band file)
chan_counts = {}
for sub in tqdm(all_subjects, desc="Counting channels"):
    n_ch = get_subject_channel_count(sub)
    if n_ch is not None:
        chan_counts[sub] = n_ch

# Find most common channel count
count_freq = {}
for n_ch in chan_counts.values():
    count_freq[n_ch] = count_freq.get(n_ch, 0) + 1
most_common_ch = max(count_freq, key=count_freq.get)
print(f"Most common channel count: {most_common_ch} (appears in {count_freq[most_common_ch]} subjects)")

# Keep only subjects with that channel count
subjects = [s for s, n_ch in chan_counts.items() if n_ch == most_common_ch]
print(f"Retained {len(subjects)} subjects with consistent channel count.\n")

# y vector (same for all)
y = np.array([labels[s] for s in subjects])

# ----------------------------------------------------------------------
#  Compute connectivity features for each subject and band
# ----------------------------------------------------------------------
features = {'plv': {}, 'pli': {}, 'wpli': {}}
for band in FREQUENCY_BANDS:
    for metric in features.keys():
        features[metric][band] = []

for sub in tqdm(subjects, desc="Computing connectivity"):
    for band in FREQUENCY_BANDS:
        band_file = os.path.join(BAND_DECOMP_DIR, band, f"{sub}_{band}.npy")
        if not os.path.exists(band_file):
            print(f"Missing {band_file}, skipping subject {sub} for band {band}")
            continue
        epochs = np.load(band_file)
        if epochs.ndim != 3:
            continue
        plv_avg, pli_avg, wpli_avg = average_connectivity_over_epochs(epochs)
        n_ch = plv_avg.shape[0]
        triu_indices = np.triu_indices(n_ch, k=1)
        for metric, mat in zip(['plv', 'pli', 'wpli'], [plv_avg, pli_avg, wpli_avg]):
            vec = mat[triu_indices]
            features[metric][band].append(vec)

# Convert to numpy arrays
for metric in features:
    for band in FREQUENCY_BANDS:
        if features[metric][band]:
            X_list = features[metric][band]
            # All vectors should have same length now
            features[metric][band] = (np.array(X_list), y)
        else:
            features[metric][band] = (None, y)

# ----------------------------------------------------------------------
#  Classifiers and hyperparameter grids
# ----------------------------------------------------------------------
classifiers = {
    'LogisticRegression': {
        'model': LogisticRegression(random_state=42, max_iter=1000),
        'param_grid': {'C': [0.01, 0.1, 1, 10, 100]}
    },
    'DecisionTree': {
        'model': DecisionTreeClassifier(random_state=42),
        'param_grid': {'max_depth': [3, 5, 7, 10], 'min_samples_split': [2, 5, 10]}
    },
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42),
        'param_grid': {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, None],
                       'min_samples_split': [2, 5]}
    },
    'SVM': {
        'model': SVC(random_state=42, probability=True),
        'param_grid': {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto']}
    },
    'XGBoost': {
        'model': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
        'param_grid': {'n_estimators': [50, 100], 'max_depth': [3, 5],
                       'learning_rate': [0.01, 0.1], 'subsample': [0.8, 1.0]}
    }
}


# ----------------------------------------------------------------------
#  Nested cross‑validation (with clear variable names)
# ----------------------------------------------------------------------
def nested_cv(X, y, clf_dict, n_outer=5, n_inner=3, random_state=42):
    outer_split = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=random_state)

    y_true_all = []
    y_pred_all = []
    y_prob_all = []

    for train_idx, test_idx in outer_split.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Inner CV for hyperparameter tuning
        inner_cv = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=random_state)
        gs = GridSearchCV(clf_dict['model'], clf_dict['param_grid'],
                          cv=inner_cv, scoring='accuracy', n_jobs=-1)
        gs.fit(X_train_scaled, y_train)
        best_model = gs.best_estimator_

        y_pred = best_model.predict(X_test_scaled)
        y_prob = best_model.predict_proba(X_test_scaled)[:, 1] if hasattr(best_model, "predict_proba") else None

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
        if y_prob is not None:
            y_prob_all.extend(y_prob)

    y_true = np.array(y_true_all)
    y_pred = np.array(y_pred_all)
    acc = accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred, pos_label=1)
    spec = recall_score(y_true, y_pred, pos_label=0)
    prec = precision_score(y_true, y_pred, pos_label=1)
    f1 = f1_score(y_true, y_pred, pos_label=1)
    auc = roc_auc_score(y_true, y_prob_all) if y_prob_all else np.nan

    return {'accuracy': acc, 'sensitivity': sens, 'specificity': spec,
            'precision': prec, 'f1': f1, 'roc_auc': auc}


# ----------------------------------------------------------------------
#  Run evaluation for each metric and band
# ----------------------------------------------------------------------
results = {'plv': {}, 'pli': {}, 'wpli': {}}
for metric in results.keys():
    for band in FREQUENCY_BANDS:
        X, y_band = features[metric][band]
        if X is None:
            print(f"Skipping {metric} {band} – no data.")
            continue
        print(f"\nRunning nested CV for {metric} {band} with {X.shape[0]} subjects, {X.shape[1]} features")
        results[metric][band] = {}
        for clf_name, clf_dict in classifiers.items():
            print(f"  Classifier: {clf_name}")
            res = nested_cv(X, y_band, clf_dict)
            results[metric][band][clf_name] = res

# ----------------------------------------------------------------------
#  Save results as tab‑separated tables (one per performance metric)
# ----------------------------------------------------------------------
metrics_list = ['accuracy', 'sensitivity', 'specificity', 'precision', 'f1', 'roc_auc']
band_names = list(FREQUENCY_BANDS.keys())
classifier_names = list(classifiers.keys())

for metric in ['plv', 'pli', 'wpli']:
    metric_dir = os.path.join(OUTPUT_RESULTS, metric)
    os.makedirs(metric_dir, exist_ok=True)

    for perf in metrics_list:
        data = {}
        for band in band_names:
            if band not in results[metric]:
                continue
            band_res = results[metric][band]
            col_vals = []
            for clf in classifier_names:
                if clf in band_res and perf in band_res[clf]:
                    col_vals.append(band_res[clf][perf])
                else:
                    col_vals.append(np.nan)
            data[band] = col_vals
        df = pd.DataFrame(data, index=classifier_names)
        out_file = os.path.join(metric_dir, f"{perf}.txt")
        df.to_csv(out_file, sep='\t', float_format='%.4f')
        print(f"Saved {out_file}")

print("\nAll ML results saved. ")