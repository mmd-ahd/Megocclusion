import os
import numpy as np
import mne
import itertools
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from mne.decoding import SlidingEstimator, cross_val_multiscore, Vectorizer
import matplotlib.pyplot as plt

dataset_dir = 'Megocclusion'
derivatives_dir = os.path.join(dataset_dir, 'derivatives')
inv_op_dir = os.path.join(derivatives_dir, 'inv_op')
decoding_dir = os.path.join(derivatives_dir, 'decoding_vertex_svm_lin_0.1')
os.makedirs(decoding_dir, exist_ok=True)

sub_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15]
occlusion_levels = ['0', '60', '80']
categories = ['camel', 'deer', 'car', 'motor']
category_pairs = list(itertools.combinations(categories, 2))
lambda2 = 1.0 / 3.0 ** 2
method = "MNE"
cv_folds = 8
win_len = 0.05
win_step = 1/250
scoring = 'roc_auc'

merged_rois = {
    'V1': ['V1'], 'EVC': ['V2', 'V3', 'V4'], 
    'LOC': ['LO1', 'LO2', 'LO3', 'V4t', 'PIT'],
    'TE': ['TE1a', 'TE1p', 'TE1m', 'TE2a', 'TE2p'], 
    'DVS': ['V6', 'V6A', 'V7',  'V3A', 'V3B', 'IPS1'],
    'IPL': ['PFm', 'PGi', 'PGp', 'PGs', 'IP0', 'IP1', 'IP2', 'V3CD'],
    'PTVC': ['MT', 'MST', 'FST', 'PH', 'PHT'],
    'Oculomotor': ['FEF', 'PEF', '55b']
}

subjects_dir = mne.datasets.sample.data_path() / 'subjects'
mne.datasets.fetch_hcp_mmp_parcellation(subjects_dir=subjects_dir, accept=True)
all_labels = mne.read_labels_from_annot('fsaverage', parc='HCPMMP1', subjects_dir=subjects_dir)

merged_labels = {}
for new_name, parts in merged_rois.items():
    for hemi in ['lh', 'rh']:
        labels_to_merge = [lbl for lbl in all_labels if any(f'_{p}_' in lbl.name for p in parts) and lbl.hemi == hemi]
        if labels_to_merge:
            merged_label = labels_to_merge[0]
            for lbl in labels_to_merge[1:]:
                merged_label += lbl
            final_name = f'{new_name}-{hemi}'
            merged_label.name = final_name
            merged_labels[final_name] = merged_label

for sub_id in sub_ids:
    print(f'Subject: {sub_id}')
    sub_bids = f'sub-{sub_id:02d}'
    sub_inv_op = os.path.join(inv_op_dir, sub_bids, f'{sub_bids}-inv.fif')
    sub_decoding_dir = os.path.join(decoding_dir, sub_bids)
    os.makedirs(sub_decoding_dir, exist_ok=True)
    sub_epochs_dir = os.path.join(dataset_dir, sub_bids, 'meg')
    inv_op = mne.minimum_norm.read_inverse_operator(sub_inv_op)

    for level in occlusion_levels:
        print(f'Occlusion Level: {level}')
        sub_level_dir = os.path.join(sub_decoding_dir, level)
        os.makedirs(sub_level_dir, exist_ok=True)
        
        for roi in merged_labels.keys():
            print(f'     ROI: {roi}')
            subject_pair_scores = []
            
            for cat_1, cat_2 in category_pairs:
                print(f'       Processing pair: {cat_1} vs {cat_2}')
                
                epochs_1 = mne.read_epochs(os.path.join(sub_epochs_dir, f'{sub_bids}_task-objectrecognition_occlusion-{level}-{cat_1}_meg.fif'))
                epochs_2 = mne.read_epochs(os.path.join(sub_epochs_dir, f'{sub_bids}_task-objectrecognition_occlusion-{level}-{cat_2}_meg.fif'))
                y = np.concatenate((np.zeros(len(epochs_1)), np.ones(len(epochs_2))))
                epochs = mne.concatenate_epochs([epochs_1, epochs_2])
                
                roi_label = merged_labels[roi]
                stcs = mne.minimum_norm.apply_inverse_epochs(epochs, inv_op, lambda2, method=method, label=roi_label,
                                                             pick_ori='normal', return_generator=False, verbose=False)
                
                for stc in stcs:
                    stc.apply_baseline(baseline=(None, 0), verbose=False).resample(sfreq=250)
                    
                X = np.array([stc.data for stc in stcs])
                n_stcs, n_vertices, n_times = X.shape
                win_size = int(win_len * 250); step_size = int(win_step * 250)
                
                X_windows = [X[:, :, start:start + win_size] for start in range(0, n_times - win_size + 1, step_size)]
                X_sliding = np.stack(X_windows, axis=-1)
                
                pipeline = make_pipeline(Vectorizer(),
                                         StandardScaler(),
                                         SVC(kernel='linear', C=0.1, random_state=715))
                sliding = SlidingEstimator(pipeline, scoring=scoring)
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=715)
                
                scores = cross_val_multiscore(sliding, X_sliding, y, cv=cv)
                scores_mean = scores.mean(axis=0)
                subject_pair_scores.append(scores_mean)
                
                times = stcs[0].times
                win_starts = np.arange(0, n_times - win_size + 1, step_size)
                time_centers = times[win_starts + win_size // 2]
                

                np.save(os.path.join(decoding_dir, 'time_centers.npy'), time_centers)


                sub_pair_dir = os.path.join(sub_level_dir, f'{cat_1}_{cat_2}')
                os.makedirs(sub_pair_dir, exist_ok=True)
                np.save(os.path.join(sub_pair_dir, f'scores_{roi}.npy'), scores_mean)
                
                fig, ax = plt.subplots(1, dpi=150)
                ax.plot(time_centers, scores_mean, label='score')
                ax.axhline(0.5, color='k', linestyle='--', label='chance')
                ax.set_xlabel('Time (s)'); ax.set_ylabel('ROC AUC')
                ax.legend(); ax.set_title(f'Sub {sub_id} | {level}% | {roi}\n{cat_1} vs {cat_2}')
                plt.tight_layout()
                fig.savefig(os.path.join(sub_pair_dir, f'plot_{roi}.png'))
                plt.close(fig)

            subject_avg_scores = np.mean(subject_pair_scores, axis=0)
            np.save(os.path.join(sub_level_dir, f'scores_{roi}_avg-pairs.npy'), subject_avg_scores)

            fig, ax = plt.subplots(1, dpi=150)
            ax.plot(time_centers, subject_avg_scores, label='score')
            ax.axhline(0.5, color='k', linestyle='--', label='chance')
            ax.set_xlabel('Time (s)'); ax.set_ylabel('ROC AUC')
            ax.legend(); ax.set_title(f'Sub {sub_id} | {level}% | {roi}\n(Averaged over pairs)')
            plt.tight_layout()
            fig.savefig(os.path.join(sub_level_dir, f'plot_{roi}_avg-pairs.png'))
            plt.close(fig)
