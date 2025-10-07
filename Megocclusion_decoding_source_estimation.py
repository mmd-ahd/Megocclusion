import os
import mne
import numpy as np
import json

dataset_dir = 'Megocclusion'
derivatives_dir = os.path.join(dataset_dir, 'derivatives')
inv_op_dir = os.path.join(derivatives_dir, 'inv_op')
stc_dir = os.path.join(derivatives_dir, 'decoding_stcs')
os.makedirs(stc_dir, exist_ok=True)

sub_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15]
EVENT_ID_MAP = {
    '0-camel': [3], '60-camel': [4], '80-camel': [5],
    '0-deer': [9], '60-deer': [10], '80-deer': [11],
    '0-car': [15], '60-car': [16], '80-car': [17],
    '0-motor': [21], '60-motor': [22], '80-motor': [23]
}

lambda2 = 1.0 / 3.0 ** 2
method = "MNE"

merged_rois = {
    'V1': ['V1'],
    'EVC': ['V2', 'V3', 'V4'],
    'LOC': ['LO1', 'LO2', 'LO3', 'V4t', 'PIT'],
    'TE': ['TE1a', 'TE1p', 'TE1m', 'TE2a', 'TE2p'],
    'DVS': ['V6', 'V6A', 'V7',  'V3A', 'V3B', 'IPS1'],
    'IPL': ['PFm', 'PGi', 'PGp', 'PGs', 'IP0', 'IP1', 'IP2', 'V3CD'],
    'PTVC': ['MT', 'MST', 'FST', 'PH', 'PHT']
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

if not merged_labels:
    print("No merged labels were created.")
    exit()

print(f"Successfully created {len(merged_labels)} merged labels: {list(merged_labels.keys())}")

for sub in sub_ids:
    sub_bids_id = f'sub-{sub:02d}'
    print(f"\nProcessing {sub_bids_id}")

    sub_epochs_dir = os.path.join(dataset_dir, sub_bids_id, 'meg')
    sub_inv_op_fname = os.path.join(inv_op_dir, sub_bids_id, f'{sub_bids_id}-inv.fif')
    sub_stc_dir = os.path.join(stc_dir, sub_bids_id)
    os.makedirs(sub_stc_dir, exist_ok=True)


    inv_op = mne.minimum_norm.read_inverse_operator(sub_inv_op_fname)

    for level in EVENT_ID_MAP.keys():
        epochs_fname = os.path.join(sub_epochs_dir, f'{sub_bids_id}_task-objectrecognition_occlusion-{level}_meg.fif')

        if not os.path.isfile(epochs_fname):
            continue

        epochs = mne.read_epochs(epochs_fname, preload=True, verbose=False)

        for roi_name, roi_label in merged_labels.items():
            stcs = mne.minimum_norm.apply_inverse_epochs(
                epochs, inv_op, lambda2, method=method, label=roi_label,
                pick_ori='normal', return_generator=False, verbose=False
            )

            if not stcs:
                continue
            
            for stc in stcs:
                stc.apply_baseline(baseline=(None, 0), verbose=False)
                stc.resample(sfreq=250)
            
            stc_data = np.array([stc.data for stc in stcs])

            metadata = {
                'subject_id': sub_bids_id,
                'condition': level,
                'roi': roi_name,
                'sfreq': epochs.info['sfreq'],
                'tmin': epochs.tmin,
                'tmax': epochs.tmax,
                'baseline': epochs.baseline,
                'vertices': [v.tolist() for v in stcs[0].vertices],
                'method': method,
                'lambda2': lambda2
            }

            base_fname = f'{sub_bids_id}_task-objectrecognition-occlusion-{level}_roi-{roi_name}_stc'
            output_npy_fname = os.path.join(sub_stc_dir, f'{base_fname}.npy')
            output_json_fname = os.path.join(sub_stc_dir, f'{base_fname}.json')

            np.save(output_npy_fname, stc_data)
            with open(output_json_fname, 'w') as f:
                json.dump(metadata, f, indent=4)

            print(f'Saved: {output_npy_fname}')
