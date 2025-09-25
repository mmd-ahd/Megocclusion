import os
import mne
import numpy as np

dataset_dir = 'Megocclusion'
derivatives_dir = os.path.join(dataset_dir, 'derivatives')
inv_op_dir = os.path.join(derivatives_dir, 'inv_op')
stc_dir = os.path.join(derivatives_dir, 'evoked_stc_HCP_merged')
os.makedirs(stc_dir, exist_ok=True)

sub_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15]
occlusion_levels = ['0', '60', '80']
lambda2 = 1.0 / 3 ** 2
method = "MNE"

merged_rois = {
    'V1': ['V1'],
    'EVC': ['V2', 'V3', 'V4', 'V3CD'],
    'LO': ['LO1', 'LO2', 'LO3', 'V4t'],
    'PIT': ['PIT'],
    'TE': ['TE1a', 'TE1p', 'TE2p'],
    'PFC': ['45', '47m', '47l']
}

subjects_dir = mne.datasets.sample.data_path() / 'subjects'
parcellation = mne.datasets.fetch_hcp_mmp_parcellation(
    subjects_dir=subjects_dir, accept=True
)
all_labels = mne.read_labels_from_annot(
    'fsaverage', parc='HCPMMP1', subjects_dir=subjects_dir
)

merged_labels = []
for new_name, parts in merged_rois.items():
    lh_labels_to_merge = [lbl for lbl in all_labels if any(f'_{p}_' in lbl.name for p in parts) and lbl.hemi == 'lh']
    rh_labels_to_merge = [lbl for lbl in all_labels if any(f'_{p}_' in lbl.name for p in parts) and lbl.hemi == 'rh']

    if lh_labels_to_merge:
        lh_merged = lh_labels_to_merge[0]
        for lbl in lh_labels_to_merge[1:]:
            lh_merged += lbl
        lh_merged.name = f'{new_name}-lh'
        merged_labels.append(lh_merged)

    if rh_labels_to_merge:
        rh_merged = rh_labels_to_merge[0]
        for lbl in rh_labels_to_merge[1:]:
            rh_merged += lbl
        rh_merged.name = f'{new_name}-rh'
        merged_labels.append(rh_merged)

merged_label_names = [label.name for label in merged_labels]
print(f"Created {len(merged_labels)} merged labels: {merged_label_names}")

for sub in sub_ids:
    sub_bids_id = f'sub-{sub:02d}'
    print(f"Processing {sub_bids_id}")

    sub_epochs_dir = os.path.join(dataset_dir, sub_bids_id, 'meg')
    sub_inv_op_fname = os.path.join(inv_op_dir, sub_bids_id, f'{sub_bids_id}-inv.fif')
    sub_stc_dir = os.path.join(stc_dir, sub_bids_id)
    os.makedirs(sub_stc_dir, exist_ok=True)

    try:
        inv_op = mne.minimum_norm.read_inverse_operator(sub_inv_op_fname)
    except FileNotFoundError:
        print(f"Inverse operator not found for {sub_bids_id}, skipping.")
        continue

    for level in occlusion_levels:
        epochs_fname = os.path.join(sub_epochs_dir, f'{sub_bids_id}_task-objectrecognition_occlusion-{level}_meg.fif')

        if not os.path.isfile(epochs_fname):
            print(f"Epochs file not found for occlusion level {level}, skipping.")
            continue

        epochs = mne.read_epochs(epochs_fname, preload=True)
        evoked = epochs.average()

        stc = mne.minimum_norm.apply_inverse(
            evoked, inv_op, lambda2, method=method
        )
        stc.apply_baseline(baseline=(None, 0))

        label_ts = mne.extract_label_time_course(
            stc, merged_labels, inv_op['src'], mode='pca_flip', return_generator=False
        )

        for i in range(len(label_ts)):
            peak_idx = np.argmax(np.abs(label_ts[i, :]))
            if label_ts[i, peak_idx] < 0:
                label_ts[i, :] *= -1

        info = mne.create_info(ch_names=merged_label_names, sfreq=epochs.info['sfreq'], ch_types='misc')
        roi_evoked = mne.EvokedArray(np.array(label_ts), info, tmin=epochs.tmin)

        output_fname = os.path.join(sub_stc_dir, f'{sub_bids_id}_occlusion-{level}_merged-rois-evoked-ts.fif')
        roi_evoked.save(output_fname, overwrite=True)

