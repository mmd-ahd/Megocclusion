import os
import mne
import numpy as np
from scipy.spatial.distance import pdist

dataset_dir = 'Megocclusion'
derivatives_dir = os.path.join(dataset_dir, 'derivatives')
inv_op_dir = os.path.join(derivatives_dir, 'inv_op')

rdm_dir = os.path.join(derivatives_dir, 'RDMs')
os.makedirs(rdm_dir, exist_ok=True)

sub_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15]
occlusion_levels = ['0', '60', '80']
categories = ['camel', 'deer', 'car', 'motor']

lambda2 = 1.0 / 3.0 ** 2
method = "MNE"

rois_to_merge = {
    'V1-3': ['V1', 'V2', 'V3'],
    'LOC': ['LO1', 'LO2', 'LO3', 'V4t', 'PIT']
}

subjects_dir = mne.datasets.sample.data_path() / 'subjects'
mne.datasets.fetch_hcp_mmp_parcellation(subjects_dir=subjects_dir, accept=True)
all_labels = mne.read_labels_from_annot('fsaverage', parc='HCPMMP1', subjects_dir=subjects_dir)

merged_labels = {}
for new_name, parts in rois_to_merge.items():
    labels_to_merge_bihemi = []
    for hemi in ['lh', 'rh']:
        labels_to_merge_hemi = [lbl for lbl in all_labels if any(f'_{p}_' in lbl.name for p in parts) and lbl.hemi == hemi]
        
        if labels_to_merge_hemi:
            merged_label_hemi = labels_to_merge_hemi[0]
            for lbl in labels_to_merge_hemi[1:]:
                merged_label_hemi += lbl
            labels_to_merge_bihemi.append(merged_label_hemi)
    
    if labels_to_merge_bihemi:
        final_merged_label = labels_to_merge_bihemi[0]
        if len(labels_to_merge_bihemi) > 1:
            for lbl in labels_to_merge_bihemi[1:]:
                final_merged_label += lbl
        
        final_merged_label.name = new_name
        merged_labels[new_name] = final_merged_label

for sub in sub_ids:
    sub_bids_id = f'sub-{sub:02d}'
    print(f"\nProcessing {sub_bids_id}...")

    sub_epochs_dir = os.path.join(dataset_dir, sub_bids_id, 'meg')
    sub_inv_op_fname = os.path.join(inv_op_dir, sub_bids_id, f'{sub_bids_id}-inv.fif')
    sub_rdm_dir = os.path.join(rdm_dir, sub_bids_id)
    os.makedirs(sub_rdm_dir, exist_ok=True)

    try:
        inv_op = mne.minimum_norm.read_inverse_operator(sub_inv_op_fname)
    except FileNotFoundError:
        print(f"  Inverse operator not found, skipping {sub_bids_id}.")
        continue

    for level in occlusion_levels:
        
        stcs_list = []
        all_files_found = True
        for cat in categories:
            epochs_fname = os.path.join(
                sub_epochs_dir,
                f'{sub_bids_id}_task-objectrecognition_occlusion-{level}-{cat}_meg.fif'
            )
            
            if not os.path.isfile(epochs_fname):
                print(f"    Epochs file not found, skipping: {epochs_fname}")
                all_files_found = False
                break
            
            epochs = mne.read_epochs(epochs_fname, preload=True, verbose=False)
            evoked = epochs.average()
            stc = mne.minimum_norm.apply_inverse(
                evoked, inv_op, lambda2, method=method, verbose=False
            )
            stc.apply_baseline(baseline=(None, 0))
            stcs_list.append(stc)
            
        times = stcs_list[0].times

        for roi_name, label_obj in merged_labels.items():

            label_data_list = [stc.in_label(label_obj).data for stc in stcs_list]

            n_conditions = len(label_data_list)
            n_timepoints = label_data_list[0].shape[1]
            n_dissimilarities = (n_conditions * (n_conditions - 1)) // 2

            rdm_vec = np.zeros((n_dissimilarities, n_timepoints))

            for t in range(n_timepoints):
                data_at_t = np.array([label_data[:, t] for label_data in label_data_list])
                rdm_vec[:, t] = pdist(data_at_t, metric='correlation')

            if rdm_vec is not None:
                save_fname = os.path.join(
                    sub_rdm_dir,
                    f'{sub_bids_id}_occlusion-{level}_roi-{roi_name}_rdm.npy'
                )
                np.save(save_fname, rdm_vec)
                
                if roi_name == 'V1-3':
                    time_fname = os.path.join(
                        sub_rdm_dir,
                        f'{sub_bids_id}_occlusion-{level}_times.npy'
                    )
                    if not os.path.exists(time_fname):
                         np.save(time_fname, times)
