import os
import mne
import numpy as np

dataset_dir = 'Megocclusion'
derivatives_dir = os.path.join(dataset_dir, 'derivatives')
inv_op_dir = os.path.join(derivatives_dir, 'inv_op')
stc_dir = os.path.join(derivatives_dir, 'stc_HCP')
os.makedirs(stc_dir, exist_ok=True)

sub_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15]
occlusion_levels = ['0', '60', '80']

lambda2 = 1.0 / 3 ** 2
method = "MNE"

parcellation = mne.datasets.fetch_hcp_mmp_parcellation(
    subjects_dir=mne.datasets.sample.data_path() / 'subjects',
    accept=True
)
labels = mne.read_labels_from_annot(
    'fsaverage',
    parc='HCPMMP1',
    subjects_dir=mne.datasets.sample.data_path() / 'subjects'
)
label_names = [label.name for label in labels]

for sub in sub_ids:
    sub_bids_id = f'sub-{sub:02d}'
    print(f"Processing {sub_bids_id}...")

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
        
        stcs = mne.minimum_norm.apply_inverse_epochs(
            epochs, inv_op, lambda2, method=method, pick_ori="normal"
        )
        
        for stc in stcs:
            stc.apply_baseline(baseline=(None, 0))

        label_ts = mne.extract_label_time_course(
            stcs, labels, inv_op['src'], mode='pca_flip', return_generator=False
        )
        
        info = mne.create_info(ch_names=label_names, sfreq=epochs.info['sfreq'], ch_types='misc')

        roi_epochs = mne.EpochsArray(np.array(label_ts), info, tmin=epochs.tmin)

        output_fname = os.path.join(sub_stc_dir, f'{sub_bids_id}_occlusion-{level}_mmp-ts.fif')
        roi_epochs.save(output_fname, overwrite=True)
