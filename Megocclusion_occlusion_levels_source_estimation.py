import os
import h5py
import numpy as np
import mne
from mne.minimum_norm import make_inverse_operator, apply_inverse

# configuration
dataset_dir = 'Megocclusion' 
output_dir = 'Megocclusion/derivatives/source_estimates'
os.makedirs(output_dir, exist_ok=True)

sub_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15]
TMIN, TMAX = -0.2, 1.0
BASELINE = (TMIN, 0)
REJECT = dict(grad=6000e-13, mag=6e-12)

# define occlusion levels and their corresponding event IDs
EVENT_ID_MAP = {
    '0': [3, 9, 15, 21],
    '60': [4, 10, 16, 22],
    '80': [5, 11, 17, 23]
}

# build the forward model once
print("Building forward model...")
subjects_dir = mne.datasets.sample.data_path() / 'subjects'
os.environ['SUBJECTS_DIR'] = str(subjects_dir)
src = mne.setup_source_space(subject='fsaverage', spacing='ico4', subjects_dir=subjects_dir, add_dist=False)

sample_raw = mne.io.read_raw_fif('subj04NN_sess01-0_tsss.fif', preload=False, on_split_missing='ignore')
info = mne.pick_info(sample_raw.info, mne.pick_types(sample_raw.info, meg=True, eeg=False))
info['bads'] = []
sample_raw.close()

model = mne.make_bem_model(subject='fsaverage', subjects_dir=subjects_dir)
bem = mne.make_bem_solution(model)
trans = os.path.join(subjects_dir, 'fsaverage', 'bem', 'fsaverage-trans.fif')
fwd = mne.make_forward_solution(info, trans=trans, src=src, bem=bem, meg=True, eeg=False, n_jobs=-1)
print("Forward model is ready.")

# main loop to process each subject
for sub in sub_ids:
    sub_bids_id = f'sub-{sub:02d}'
    print(f"\nProcessing {sub_bids_id}...")
    mat_path = os.path.join(dataset_dir, f'subject{sub}.mat')
    if not os.path.isfile(mat_path):
        print(f"  File not found, skipping.")
        continue

    sub_output_dir = os.path.join(output_dir, sub_bids_id, 'meg')
    os.makedirs(sub_output_dir, exist_ok=True)
    
    # create noise covariance from all conditions
    epochs_list_for_cov = []
    with h5py.File(mat_path, 'r') as f:
        meg_signals_ref = f['Data']['MEG_Signals']
        all_conditions = np.concatenate(list(EVENT_ID_MAP.values()))
        for cond in all_conditions:
            ref = meg_signals_ref[cond, 0]
            epochs_list_for_cov.append(f[ref][()])
            
    stacked_data = np.vstack(epochs_list_for_cov)
    data_for_cov = stacked_data.transpose(0, 2, 1) 
    epochs_for_cov = mne.EpochsArray(data_for_cov, info, tmin=TMIN, reject=REJECT, verbose=False)
    noise_cov = mne.compute_covariance(epochs_for_cov, tmax=0.0, method='auto', rank='info')
    inv_op = make_inverse_operator(epochs_for_cov.info, fwd, noise_cov, loose=0.2, depth=None)
    del epochs_for_cov, data_for_cov, stacked_data, epochs_list_for_cov

    # process and save each occlusion level
    with h5py.File(mat_path, 'r') as f:
        for level, event_ids in EVENT_ID_MAP.items():
            # The base_fname will now correctly include the 'meg' subdirectory
            base_fname = os.path.join(sub_output_dir, f'{sub_bids_id}_task-objectrecognition_occlusion-{level}_meg')
            
            if os.path.isfile(f'{base_fname}-lh.stc') and os.path.isfile(f'{base_fname}-rh.stc'):
                print(f"  Occlusion {level}% already processed, skipping.")
                continue

            print(f"  Processing occlusion level: {level}%")
            
            level_epochs_data = []
            meg_signals_ref = f['Data']['MEG_Signals']
            for cond in event_ids:
                ref = meg_signals_ref[cond, 0]
                level_epochs_data.append(f[ref][()])
            
            stacked_level_data = np.vstack(level_epochs_data)
            data_for_level = stacked_level_data.transpose(0, 2, 1)
            
            epochs_for_level = mne.EpochsArray(data_for_level, info, tmin=TMIN, baseline=BASELINE, reject=REJECT, verbose=False)
            
            evoked = epochs_for_level.average()
            stc = apply_inverse(evoked, inv_op, lambda2=1.0/9.0, method='MNE', pick_ori=None)
            
            stc.apply_baseline(baseline=BASELINE)
            
            stc.save(base_fname, overwrite=True)
            print(f"  Saved baseline-corrected {base_fname}-lh.stc and {base_fname}-rh.stc")
            del epochs_for_level, evoked, stc

print("\nSource estimation complete for all subjects.")