import os
import mne
from mne.minimum_norm import make_inverse_operator, write_inverse_operator

dataset_dir = 'Megocclusion'
output_dir = os.path.join(dataset_dir, 'derivatives', 'inv_op')
os.makedirs(output_dir, exist_ok=True)

sub_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15]
occlusion_levels = ['0', '60', '80']
subject_ranks = {
    1: 70, 2: 72, 3: 71, 4: 73, 5: 59, 6: 70, 7: 67, 8: 69,
    9: 70, 10: 70, 12: 65, 13: 68, 14: 71, 15: 66
}

fwd_fname = os.path.join(dataset_dir, 'fwd', 'fsaverage-meg-oct6-fwd.fif')
fwd = mne.read_forward_solution(fwd_fname)

for sub in sub_ids:
    sub_bids_id = f'sub-{sub:02d}'
    print(f"Processing {sub_bids_id}")

    sub_epochs_dir = os.path.join(dataset_dir, sub_bids_id, 'meg')
    sub_output_dir = os.path.join(output_dir, sub_bids_id)
    os.makedirs(sub_output_dir, exist_ok=True)
    
    inv_fname = os.path.join(sub_output_dir, f'{sub_bids_id}-inv.fif')

    epochs_list = []
    for level in occlusion_levels:
        epochs_fname = os.path.join(sub_epochs_dir, f'{sub_bids_id}_task-objectrecognition_occlusion-{level}_meg.fif')
        if os.path.isfile(epochs_fname):
            epochs_list.append(mne.read_epochs(epochs_fname, preload=True, verbose=False))

    if not epochs_list:
        print(f"  No epoch files found for {sub_bids_id}, skipping.")
        continue
    
    all_epochs = mne.concatenate_epochs(epochs_list)

    noise_cov = mne.compute_covariance(all_epochs, tmax=0.0, method='auto', rank={'meg': subject_ranks[sub]})
    
    inv_op = make_inverse_operator(all_epochs.info, fwd, noise_cov, loose=0.2, depth=None)
    
    write_inverse_operator(inv_fname, inv_op, overwrite=True)
    print(f"  Saved {inv_fname}")
