import os
import mne
from mne.minimum_norm import make_inverse_operator, apply_inverse

dataset_dir = 'Megocclusion'
output_dir = os.path.join(dataset_dir, 'derivatives', 'occlusion_level_source_estimates')
os.makedirs(output_dir, exist_ok=True)

sub_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15]
TMIN, TMAX = -0.2, 1.0
BASELINE = (TMIN, 0)
occlusion_levels = ['0', '60', '80']

fwd_fname = os.path.join(dataset_dir, 'fwd', 'fsaverage-meg-ico4-fwd.fif')
fwd = mne.read_forward_solution(fwd_fname)

for sub in sub_ids:
    sub_bids_id = f'sub-{sub:02d}'
    print(f"Processing {sub_bids_id}...")

    sub_epochs_dir = os.path.join(dataset_dir, sub_bids_id, 'meg')

    sub_output_dir = os.path.join(output_dir, sub_bids_id, 'meg')
    os.makedirs(sub_output_dir, exist_ok=True)

    all_epochs_list = []
    for level in occlusion_levels:
        epochs_fname = os.path.join(sub_epochs_dir, f'{sub_bids_id}_task-objectrecognition_occlusion-{level}_meg.fif')
        if os.path.isfile(epochs_fname):
            all_epochs_list.append(mne.read_epochs(epochs_fname, preload=True, verbose=False))

    if not all_epochs_list:
        print(f"  No epoch files found in {sub_epochs_dir}, skipping.")
        continue

    all_epochs = mne.concatenate_epochs(all_epochs_list)

    noise_cov = mne.compute_covariance(all_epochs, tmax=0.0, method='auto', rank='info')
    inv_op = make_inverse_operator(all_epochs.info, fwd, noise_cov, loose=0.2, depth=None)
    del all_epochs

    for level, epochs in zip(occlusion_levels, all_epochs_list):
        base_fname = os.path.join(sub_output_dir, f'{sub_bids_id}_task-objectrecognition_occlusion-{level}_meg')

        if os.path.isfile(f'{base_fname}-lh.stc') and os.path.isfile(f'{base_fname}-rh.stc'):
            continue

        evoked = epochs.average()
        stc = apply_inverse(evoked, inv_op, lambda2=1.0/9.0, method='MNE', pick_ori=None)
        stc.apply_baseline(baseline=BASELINE)
        stc.save(base_fname, overwrite=True)
        del epochs, evoked, stc

print("\nSource estimation complete for all subjects.")