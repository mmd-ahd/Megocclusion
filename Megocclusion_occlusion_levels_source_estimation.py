import os
import mne
from mne.minimum_norm import apply_inverse, read_inverse_operator

dataset_dir = 'Megocclusion'
source_estimates_dir = os.path.join(dataset_dir, 'derivatives', 'occlusion_level_source_estimates')
inverse_operators_dir = os.path.join(dataset_dir, 'derivatives', 'inv_op')
os.makedirs(source_estimates_dir, exist_ok=True)

sub_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15]
TMIN, TMAX = -0.2, 1.0
BASELINE = (TMIN, 0)
occlusion_levels = ['0', '60', '80']

for sub in sub_ids:
    sub_bids_id = f'sub-{sub:02d}'
    print(f"Processing {sub_bids_id}...")
    
    sub_epochs_dir = os.path.join(dataset_dir, sub_bids_id, 'meg')
    sub_inv_op_dir = os.path.join(inverse_operators_dir, sub_bids_id)
    sub_output_dir = os.path.join(source_estimates_dir, sub_bids_id, 'meg')
    os.makedirs(sub_output_dir, exist_ok=True)

    inv_fname = os.path.join(sub_inv_op_dir, f'{sub_bids_id}-inv.fif')
    if not os.path.isfile(inv_fname):
        print(f"  Inverse operator not found for {sub_bids_id}, skipping.")
        continue
    inv_op = read_inverse_operator(inv_fname)

    for level in occlusion_levels:
        epochs_fname = os.path.join(sub_epochs_dir, f'{sub_bids_id}_task-objectrecognition_occlusion-{level}_meg.fif')
        base_fname = os.path.join(sub_output_dir, f'{sub_bids_id}_task-objectrecognition_occlusion-{level}_meg')

        if not os.path.isfile(epochs_fname):
            continue

        if os.path.isfile(f'{base_fname}-lh.stc') and os.path.isfile(f'{base_fname}-rh.stc'):
            continue
            
        epochs = mne.read_epochs(epochs_fname, preload=True, verbose=False)
        evoked = epochs.average()

        stc = apply_inverse(evoked, inv_op, lambda2=1.0/9.0, method='MNE', pick_ori=None)
        stc.apply_baseline(baseline=BASELINE)
        stc.save(base_fname, overwrite=True)
        
        del epochs, evoked, stc

print("\nSource estimation complete for all subjects.")
