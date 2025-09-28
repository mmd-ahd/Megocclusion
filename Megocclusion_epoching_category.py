import os
import h5py
import numpy as np
import mne

output_dir = 'Megocclusion'
sample_raw_fname = 'subj04NN_sess01-0_tsss.fif'
sub_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15] # Note: subject 11 is excluded due to noise issues
TMIN, TMAX = -0.2, 1.0
BASELINE = (TMIN, 0)
REJECT = dict(grad=6000e-13, mag=6e-12) # lenient rejection criteria

EVENT_ID_MAP = {
    '0/camel': [3],
    '60/camel': [4],
    '80/camel': [5],
    '0/deer': [9],
    '60/deer': [10],
    '80/deer': [11],
    '0/car': [15],
    '60/car': [16],
    '80/car': [17],
    '0/motor': [21],
    '60/motor': [22],
    '80/motor': [23]
}

if not os.path.exists(sample_raw_fname):
    raise FileNotFoundError(f"Info source file not found at '{sample_raw_fname}'")

sample_raw = mne.io.read_raw_fif(sample_raw_fname, preload=False, on_split_missing='ignore') # raw data is missing so we load sample info of a similar data
info = mne.pick_info(sample_raw.info, mne.pick_types(sample_raw.info, meg=True, eeg=False))
info['bads'] = []
sample_raw.close()

for sub in sub_ids:
    sub_bids_id = f'sub-{sub:02d}'
    mat_path = os.path.join(output_dir, f'subject{sub}.mat')
    if not os.path.isfile(mat_path):
        continue

    sub_output_dir = os.path.join(output_dir, sub_bids_id, 'meg')
    os.makedirs(sub_output_dir, exist_ok=True)

    with h5py.File(mat_path, 'r') as f:
        for level, event_ids in EVENT_ID_MAP.items():
            level_epochs_data = []
            meg_signals_ref = f['Data']['MEG_Signals']

            for cond in event_ids:
                if cond < meg_signals_ref.shape[0]:
                    ref = meg_signals_ref[cond, 0]
                    level_epochs_data.append(f[ref][()])

            if not level_epochs_data:
                continue

            stacked_data = np.vstack(level_epochs_data)
            data_for_epochs = stacked_data.transpose(0, 2, 1)

            epochs = mne.EpochsArray(data_for_epochs, info, tmin=TMIN,
                                     reject=REJECT, baseline=BASELINE)

            output_fname = os.path.join(sub_output_dir, f'{sub_bids_id}_task-objectrecognition-occlusion-{level.replace("/", "-")}_meg.fif')
            epochs.save(output_fname, overwrite=True)
            print(f'Saved: {output_fname}')
