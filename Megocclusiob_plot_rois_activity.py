import os
import mne
import numpy as np

dataset_dir = 'Megocclusion'
derivatives_dir = os.path.join(dataset_dir, 'derivatives')
stc_dir = os.path.join(derivatives_dir, 'evoked_stc_HCP')
subjects_dir = mne.datasets.sample.data_path() / 'subjects'

sub_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15]
occlusion_level_to_plot = '60'

all_subject_evokeds = []
for sub in sub_ids:
    sub_bids_id = f'sub-{sub:02d}'
    sub_stc_dir = os.path.join(stc_dir, sub_bids_id)
    evoked_fname = os.path.join(sub_stc_dir, f'{sub_bids_id}_occlusion-{occlusion_level_to_plot}_mmp-evoked-ts.fif')
    
    if os.path.isfile(evoked_fname):
        evoked = mne.read_evokeds(evoked_fname, verbose=False)[0]
        all_subject_evokeds.append(evoked)

if not all_subject_evokeds:
    raise RuntimeError(f"No data found for occlusion level {occlusion_level_to_plot}.")

grand_average = mne.grand_average(all_subject_evokeds)

labels = mne.read_labels_from_annot(
    'fsaverage',
    parc='HCPMMP1',
    subjects_dir=subjects_dir,
)

label_ts_data = grand_average.data

stc = mne.labels_to_stc(labels, label_ts_data, tmin=grand_average.tmin, subject='fsaverage', tstep=1/grand_average.info['sfreq'])

brain = stc.plot(
    subject='fsaverage',
    subjects_dir=subjects_dir,
    hemi='both',
    clim='auto',
    time_viewer=True
)
input()
