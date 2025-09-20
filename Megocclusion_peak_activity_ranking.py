import os
import mne
import numpy as np
import pandas as pd

dataset_dir = 'Megocclusion'
derivatives_dir = os.path.join(dataset_dir, 'derivatives')
stc_dir = os.path.join(derivatives_dir, 'evoked_stc_HCP')

sub_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15]
occlusion_levels = ['0', '60', '80']

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

for level in occlusion_levels:
    print(f"Processing occlusion level {level}")
    
    peak_activity_data = []

    all_subject_evokeds = []
    
    for sub in sub_ids:
        sub_bids_id = f'sub-{sub:02d}'
        sub_stc_dir = os.path.join(stc_dir, sub_bids_id)
        evoked_fname = os.path.join(sub_stc_dir, f'{sub_bids_id}_occlusion-{level}_mmp-evoked-ts.fif')
        
        if os.path.isfile(evoked_fname):
            evoked = mne.read_evokeds(evoked_fname, verbose=False)[0]
            all_subject_evokeds.append(evoked)

    if not all_subject_evokeds:
        print(f"No data found for occlusion level {level}, skipping.")
        continue

    grand_average = mne.grand_average(all_subject_evokeds)

    for i, roi in enumerate(label_names):
        data = grand_average.data[i]
        peak_amp = np.max(data)
        peak_time_idx = np.argmax(data)
        peak_time = grand_average.times[peak_time_idx]
        
        peak_activity_data.append({
            'roi': roi,
            'peak_amplitude': peak_amp,
            'peak_time_s': peak_time
        })

    df = pd.DataFrame(peak_activity_data)
    df_sorted = df.sort_values(by='peak_amplitude', ascending=False)

    output_fname = os.path.join(derivatives_dir, f'roi_peak_activity_ranking_occlusion-{level}.csv')
    df_sorted.to_csv(output_fname, index=False)
