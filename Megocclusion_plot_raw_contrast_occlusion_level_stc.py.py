import os
import numpy as np
import mne

OCCLUSION_LEVEL_A = '60'  
OCCLUSION_LEVEL_B = '0'   
SUB_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15]  

dataset_dir = 'Megocclusion' 
stc_dir = os.path.join(dataset_dir, 'derivatives/occlusion_level_source_estimates')  
subjects_dir = mne.datasets.sample.data_path() / 'subjects'  

stcs_a, stcs_b = [], []  
for sub_id in sub_ids:
    sub_bids_id = f'sub-{sub_id:02d}'  
    base_fname_a = os.path.join(stc_dir, sub_bids_id, 'meg', f'{sub_bids_id}_task-objectrecognition_occlusion-{OCCLUSION_LEVEL_A}_meg')
    base_fname_b = os.path.join(stc_dir, sub_bids_id, 'meg', f'{sub_bids_id}_task-objectrecognition_occlusion-{OCCLUSION_LEVEL_B}_meg')
    try:
        stc_a = mne.read_source_estimate(base_fname_a, subject='fsaverage')  
        stc_b = mne.read_source_estimate(base_fname_b, subject='fsaverage')  
        stcs_a.append(stc_a)
        stcs_b.append(stc_b)
    except FileNotFoundError:
        continue  

grand_average_a_data = np.mean([stc.data for stc in stcs_a], axis=0)
grand_average_b_data = np.mean([stc.data for stc in stcs_b], axis=0)
stc_template = stcs_a[0]  
stc_grand_average_a = mne.SourceEstimate(grand_average_a_data, vertices=stc_template.vertices, tmin=stc_template.tmin, tstep=stc_template.tstep, subject='fsaverage')
stc_grand_average_b = mne.SourceEstimate(grand_average_b_data, vertices=stc_template.vertices, tmin=stc_template.tmin, tstep=stc_template.tstep, subject='fsaverage')
stc_difference = stc_grand_average_a - stc_grand_average_b 

mne.viz.set_3d_backend('pyvistaqt')
 
stc_difference.plot(
    subject='fsaverage',
    subjects_dir=subjects_dir,
    hemi='both',
    smoothing_steps=10,
    size=(1000, 500),
    title=f'Difference Wave: {OCCLUSION_LEVEL_A}% - {OCCLUSION_LEVEL_B}%',
    clim='auto'
)

import matplotlib.pyplot as plt
plt.show()

input("Press Enter to close the viewer and exit...")

