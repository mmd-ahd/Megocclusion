import os
import numpy as np
import mne

OCCLUSION_LEVEL = '0'
SUB_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15]  

dataset_dir = 'Megocclusion' 
stc_dir = os.path.join(dataset_dir, 'derivatives/occlusion_level_source_estimates')  
subjects_dir = mne.datasets.sample.data_path() / 'subjects'  

stcs, stcs_b = [], []  
for sub_id in SUB_IDS:
    sub_bids_id = f'sub-{sub_id:02d}'  
    base_fname = os.path.join(stc_dir, sub_bids_id, 'meg', f'{sub_bids_id}_task-objectrecognition_occlusion-{OCCLUSION_LEVEL}_meg')
    try:
        stc = mne.read_source_estimate(base_fname, subject='fsaverage')  
        stcs.append(stc)
    except FileNotFoundError:
        continue  
    
grand_average = np.mean([stc.data for stc in stcs], axis=0)
stc_template = stcs[0]  
stc_grand_average = mne.SourceEstimate(grand_average, vertices=stc_template.vertices, tmin=stc_template.tmin, tstep=stc_template.tstep, subject='fsaverage')

mne.viz.set_3d_backend('pyvistaqt')
 
stc_grand_average.plot(
    subject='fsaverage',
    subjects_dir=subjects_dir,
    hemi='both',
    smoothing_steps=10,
    size=(1000, 500),
    clim='auto'
)

import matplotlib.pyplot as plt
plt.show()

input("Press Enter to close the viewer and exit...")
