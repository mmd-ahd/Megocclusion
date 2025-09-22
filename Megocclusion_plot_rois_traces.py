import os
import numpy as np
import mne
from scipy import stats
from matplotlib import pyplot as plt
from mne.stats import spatio_temporal_cluster_1samp_test

dataset_dir = 'Megocclusion'
derivatives_dir = os.path.join(dataset_dir, 'derivatives')
roi_ts_dir = os.path.join(derivatives_dir, 'evoked_stc_HCP') 
figures_dir_base = os.path.join(derivatives_dir, 'figures', 'roi_tvalues_clustered')
os.makedirs(figures_dir_base, exist_ok=True)

sub_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15]
occlusion_levels = ['0', '60', '80']
contrasts = [('60', '0'), ('80', '0')]

tmin = 0.0  
tmax = 0.6 

rois_of_interest_short = [
    'V1', 'V2', 'V3', 'V3A', 'V3CD', 'V4', 'V4t', 'V6', 'V6A', '7PL', 'V7',
    'PGp', 'STGa', 'PIT', '45', '47m', '47l', 'LO1', 'LO2', 'LO3', 'MT',
    'TE1p', 'TE1a', 'TE2p', 'IPS1', 'POS2'
]

n_permutations = 1024
p_threshold = 0.05 

n_subjects = len(sub_ids)
df = n_subjects - 1
t_threshold = stats.distributions.t.ppf(1 - p_threshold / 2, df=df)

sub_bids_id_template = f'sub-{sub_ids[0]:02d}'
template_fname = os.path.join(roi_ts_dir, sub_bids_id_template, f'{sub_bids_id_template}_occlusion-0_mmp-evoked-ts.fif')
if not os.path.exists(template_fname):
    raise FileNotFoundError(f"Template file not found at {template_fname}. Please ensure the ROI time course files exist and the path is correct.")
    
template_evoked = mne.read_evokeds(template_fname, verbose=False)[0]
all_roi_names = template_evoked.ch_names

selected_roi_indices = []
selected_roi_full_names = []
for roi_short_name in rois_of_interest_short:
    matching_indices = [i for i, name in enumerate(all_roi_names) if f"_{roi_short_name}_" in name]
    if not matching_indices:
        print(f"Warning: Could not find ROI '{roi_short_name}' in the data files.")
        continue
    for idx in matching_indices:
        selected_roi_indices.append(idx)
        selected_roi_full_names.append(all_roi_names[idx])

n_rois = len(selected_roi_indices)
n_times = len(template_evoked.times)
all_label_ts = {level: np.zeros([n_subjects, n_rois, n_times]) for level in occlusion_levels}

for i, sub in enumerate(sub_ids):
    sub_bids_id = f'sub-{sub:02d}'
    print(f"Loading data for {sub_bids_id}")

    for level in occlusion_levels:
        fname = os.path.join(roi_ts_dir, sub_bids_id, f'{sub_bids_id}_occlusion-{level}_mmp-evoked-ts.fif')
        
        if not os.path.exists(fname):
            print(f"  Skipping missing file: {fname}")
            all_label_ts[level][i, :, :] = np.nan
            continue
            
        evoked_ts = mne.read_evokeds(fname, verbose=False)[0]
        selected_data = evoked_ts.data[selected_roi_indices, :]
        all_label_ts[level][i, :, :] = selected_data

for cond1, cond2 in contrasts:
    contrast_folder_name = f"{cond1}_vs_{cond2}"
    contrast_figures_dir = os.path.join(figures_dir_base, contrast_folder_name)
    os.makedirs(contrast_figures_dir, exist_ok=True)
    print(f"\nRunning analysis for contrast: {contrast_folder_name}. Figures will be saved to {contrast_figures_dir}")

    data_cond1_all_rois = all_label_ts[cond1]
    data_cond2_all_rois = all_label_ts[cond2]

    times = template_evoked.times
    time_mask = (times >= tmin) & (times <= tmax)
    
    X = (data_cond1_all_rois - data_cond2_all_rois).transpose(0, 2, 1)
    X = X[:, time_mask, :]

    T_obs, clusters, cluster_p_values, H0 = spatio_temporal_cluster_1samp_test(
        X,
        n_permutations=n_permutations,
        threshold=t_threshold, 
        tail=0, 
        adjacency=None,
        t_power=2
    )
    
    for roi_idx, roi_full_name in enumerate(selected_roi_full_names):
        fig, ax1 = plt.subplots(figsize=(12, 6))
        fig.suptitle(f'ROI: {roi_full_name} | Contrast: {cond1}% vs {cond2}%', fontsize=16)

        full_times_ms = times * 1000
        mean_cond1 = all_label_ts[cond1][:, roi_idx, :].mean(0)
        mean_cond2 = all_label_ts[cond2][:, roi_idx, :].mean(0)

        window_len = 10
        smooth_kernel = np.ones(window_len) / window_len
        mean_cond1_smooth = np.convolve(mean_cond1, smooth_kernel, mode='same')
        mean_cond2_smooth = np.convolve(mean_cond2, smooth_kernel, mode='same')
        diff_wave = mean_cond1_smooth - mean_cond2_smooth

        ax1.plot(full_times_ms, mean_cond1_smooth, label=f'Occlusion {cond1}%', color='darkblue', linewidth=1.5)
        ax1.plot(full_times_ms, mean_cond2_smooth, label=f'Occlusion {cond2}%', color='darkcyan', linewidth=1.5)
        ax1.plot(full_times_ms, diff_wave, label='Difference', color='purple', linewidth=1.5, linestyle='--')
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Activation (AU)', color='darkblue')
        ax1.tick_params(axis='y', labelcolor='darkblue')
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()
        t_values_roi = T_obs[:, roi_idx]
        ax2.plot(times[time_mask] * 1000, t_values_roi, color='gray', linestyle=':', label='t-values')
        ax2.set_ylabel('t-values', color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')
        ax2.axhline(0, color='gray', linestyle=':', linewidth=0.8)
        ax2.legend(loc='upper right')
        
        for cluster_idx, (cluster, p_val) in enumerate(zip(clusters, cluster_p_values)):
            if p_val < p_threshold and roi_idx in cluster[1]:
                time_indices_in_cluster = cluster[0][np.where(cluster[1] == roi_idx)]
                
                if time_indices_in_cluster.any():
                    t_start = times[time_mask][time_indices_in_cluster.min()] * 1000
                    t_end = times[time_mask][time_indices_in_cluster.max()] * 1000
                    ax1.axvspan(t_start, t_end, color='red', alpha=0.3, zorder=-1)
        
        ax1.set_xlim(full_times_ms.min(), full_times_ms.max())
        ax1.axvline(0, color='black', linestyle='-', linewidth=0.8)

        fig_fname = os.path.join(contrast_figures_dir, f'{roi_full_name}_{cond1}_vs_{cond2}_clustered.png')
        fig.savefig(fig_fname)
        plt.close(fig)

print(f"\nAnalysis complete. Figures saved to subfolders in: {figures_dir_base}")
