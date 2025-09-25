import os
import numpy as np
import mne
from scipy import stats
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from mne.stats import spatio_temporal_cluster_1samp_test
import math

dataset_dir = 'Megocclusion'
derivatives_dir = os.path.join(dataset_dir, 'derivatives')
roi_ts_dir = os.path.join(derivatives_dir, 'evoked_stc_HCP_merged')
figures_dir_base = os.path.join(derivatives_dir, 'figures', 'roi_activity_sig_merged')
os.makedirs(figures_dir_base, exist_ok=True)

sub_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15]
occlusion_levels = ['0', '60', '80']
contrasts = [('60', '0'), ('80', '0'), ('80', '60')]
tmin, tmax = 0.0, 0.6
window_len = 10

n_permutations = 2048
cluster_forming_p_threshold = 0.1
p_strict, p_liberal = 0.05, 0.1
n_subjects = len(sub_ids)
df = n_subjects - 1
t_threshold = stats.distributions.t.ppf(1 - cluster_forming_p_threshold / 2, df=df)
print(f"Using a t-threshold of {t_threshold:.2f} for cluster formation (p={cluster_forming_p_threshold})")

sub_bids_id_template = f'sub-{sub_ids[0]:02d}'
template_fname = os.path.join(roi_ts_dir, sub_bids_id_template, f'{sub_bids_id_template}_occlusion-0_merged-rois-evoked-ts.fif')
if not os.path.exists(template_fname):
    raise FileNotFoundError(f"Template file not found: {template_fname}.")

template_evoked = mne.read_evokeds(template_fname, verbose=False)[0]
roi_names = template_evoked.ch_names
n_rois = len(roi_names)
n_times = len(template_evoked.times)
times = template_evoked.times
time_mask = (times >= tmin) & (times <= tmax)
times_masked = times[time_mask]

all_label_ts = {level: np.zeros([n_subjects, n_rois, n_times]) for level in occlusion_levels}
for i, sub in enumerate(sub_ids):
    sub_bids_id = f'sub-{sub:02d}'
    print(f"Loading data for {sub_bids_id}")
    for level in occlusion_levels:
        fname = os.path.join(roi_ts_dir, sub_bids_id, f'{sub_bids_id}_occlusion-{level}_merged-rois-evoked-ts.fif')
        if not os.path.exists(fname):
            print(f"  Skipping missing file: {fname}")
            all_label_ts[level][i, :, :] = np.nan
            continue
        evoked_ts = mne.read_evokeds(fname, verbose=False)[0]
        all_label_ts[level][i, :, :] = evoked_ts.data

sig_masks = {}
for level in occlusion_levels:
    print(f"\nRunning baseline cluster test for Occlusion {level}%")
    X = all_label_ts[level].transpose(0, 2, 1)
    X_masked = X[:, time_mask, :]
    _, clusters, p_values, _ = spatio_temporal_cluster_1samp_test(
        X_masked, n_permutations=n_permutations, threshold=t_threshold,
        tail=0, adjacency=None
    )
    mask = np.zeros((n_rois, len(times_masked)), dtype=bool)
    for cl, p in zip(clusters, p_values):
        if p < p_strict:
            for r in range(n_rois):
                if r in cl[1]:
                    time_indices = cl[0][cl[1] == r]
                    mask[r, time_indices] = True
    sig_masks[level] = mask

for cond1, cond2 in contrasts:
    contrast_folder_name = f"{cond1}_vs_{cond2}"
    contrast_figures_dir = os.path.join(figures_dir_base, contrast_folder_name)
    os.makedirs(contrast_figures_dir, exist_ok=True)
    print(f"\nAnalyzing contrast: {contrast_folder_name}")

    X_diff = (all_label_ts[cond1] - all_label_ts[cond2]).transpose(0, 2, 1)
    X_diff_masked = X_diff[:, time_mask, :]

    T_obs, clusters_diff, p_values_diff, _ = spatio_temporal_cluster_1samp_test(
        X_diff_masked, n_permutations=n_permutations, threshold=t_threshold,
        tail=0, adjacency=None
    )

    lh_roi_indices = [i for i, name in enumerate(roi_names) if name.endswith('-lh')]
    rh_roi_indices = [i for i, name in enumerate(roi_names) if name.endswith('-rh')]

    fig_lh, fig_rh = None, None
    if lh_roi_indices:
        n_lh_rois = len(lh_roi_indices)
        n_cols = math.ceil(math.sqrt(n_lh_rois * 9 / 16))
        n_rows = math.ceil(n_lh_rois / n_cols)
        fig_lh, axes_lh = plt.subplots(n_rows, n_cols, figsize=(16, 9), squeeze=False)
        fig_lh.suptitle(f'Left Hemisphere ROIs | Contrast: {cond1}% vs {cond2}%', fontsize=20)
        axes_lh_flat = axes_lh.flatten()

    if rh_roi_indices:
        n_rh_rois = len(rh_roi_indices)
        n_cols = math.ceil(math.sqrt(n_rh_rois * 9 / 16))
        n_rows = math.ceil(n_rh_rois / n_cols)
        fig_rh, axes_rh = plt.subplots(n_rows, n_cols, figsize=(16, 9), squeeze=False)
        fig_rh.suptitle(f'Right Hemisphere ROIs | Contrast: {cond1}% vs {cond2}%', fontsize=20)
        axes_rh_flat = axes_rh.flatten()

    legend_handles = None
    for roi_idx, roi_name in enumerate(roi_names):
        ax_combined = None
        is_leftmost = False
        is_rightmost = False

        if roi_idx in lh_roi_indices:
            idx = lh_roi_indices.index(roi_idx)
            ax_combined = axes_lh_flat[idx]
            if idx % n_cols == 0: is_leftmost = True
            if (idx + 1) % n_cols == 0 or (idx + 1) == n_lh_rois: is_rightmost = True
        elif roi_idx in rh_roi_indices:
            idx = rh_roi_indices.index(roi_idx)
            ax_combined = axes_rh_flat[idx]
            if idx % n_cols == 0: is_leftmost = True
            if (idx + 1) % n_cols == 0 or (idx + 1) == n_rh_rois: is_rightmost = True

        fig_single, ax_single = plt.subplots(figsize=(12, 6))
        axes_to_plot = [ax_single]
        if ax_combined:
            axes_to_plot.append(ax_combined)

        for i, ax in enumerate(axes_to_plot):
            is_summary_plot = (i == 1)
            ax.set_title(roi_name)
            mean_c1 = np.nanmean(all_label_ts[cond1][:, roi_idx, :], axis=0)
            mean_c2 = np.nanmean(all_label_ts[cond2][:, roi_idx, :], axis=0)
            
            smooth_kernel = np.ones(window_len) / window_len
            mean_c1_smooth = np.convolve(mean_c1, smooth_kernel, mode='same')
            mean_c2_smooth = np.convolve(mean_c2, smooth_kernel, mode='same')

            ax.plot(times * 1000, mean_c1_smooth, color='#377eb8', linewidth=1.5, label=f'Occlusion {cond1}%')
            ax.plot(times * 1000, mean_c2_smooth, color='#e41a1c', linewidth=1.5, label=f'Occlusion {cond2}%')
            
            ax.plot(times_masked * 1000, np.where(sig_masks[cond1][roi_idx], mean_c1_smooth[time_mask], np.nan), color='#377eb8', linewidth=2.5)
            ax.plot(times_masked * 1000, np.where(sig_masks[cond2][roi_idx], mean_c2_smooth[time_mask], np.nan), color='#e41a1c', linewidth=2.5)

            ax.set_xlabel('Time (ms)')

            if not is_summary_plot or is_leftmost:
                 ax.set_ylabel('Activation (AU)', color='black')
            ax.tick_params(axis='y', labelcolor='black')

            ax2 = ax.twinx()
            t_values_roi_smooth = np.convolve(T_obs[:, roi_idx], smooth_kernel, mode='same')
            ax2.plot(times_masked * 1000, t_values_roi_smooth, color='#5ab4ac', linestyle=':', label='t-values (Difference)')
            if not is_summary_plot or is_rightmost:
                ax2.set_ylabel('t-values', color='#5ab4ac')
            ax2.tick_params(axis='y', labelcolor='#5ab4ac')
            ax2.axhline(0, color='#5ab4ac', linestyle=':', linewidth=0.8)

            for cluster, p_val in zip(clusters_diff, p_values_diff):
                if p_val < p_strict and roi_idx in cluster[1]:
                    time_idx = cluster[0][np.where(cluster[1] == roi_idx)]
                    if time_idx.any():
                        ax.axvspan(times_masked[time_idx.min()] * 1000, times_masked[time_idx.max()] * 1000, color='red', alpha=0.3)
                elif p_strict <= p_val < p_liberal and roi_idx in cluster[1]:
                     time_idx = cluster[0][np.where(cluster[1] == roi_idx)]
                     if time_idx.any():
                        ax.axvspan(times_masked[time_idx.min()] * 1000, times_masked[time_idx.max()] * 1000, color='yellow', alpha=0.4)

            ax.set_xlim(tmin * 1000, tmax * 1000)
            ax.axvline(0, color='black', linestyle='-', linewidth=0.8)

            if not legend_handles:
                h1, _ = ax.get_legend_handles_labels()
                h2, _ = ax2.get_legend_handles_labels()
                red_patch = mpatches.Patch(color='red', alpha=0.3, label=f'p < {p_strict}')
                yellow_patch = mpatches.Patch(color='yellow', alpha=0.4, label=f'{p_strict} \u2264 p < {p_liberal}')
                legend_handles = h1 + h2 + [red_patch, yellow_patch]

        fig_single.suptitle(f'ROI: {roi_name} | Contrast: {cond1}% vs {cond2}%', fontsize=16)
        fig_single.legend(handles=legend_handles, loc='upper right')
        fig_single.tight_layout(rect=[0, 0, 1, 0.95])
        fig_fname = os.path.join(contrast_figures_dir, f'{roi_name}_{cond1}_vs_{cond2}.png')
        fig_single.savefig(fig_fname, dpi=300)
        plt.close(fig_single)

    if fig_lh:
        for i in range(len(lh_roi_indices), len(axes_lh_flat)): fig_lh.delaxes(axes_lh_flat[i])
        fig_lh.legend(handles=legend_handles, bbox_to_anchor=(0.5, 0.05), loc='upper center', ncol=len(legend_handles))
        fig_lh.tight_layout(rect=[0, 0.08, 1, 0.95])
        fig_lh.savefig(os.path.join(contrast_figures_dir, f'lh_rois_{cond1}_vs_{cond2}.png'), dpi=300)
        plt.close(fig_lh)

    if fig_rh:
        for i in range(len(rh_roi_indices), len(axes_rh_flat)): fig_rh.delaxes(axes_rh_flat[i])
        fig_rh.legend(handles=legend_handles, bbox_to_anchor=(0.5, 0.05), loc='upper center', ncol=len(legend_handles))
        fig_rh.tight_layout(rect=[0, 0.08, 1, 0.95])
        fig_rh.savefig(os.path.join(contrast_figures_dir, f'rh_rois_{cond1}_vs_{cond2}.png'), dpi=300)
        plt.close(fig_rh)
