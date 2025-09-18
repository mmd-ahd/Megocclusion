import os
import numpy as np
import mne
from scipy import stats
from mne.stats import permutation_cluster_1samp_test

# --- Analysis Parameters ---
DERIVATIVES_DIR = 'Megocclusion/derivatives'
SUB_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15]
P_THRESHOLD = 0.05
N_PERMUTATIONS = 1024 

CONTRASTS = [('60', '0'), ('80', '0')]
TIME_WINDOWS = [(t, t + 0.100) for t in np.arange(0.050, 0.550, 0.100)]

stc_dir = os.path.join(DERIVATIVES_DIR, 'occlusion_level_source_estimates')
results_dir = os.path.join(DERIVATIVES_DIR, 'cluster_results')
os.makedirs(results_dir, exist_ok=True)
subjects_dir = mne.datasets.sample.data_path() / 'subjects'


src = mne.setup_source_space(subject='fsaverage', spacing='ico4',
                                subjects_dir=subjects_dir, add_dist=False)
adjacency = mne.spatial_src_adjacency(src)

for level_a, level_b in CONTRASTS:
    print(f"\nProcessing contrast: {level_a}% vs. {level_b}%")

    stcs_a_full, stcs_b_full = [], []
    for sub_id in SUB_IDS:
        sub_bids_id = f'sub-{sub_id:02d}'
        base_fname_a = os.path.join(stc_dir, sub_bids_id, 'meg',
                                    f'{sub_bids_id}_task-objectrecognition_occlusion-{level_a}_meg')
        base_fname_b = os.path.join(stc_dir, sub_bids_id, 'meg',
                                    f'{sub_bids_id}_task-objectrecognition_occlusion-{level_b}_meg')
        try:
            stcs_a_full.append(mne.read_source_estimate(base_fname_a, subject='fsaverage'))
            stcs_b_full.append(mne.read_source_estimate(base_fname_b, subject='fsaverage'))
        except FileNotFoundError:
            print(f"  Warning: File not found for sub-{sub_id:02d}. Skipping.")
            continue

    if not stcs_a_full:
        print(f"Error: No data loaded for contrast {level_a} vs {level_b}. Skipping.")
        continue

    for tmin, tmax in TIME_WINDOWS:
        print(f"  Running analysis for time window: {tmin*1000:.0f} - {tmax*1000:.0f} ms")

        stcs_a = [stc.copy().crop(tmin=tmin, tmax=tmax).mean() for stc in stcs_a_full]
        stcs_b = [stc.copy().crop(tmin=tmin, tmax=tmax).mean() for stc in stcs_b_full]

        X_diff = np.array([a.data - b.data for a, b in zip(stcs_a, stcs_b)])

        t_threshold = -stats.distributions.t.ppf(P_THRESHOLD / 2., len(X_diff) - 1)

        t_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(
            X_diff,
            adjacency=adjacency,
            n_jobs=-1,
            threshold=t_threshold,
            n_permutations=N_PERMUTATIONS,
            tail=0,
            out_type='indices'
        )

        fname_tmin = int(tmin * 1000)
        fname_tmax = int(tmax * 1000)
        results_fname = os.path.join(
            results_dir,
            f'results_contrast_{level_a}v{level_b}_{fname_tmin}-{fname_tmax}ms.npz'
        )
        
        np.savez(
            results_fname,
            t_obs=t_obs,
            clusters=np.array(clusters, dtype=object),
            cluster_pv=cluster_pv,
            fsave_vertices=[s['vertno'] for s in src]
        )

        n_significant = np.sum(cluster_pv < P_THRESHOLD)
        print(f"    Found {n_significant} significant clusters.")
        print(f"    Results saved to: {results_fname}")
