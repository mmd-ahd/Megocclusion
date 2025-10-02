import os
import mne

dataset_dir = 'Megocclusion'
output_dir = 'fwd'
output_path = os.path.join(dataset_dir, output_dir)
os.makedirs(output_path, exist_ok=True)
fwd_fname = os.path.join(output_path, 'fsaverage-meg-oct6-fwd.fif')

sample_raw_fname = 'subj04NN_sess01-0_tsss.fif'

subjects_dir = mne.datasets.sample.data_path() / 'subjects'
os.environ['SUBJECTS_DIR'] = str(subjects_dir)

src = mne.setup_source_space(subject='fsaverage', spacing='oct6',
                             subjects_dir=subjects_dir, add_dist=False)

model = mne.make_bem_model(subject='fsaverage', subjects_dir=subjects_dir)
bem = mne.make_bem_solution(model)

sample_raw = mne.io.read_raw_fif(sample_raw_fname, preload=False, on_split_missing='ignore')
info = mne.pick_info(sample_raw.info, mne.pick_types(sample_raw.info, meg=True, eeg=False))
info['bads'] = []
sample_raw.close()

trans = os.path.join(str(subjects_dir), 'fsaverage', 'bem', 'fsaverage-trans.fif')

fwd = mne.make_forward_solution(info, trans=trans, src=src, bem=bem,
                                meg=True, eeg=False, n_jobs=-1)

mne.write_forward_solution(fwd_fname, fwd, overwrite=True)

print(f"\nForward model computed and saved to:\n{os.path.abspath(fwd_fname)}")
