import mne
import numpy as np
import matplotlib.pyplot as plt
import os

sub_id = 1

base_path = os.path.join('Megocclusion', f'sub-{sub_id:02d}', 'meg')

events = [
    '0-camel', '60-camel', '80-camel',
    '0-deer', '60-deer', '80-deer',
    '0-car', '60-car', '80-car',
    '0-motor', '60-motor', '80-motor'
]

all_epochs = []
for event in events:
    fname = f"sub-{sub_id:02d}_task-objectrecognition_occlusion-{event}_meg.fif"
    file_path = os.path.join(base_path, fname)
    epochs = mne.read_epochs(file_path, preload=True)
    all_epochs.append(epochs)

concatenated_epochs = mne.concatenate_epochs(all_epochs)

data = concatenated_epochs.get_data(picks='meg')
n_epochs, n_channels, n_times = data.shape
data_reshaped = data.transpose(1, 0, 2).reshape(n_channels, n_epochs * n_times)

U, s, Vh = np.linalg.svd(data_reshaped, full_matrices=False)

plt.figure(figsize=(8, 6))
plt.plot(s, 'o-')
plt.yscale('log') 
plt.title(f'Singular Value Spectrum for sub-{sub_id}')
plt.xlabel('Singular Value Index')
plt.ylabel('Singular Value (log scale)')
plt.grid(True)
plt.show()
