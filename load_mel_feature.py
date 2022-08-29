import librosa
import soundfile
import numpy as np
from pathlib import Path
import torch
import h5py


nfft = 256
hop_len = 64
SAMP_RATE = 22050
SEG_LEN = 5
frame_per_sec = int(SAMP_RATE / hop_len)
mel_seg_len = frame_per_sec * SEG_LEN

p = Path("./data_clean_new")
with h5py.File('dd_features_clean.hdf5', "w") as f:
    dset_x = f.create_dataset("feature_x", (1, 128, mel_seg_len), maxshape=(None, 128, mel_seg_len))
    idx = 0
    for fname in p.glob("*.flac"):
        try:
            x, fs = soundfile.read(fname.as_posix())
            # https://librosa.org/doc/latest/auto_examples/plot_vocal_separation.html
            x_mel = librosa.feature.melspectrogram(y=x, sr=fs, n_fft=nfft, hop_length=hop_len)
            st_pos = 0
            while st_pos + mel_seg_len < x_mel.shape[1]:
                x_mel_seg = x_mel[:, st_pos:st_pos + mel_seg_len]

                if idx > 0:
                    dset_x.resize((idx + 1, 128, mel_seg_len))
                dset_x[idx] = x_mel_seg

                st_pos += mel_seg_len
                idx += 1
        except:
            print("failed to load", fname)
