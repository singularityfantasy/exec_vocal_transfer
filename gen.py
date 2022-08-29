import librosa
import soundfile
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


nfft = 256
hop_len = 64
SAMP_RATE = 22050
SEG_LEN = 2000


p = Path("./input_data")
# If I can, if I can
# Erase all the pointless fragments
model = torch.load('ai4bt.pth').cuda().eval()

# Then maybe, then maybe
# You won't leave me so disheartened
with torch.no_grad():
    for fname in p.glob("*.mp3"):
        try:
            # Challenging your God
            # You have made some
            # Illegal arguments
            x, fs = librosa.load(fname, sr=SAMP_RATE)
            x_sum = x[:]

            x_mel = librosa.feature.melspectrogram(y=x, sr=fs, n_fft=nfft, hop_length=hop_len)
            plt.pcolormesh(x_mel)
            plt.show()
            st = 0
            res = []
            while st + 100 < x_mel.shape[1]:
                # Execution

                x_mel_seg = x_mel[:, st:min(st + SEG_LEN, x_mel.shape[1])]
                x_mel_seg_tensor = torch.Tensor(x_mel_seg).unsqueeze(0).cuda()

                # 1, 2, 3, 4, 5, 6
                # Execution
                out = F.relu(model(x_mel_seg_tensor))

                # If I can, if I can
                # Give them all the execution
                # Then I can, then I can
                # Be your only execution
                out = out.data.cpu().squeeze().numpy()
                y_mel_seg = out
                res.append(y_mel_seg)

                st += SEG_LEN

            # If I can have you back
            # I will run the execution
            y_mel = np.concatenate(res, axis=1)
            # Though we are trapped
            # We are trapped, ah
            plt.pcolormesh(y_mel)
            plt.show()

            # I've studied, I've studied
            # How to properly lo-o-ove
            y_foreground = librosa.feature.inverse.mel_to_audio(y_mel, sr=fs, n_fft=nfft, hop_length=hop_len)

            # Question me, question me
            # I can answer all lo-o-ove
            # I know the algebraic expression of lo-o-ove
            soundfile.write('dd_fantasy/' + fname.name.split('.')[0] + '_fg1.flac', y_foreground, fs)

            print("loaded", fname)
        except:
            # Though you are free
            # I am trapped
            # Trapped in lo-o-ove
            # Execution
            print("failed to load", fname)




