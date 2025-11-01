import os
import glob
import librosa
import numpy as np

noise_dir="/media/kyunster/ssd1/RawData/freesound-noise"
noise_path_list = glob.glob(os.path.join(noise_dir, "*", "*.wav"))

pr_wav_path = "processed.txt"
with open(pr_wav_path, "r") as f:
    for line in f:
        pr_path = line.strip()
        if not os.path.exists(pr_path):
            continue
        cur_wav, sr = librosa.load(pr_path, sr=16000)
        
        if len(cur_wav) == 16000*12:
            continue
        elif len(cur_wav) < 16000*12:
            pr_child_path = pr_path.replace(".wav", "_0.wav")
            if os.path.exists(pr_child_path):
                os.system("rm "+pr_child_path)
            os.system("rm "+pr_path)
        else:
            os.system("rm "+pr_path)
