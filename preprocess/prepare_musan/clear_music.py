import os
import glob
import librosa
import numpy as np
import soundfile as sf


# Cut all the noise into 12 sec
# If short, repeat the noise until 12 sec
noise_dir="/media/kyunster/ssd1/corpus/musan/music"
out_noise_dir="/media/kyunster/ssd1/corpus/musan/12s_music"
os.makedirs(out_noise_dir, exist_ok=True)
noise_path_list = glob.glob(os.path.join(noise_dir, "*", "*.wav"))

import sys
for noise_path in noise_path_list:
    if "_" in noise_path:
        music_id = noise_path.split("/")[-2]
        out_noise_path = os.path.join(out_noise_dir, music_id+"-"+os.path.basename(noise_path))
        # print(noise_path, out_noise_path)
        # sys.exit()
        os.system("mv "+noise_path+" "+out_noise_path)