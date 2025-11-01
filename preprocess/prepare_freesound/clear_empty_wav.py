import os
import glob
import librosa
import numpy as np

noise_dir="/media/kyunster/ssd1/RawData/freesound-noise"
noise_path_list = glob.glob(os.path.join(noise_dir, "*", "*.wav"))

# empty_f = open("empty.txt", "w")
# from tqdm import tqdm
# for noise_path in tqdm(noise_path_list):
#     cur_wav, sr = librosa.load(noise_path, sr=16000)
#     if np.sum(cur_wav**2) == 0:
#         empty_f.write(noise_path+"\n")
# empty_f.close()


with open("empty.txt", "r") as f:
    for line in f:
        cur_path = line.strip()
        os.system("rm "+cur_path)