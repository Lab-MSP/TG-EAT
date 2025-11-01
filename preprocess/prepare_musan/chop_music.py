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

def chop_wav(cur_wav, chop_len=12, sr=16000):
    wav_list = []
    wav_cnt = int(np.ceil(len(cur_wav)/(sr*chop_len)))
    cur_wav = np.tile(cur_wav, 2)
    for i in range(wav_cnt):
        wav_list.append(cur_wav[i*sr*chop_len:(i+1)*sr*chop_len])
    wav_list = np.array(wav_list)
    return wav_list

def write_chopped_wav(chop_wav_list, noise_path, sr=16000):
    for i, chopped_wav in enumerate(chop_wav_list):
        if len(chopped_wav) < sr*12:
            continue
        sf.write(noise_path[:-4]+"_"+str(i)+".wav", chopped_wav, sr)
import sys
from tqdm import tqdm
for noise_path in tqdm(noise_path_list):
    cur_wav, sr = librosa.load(noise_path, sr=16000)
    chop_wav_list = chop_wav(cur_wav)
    music_id = noise_path.split("/")[-2]
    out_noise_path = os.path.join(out_noise_dir, music_id+"-"+os.path.basename(noise_path))
    write_chopped_wav(chop_wav_list, noise_path)
        