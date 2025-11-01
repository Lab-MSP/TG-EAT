import os
import glob
import librosa
import numpy as np
import soundfile as sf


# Cut all the noise into 12 sec
# If short, repeat the noise until 12 sec
noise_dir="/media/kyunster/ssd1/RawData/freesound-noise"
noise_path_list = glob.glob(os.path.join(noise_dir, "*", "*.wav"))

pr_wav_path = "processed_old.txt"
pr_wav_list = []
with open(pr_wav_path, "r") as f:
    for line in f:
        pr_path = line.strip()
        os.system("rm "+pr_path)
        pr_wav_list.append(pr_path.replace(".wav", ""))
        
        
bad_path = "bad.txt"
bad_f = open(bad_path, "w")

pr_path = "processed.txt"
pr_f = open(pr_path, "w")

def chop_wav(cur_wav, chop_len=12, sr=16000):
    wav_list = []
    if len(cur_wav) < sr*chop_len:
        cur_wav = np.tile(cur_wav, int(np.ceil(sr*chop_len/len(cur_wav))))
    for i in range(int(np.ceil(len(cur_wav)/(sr*chop_len)))):
        wav_list.append(cur_wav[i*sr*chop_len:(i+1)*sr*chop_len])
    return wav_list

def write_chopped_wav(chop_wav_list, noise_path, sr=16000):
    for i, chopped_wav in enumerate(chop_wav_list):
        if len(chopped_wav) < sr*12:
            continue
        sf.write(noise_path[:-4]+"_"+str(i)+".wav", chopped_wav, sr)
    
from tqdm import tqdm
for noise_path in tqdm(noise_path_list):
    try:
        cur_wav, sr = librosa.load(noise_path, sr=16000)
        if len(cur_wav) == 16000*12:
            pr_f.write(noise_path+"\n")
            continue
        chop_wav_list = chop_wav(cur_wav)
        write_chopped_wav(chop_wav_list, noise_path)
        pr_f.write(noise_path+"\n")
    except:
        bad_f.write(noise_path+"\n")
        
pr_f.close()
bad_f.close()