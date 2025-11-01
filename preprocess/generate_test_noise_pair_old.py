import os
import sys
import numpy as np
import librosa
from collections import defaultdict
from tqdm import tqdm
sys.path.append(os.getcwd())
import utils

dtype = sys.argv[1]
seed = int(sys.argv[2])
# Chop 12 sec audio from noise files
np.random.seed(seed)
noise_filelist_path=f"data/filelist/{dtype}.txt"
speech_noise_filepair_path=f"data/filelist/{dtype}.txt"
noise_segs = []
with open(noise_filelist_path, "r") as f:
    for line in f:
        cur_path, cur_type = line.strip().split("\t")
        noise_segs += [(cur_path, cur_type)]
np.random.shuffle(noise_segs)

msp_path = "/media/kyunster/ssd1/corpus/MSP_Podcast_1.10"
audio_path = os.path.join(msp_path, "Audios/Audio")
label_path = os.path.join(msp_path, "Labels", "labels_consensus.csv")

cur_utts, cur_labs = utils.load_adv_emo_label(label_path, "test")
clean_wavs = utils.load_audio(audio_path, cur_utts)

noise_segs = noise_segs[:len(clean_wavs)]
if len(clean_wavs) > len(noise_segs):
    # repeat noise_segs
    noise_segs = noise_segs * (len(clean_wavs) // len(noise_segs)) + noise_segs[:len(clean_wavs) % len(noise_segs)]
    
assert len(clean_wavs) == len(noise_segs), print(len(clean_wavs), len(noise_segs))

os.makedirs(os.path.join("data", "test_pair", dtype), exist_ok=True)
wf = open(os.path.join("data", "test_pair", dtype, f"{seed}.txt"), "w")

noise_dur_dict=defaultdict(lambda: 0)
for clean_idx, (clean_wav, noise_info) in tqdm(enumerate(zip(clean_wavs, noise_segs)), total=len(clean_wavs)):
    clean_dur = len(clean_wav)
    noise_path, noise_type = noise_info
    noise_sidx = np.random.randint(0, 12*16000-clean_dur)
    noise_eidx = noise_sidx + clean_dur
    wf.write(f"{cur_utts[clean_idx]}\t{noise_path}\t{noise_type}\t{noise_sidx}\t{noise_eidx}\n")
    noise_dur_dict[noise_type] += clean_dur
wf.close()

import datetime
os.makedirs(os.path.join("data", "test_pair_dur", dtype), exist_ok=True)
wf = open(os.path.join("data", "test_pair_dur", dtype, f"{seed}.txt"), "w")
for noise_type, noise_dur in noise_dur_dict.items():
    timedelta_obj = datetime.timedelta(seconds=noise_dur/16000)
    print(f"{noise_type}: {timedelta_obj}")
    wf.write(f"{noise_type}\t{timedelta_obj}\n")
wf.close()