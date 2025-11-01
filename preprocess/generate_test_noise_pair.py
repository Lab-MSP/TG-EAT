import os
import sys
import glob
import numpy as np
import librosa
from collections import defaultdict
from tqdm import tqdm
sys.path.append(os.getcwd())
import utils

dtype = sys.argv[1]
assert dtype == "unseen"
seed = int(sys.argv[2])

seen_keywords = [
    "mall", "restaurant", "office", "airport", "station", "city", "park", "street", 
    "traffic", "home", "kitchen", "living room", "bathroom", "bedroom", "metro", 
    "bus", "car", "construction site", "pedestrian", "beach"
]

noise_dir="/media/kyunster/ssd1/RawData/freesound-noise"
noise_list = os.listdir(noise_dir)
unseen_list = list(set(noise_list) - set(seen_keywords))
unseen_list.sort()

print(unseen_list)
sound_dict={kwd: glob.glob(os.path.join(noise_dir, kwd, "*.wav")) for kwd in unseen_list}

msp_path = "/media/kyunster/ssd1/corpus/MSP-Podcast/1.10"
audio_path = os.path.join(msp_path, "Audios/Audio")
label_path = os.path.join(msp_path, "Labels", "labels_consensus.csv")
cur_utts, cur_labs = utils.load_adv_emo_label(label_path, "test")
clean_wavs = utils.load_audio(audio_path, cur_utts)

pair_path = os.path.join("data", "new_test_pair", dtype, f"{seed}.txt")
os.makedirs(os.path.dirname(pair_path), exist_ok=True)
wf = open(pair_path, "w")
for clean_idx, clean_wav in tqdm(enumerate(clean_wavs), total=len(clean_wavs)):
    clean_dur = len(clean_wav)
    noise_type = np.random.choice(unseen_list)
    cur_noise_wav_path = np.random.choice(sound_dict[noise_type])
    noise_wav, _ = librosa.load(cur_noise_wav_path, sr=16000)
    noise_dur = len(noise_wav)
    noise_sidx = np.random.randint(0, noise_dur-clean_dur)
    noise_eidx = noise_sidx + clean_dur
    wf.write(f"{cur_utts[clean_idx]}\t{cur_noise_wav_path}\t{noise_type}\t{noise_sidx}\t{noise_eidx}\n")
wf.close()

# noise_dur_dict=defaultdict(lambda: 0)
# for clean_idx, (clean_wav, noise_info) in tqdm(enumerate(zip(clean_wavs, noise_segs)), total=len(clean_wavs)):
#     clean_dur = len(clean_wav)
#     noise_path, noise_type = noise_info
#     noise_sidx = np.random.randint(0, 12*16000-clean_dur)
#     noise_eidx = noise_sidx + clean_dur
#     wf.write(f"{cur_utts[clean_idx]}\t{noise_path}\t{noise_type}\t{noise_sidx}\t{noise_eidx}\n")
#     noise_dur_dict[noise_type] += clean_dur
# wf.close()
