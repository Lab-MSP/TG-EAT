import os
import glob
import librosa
import datetime

noise_dir="/media/kyunster/ssd1/RawData/freesound-noise"
noise_list = os.listdir(noise_dir)
print(noise_list, len(noise_list))

unseen_noise_list= ["plaza", "school", "tram", "sea", "boat"]
train_list = []
dev_list = []
test_list = []
unseen_list = []
for noise_type in noise_list:
    cur_noise_list = glob.glob(os.path.join(noise_dir, noise_type, "*.wav"))
    if noise_type in unseen_noise_list:
        unseen_list += [(noise_path, noise_type) for noise_path in cur_noise_list]
    else:
        train_list += [(noise_path, noise_type) for noise_path in cur_noise_list[:int(len(cur_noise_list)*0.7)]]
        dev_list += [(noise_path, noise_type) for noise_path in cur_noise_list[int(len(cur_noise_list)*0.7):int(len(cur_noise_list)*0.8)]]
        test_list += [(noise_path, noise_type) for noise_path in cur_noise_list[int(len(cur_noise_list)*0.8):]]

print(len(train_list), len(test_list), len(unseen_list))

noise_dict={
    "train": train_list,
    "dev": dev_list,
    "test": test_list,
    "unseen": unseen_list
}
from tqdm import tqdm
os.makedirs(os.path.join("data", "filelist"), exist_ok=True)
# error_f = open("error.txt", "w")
for noise_type, cur_list in noise_dict.items():
    # cur_dur = 0
    # for cur_wav_path, _ in tqdm(cur_list):
    #     try:
    #         cur_wav, sr = librosa.load(cur_wav_path, sr=16000)
    #         assert len(cur_wav) == 12*sr, print(len(cur_wav))
    #         cur_dur += len(cur_wav)/sr
    #     except:
    #         error_f.write(cur_wav_path+" ")
        
    # timedelta_obj = datetime.timedelta(seconds=cur_dur)
    # print("Time in H M S format: ",timedelta_obj)
    
    with open(os.path.join("data", "filelist", noise_type+".txt"), "w") as f:
        for cur_path, noise_type in cur_list:
            f.write(cur_path+"\t"+noise_type+"\n")

# error_f.close()