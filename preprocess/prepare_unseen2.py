import os
import glob
import librosa
import datetime

unseen_noise_path_dict={
    "babble": "/media/kyunster/ssd1/corpus/crss_babble/test",
    "music": "/media/kyunster/ssd1/corpus/musan/12s_music",
}

for noise_type, noise_path in unseen_noise_path_dict.items():
    
    unseen_list = []
    cur_noise_list = glob.glob(os.path.join(noise_path, "*.wav"))
    unseen_list += [(noise_path, noise_type) for noise_path in cur_noise_list]
    
    noise_dict={
        "unseen_"+noise_type: unseen_list
    }
    from tqdm import tqdm
    os.makedirs(os.path.join("data", "filelist"), exist_ok=True)

    for noise_type, cur_list in noise_dict.items():    
        with open(os.path.join("data", "filelist", noise_type+".txt"), "w") as f:
            for cur_path, noise_type in cur_list:
                f.write(cur_path+"\t"+noise_type+"\n")