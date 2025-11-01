import numpy as np
import pickle as pk
import torch.utils as torch_utils
from . import normalizer

"""
All dataset should have the same order based on the utt_list
"""
def load_norm_stat(norm_stat_file):
    with open(norm_stat_file, 'rb') as f:
        wav_mean, wav_std = pk.load(f)
    return wav_mean, wav_std


class CombinedSet(torch_utils.data.Dataset): 
    def __init__(self, *args, **kwargs):
        super(CombinedSet, self).__init__()
        self.datasets = kwargs.get("datasets", args[0]) 
        self.data_len = len(self.datasets[0])
        for cur_dataset in self.datasets:
            assert len(cur_dataset) == self.data_len, f"All dataset should have the same order based on the utt_list {len(cur_dataset)} {self.data_len}"
    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        result = []
        for cur_dataset in self.datasets:
            result.append(cur_dataset[idx])
        return result


class WavSet(torch_utils.data.Dataset): 
    def __init__(self, *args, **kwargs):
        super(WavSet, self).__init__()
        self.wav_list = kwargs.get("wav_list", args[0]) # (N, D, T)

        self.wav_mean = kwargs.get("wav_mean", None)
        self.wav_std = kwargs.get("wav_std", None)

        self.upper_bound_max_dur = kwargs.get("max_dur", 12)
        self.sampling_rate = kwargs.get("sr", 16000)

        # check max duration
        self.max_dur = np.min([np.max([len(cur_wav) for cur_wav in self.wav_list]), self.upper_bound_max_dur*self.sampling_rate])
        if self.wav_mean is None or self.wav_std is None:
            self.wav_mean, self.wav_std = normalizer. get_norm_stat_for_wav(self.wav_list)
    
    def save_norm_stat(self, norm_stat_file):
        with open(norm_stat_file, 'wb') as f:
            pk.dump((self.wav_mean, self.wav_std), f)
            
    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, idx):
        cur_wav = self.wav_list[idx][:self.max_dur]
        cur_dur = len(cur_wav)
        cur_wav = (cur_wav - self.wav_mean) / (self.wav_std+0.000001)
        
        result = (cur_wav, cur_dur)
        return result

class ADV_EmoSet(torch_utils.data.Dataset): 
    def __init__(self, *args, **kwargs):
        super(ADV_EmoSet, self).__init__()
        self.lab_list = kwargs.get("lab_list", args[0])
        self.max_score = kwargs.get("max_score", 7)
        self.min_score = kwargs.get("min_score", 1)
    
    def __len__(self):
        return len(self.lab_list)

    def __getitem__(self, idx):
        cur_lab = self.lab_list[idx]
        cur_lab = (cur_lab - self.min_score) / (self.max_score-self.min_score)
        result = cur_lab
        return result

class SpkSet(torch_utils.data.Dataset): 
    def __init__(self, *args, **kwargs):
        super(SpkSet, self).__init__()
        self.spk_list = kwargs.get("spk_list", args[0])
    
    def __len__(self):
        return len(self.spk_list)

    def __getitem__(self, idx):
        cur_lab = self.spk_list[idx]
        result = cur_lab
        return result    

class UttSet(torch_utils.data.Dataset): 
    def __init__(self, *args, **kwargs):
        super(UttSet, self).__init__()
        self.utt_list = kwargs.get("utt_list", args[0])
    
    def __len__(self):
        return len(self.utt_list)

    def __getitem__(self, idx):
        cur_lab = self.utt_list[idx]
        result = cur_lab
        return result
import os
import glob
import numpy as np
import random

class NoiseTypeSet(torch_utils.data.Dataset): 
    def __init__(self, *args, **kwargs):
        super(NoiseTypeSet, self).__init__()
        self.noise_type_list = kwargs.get("noise_type_list", args[0])
        
    def __len__(self):
        return len(self.noise_type_list)

    def __getitem__(self, idx):
        cur_env = self.noise_type_list[idx].replace("+", " ")
        return cur_env

class NoiseCLAPSet(torch_utils.data.Dataset): 
    def __init__(self, *args, **kwargs):
        super(NoiseCLAPSet, self).__init__()
        self.noise_type_list = kwargs.get("noise_type_list", args[0])
        self.noise_vec_dir = kwargs.get("noise_vec_dir", 
                                          "data/txt_emb")
        self.miss_ratio = kwargs.get("miss_ratio", 0)
        self.noise_vec_dict = dict()
        noise_vec_path_list = glob.glob(os.path.join(self.noise_vec_dir, "*.npy"))
        for noise_vec_path in noise_vec_path_list:
            noise_type = os.path.basename(noise_vec_path).replace(".npy", "")
            noise_vec = np.load(noise_vec_path)
            self.noise_vec_dict[noise_type] = noise_vec.squeeze()
        
    def __len__(self):
        return len(self.noise_type_list)

    def __getitem__(self, idx):
        cur_env = self.noise_type_list[idx]
        if self.miss_ratio != 0:
            seed = random.random()
            if seed < self.miss_ratio:
                old_env = cur_env
                while cur_env == old_env:
                    cur_env = np.random.choice(self.noise_type_list)
                # print("Mislabeled", old_env, cur_env)
        noise_vec = self.noise_vec_dict[cur_env]
        return noise_vec

class NoiseIdxSet(torch_utils.data.Dataset): 
    def __init__(self, *args, **kwargs):
        super(NoiseIdxSet, self).__init__()
        self.noise_type_list = kwargs.get("noise_type_list", args[0])
        self.noise_idx_dict = kwargs.get("noise_idx_dict", None)
        if self.noise_idx_dict is None:
            uniq_noise_set = list(set(self.noise_type_list))
            self.noise_idx_dict = {noise_type: noise_idx for noise_idx, noise_type in enumerate(uniq_noise_set)}
        
    def __len__(self):
        return len(self.noise_type_list)

    def get_domain_num(self):
        return len(self.noise_idx_dict)
    
    def __getitem__(self, idx):
        cur_env = self.noise_type_list[idx]
        noise_idx = self.noise_idx_dict[cur_env]
        
        return noise_idx


class TextSet(torch_utils.data.Dataset): 
    def __init__(self, *args, **kwargs):
        super(TextSet, self).__init__()
        self.text_list = kwargs.get("text_list", args[0])
    
    def __len__(self):
        return len(self.text_list)

    def __getitem__(self, idx):
        cur_text = self.text_list[idx]
        cur_text_dur = len(cur_text)
        return cur_text, cur_text_dur