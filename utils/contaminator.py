import numpy as np
import soundfile
import torch

def add_gain(clean_wav, gain=2):
    noisy_wav = clean_wav * gain
    return noisy_wav

def add_convolution(clean_wav, rir_wav):
    clean_dur = len(clean_wav)
    noisy_wav = np.convolve(clean_wav, rir_wav)[:clean_dur]
    return noisy_wav

def calc_noise_gain(clean_wav, noise_wav, SNR):
    clean_power = np.mean(np.abs(clean_wav))
    noise_power = np.mean(np.abs(noise_wav))
    noise_gain = clean_power / (noise_power * (10 ** (SNR/10)) )
    return noise_gain

def add_additive(clean_wav, noise_wav, SNR):
    noise_gain = calc_noise_gain(clean_wav, noise_wav, SNR)
    noisy_wav = clean_wav + noise_gain * noise_wav
    return noisy_wav

def add_white_noise(clean_wav, SNR=10):
    clean_dur = len(clean_wav)
    white_noise = np.random.randn(clean_dur)
    noisy_wav = add_additive(clean_wav, white_noise, SNR)
    return noisy_wav
    
def contaminate_all(clean_wavs, noise_wavs, SNR):
    is_mct = True if type(SNR) == list else False
    result_noisy_wavs = []
    
    for idx, (clean_wav, noise_wav) in enumerate(zip(clean_wavs, noise_wavs)):
        cur_snr = np.random.choice(SNR) if is_mct else SNR
        if np.sum(noise_wav**2) == 0:
            cur_noisy_wav = add_white_noise(clean_wav, cur_snr)
        else:
            cur_noisy_wav = add_additive(clean_wav, noise_wav, cur_snr)
        result_noisy_wavs.append(cur_noisy_wav)
    return result_noisy_wavs

def make_noisy_test_wavs(cur_utts, cur_wavs, noise_pair_path, snr):
    import librosa
    from utils.data.wav import load_audio
    noise_wav_path_list = []
    noise_dict=dict()
    with open(noise_pair_path, "r") as f:
        data_idx = 0
        for line in f:
            clean_utt_id, noise_path, noise_type, sidx, eidx = line.strip().split("\t")
            noise_wav_path_list.append(noise_path)
            
            noise_dict[clean_utt_id] = [data_idx, int(sidx), int(eidx), noise_type]
            data_idx += 1
    raw_noisy_wavs = load_audio(noise_wav_path_list)
    noisy_wavs=[]
    noise_types = []
    for utt_id in cur_utts:
        noise_info = noise_dict[utt_id]
        
        cur_noise_wav = raw_noisy_wavs[noise_info[0]]
        cur_noise_wav = cur_noise_wav[noise_info[1]:noise_info[2]]
        noisy_wavs.append(cur_noise_wav)
        noise_types.append(noise_info[3])
    
    result_wavs = contaminate_all(cur_wavs, noisy_wavs, snr)
    return result_wavs, noise_types

def make_noisy_train_wavs(clean_wavs, noise_wavs, SNR):
    is_mct = True if type(SNR) == list else False
    result_noisy_wavs = []
    
    for idx, (clean_wav, noise_wav) in enumerate(zip(clean_wavs, noise_wavs)):
        cur_snr = SNR[np.random.randint(0, len(SNR))] if is_mct else SNR
        if len(clean_wav) != len(noise_wav):
            noise_wav = noise_wav[:len(clean_wav)]
        clean_wav = clean_wav.numpy()
        noise_wav = noise_wav.numpy()
        if np.sum(noise_wav**2) == 0:
            cur_noisy_wav = add_white_noise(clean_wav, cur_snr)
        else:
            cur_noisy_wav = add_additive(clean_wav, noise_wav, cur_snr)
        result_noisy_wavs.append(cur_noisy_wav)
    result_noisy_wavs = torch.Tensor(np.array(result_noisy_wavs))
    return result_noisy_wavs