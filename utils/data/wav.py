import os
import librosa
from tqdm import tqdm
from multiprocessing import Pool

# Load audio
def extract_wav(wav_path):
    raw_wav, _ = librosa.load(wav_path, sr=16000)
    return raw_wav
def load_audio(audio_path, utts=None, nj=24):
    # Audio path: directory of audio files
    # utts: list of utterance names with .wav extension
    if utts is None:
        wav_paths = audio_path
    else:
        wav_paths = [os.path.join(audio_path, utt) for utt in utts]
    with Pool(nj) as p:
        wavs = list(tqdm(p.imap(extract_wav, wav_paths), total=len(wav_paths)))
    return wavs
def load_random_snr_audio(audio_path, utts=None, nj=24):
    # Audio path: directory of audio files
    # utts: list of utterance names with .wav extension
    import numpy as np
    snr_lst = ["-5", "0", "5"]
    wav_paths = []
    for utt in utts:
        snr = np.random.choice(snr_lst)
        wav_paths.append(os.path.join(audio_path+f"/{snr}/0", utt))
    with Pool(nj) as p:
        wavs = list(tqdm(p.imap(extract_wav, wav_paths), total=len(wav_paths)))
    return wavs