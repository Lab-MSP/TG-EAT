# -*- coding: UTF-8 -*-
# Local modules
import os
import sys
import argparse
# 3rd-Party Modules
import numpy as np
import pickle as pk
import pandas as pd
from tqdm import tqdm
import glob
import librosa
import copy

# PyTorch Modules
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import torch.optim as optim
from transformers import AutoModel, AutoProcessor

from torch.utils.tensorboard import SummaryWriter

# Self-Written Modules
sys.path.append(os.getcwd())
import net
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--snr", type=int, default=10)
parser.add_argument("--data_path", type=str, default="/media/kyunster/ssd1/corpus/MSP_Podcast_1.10")
parser.add_argument("--noise_type", type=str, default="test")

parser.add_argument("--txt_type", type=str, choices=["bert", "clap"], default="clap")
parser.add_argument("--ssl_type", type=str, default="wavlm-base-plus")
parser.add_argument("--model_path", type=str, default="./model/wavlm-base-plus-clean")
parser.add_argument("--result_path", type=str, default="./result.txt")
parser.add_argument("--fuse_type", type=str, default="clean")
args = parser.parse_args()

assert 0 <= args.seed < 30, print("Invalid seed!")
utils.set_deterministic(args.seed)

SSL_TYPE = utils.get_ssl_type(args.ssl_type)
assert SSL_TYPE != None, print("Invalid SSL type!")
MODEL_PATH = args.model_path

from collections import defaultdict
corpus_path = args.data_path
audio_path = os.path.join(corpus_path, "Audios/Audio")
label_path = os.path.join(corpus_path, "Labels", "labels_consensus.csv")

noise_pair_path = f"data/test_pair/{args.noise_type}/{args.seed}.txt"

total_dataset=dict()
total_dataloader=dict()
for dtype in ["test"]:
    cur_utts, cur_labs = utils.load_adv_emo_label(label_path, dtype)
    cur_wavs = utils.load_audio(audio_path, cur_utts)
    wav_mean, wav_std = utils.load_norm_stat(MODEL_PATH+"/train_norm_stat.pkl")
    
    ############
    # Add noise #
    noisy_wavs, noise_type_list = utils.make_noisy_test_wavs(cur_utts, cur_wavs, noise_pair_path, snr=args.snr)
    ############
    
    cur_wav_set = utils.WavSet(noisy_wavs, wav_mean=wav_mean, wav_std=wav_std)
    cur_emo_set = utils.ADV_EmoSet(cur_labs)
    
    noise_idx_set = utils.NoiseIdxSet(noise_type_list)
    DOMAIN_NUM = noise_idx_set.get_domain_num()
    
    total_dataset[dtype] = utils.CombinedSet([cur_wav_set, cur_emo_set, noise_idx_set])
    total_dataloader[dtype] = DataLoader(
        total_dataset[dtype], batch_size=1, shuffle=False, 
        pin_memory=True, num_workers=4,
        collate_fn=utils.collate_fn_wav_lab_mask_env
    )

print("Loading pre-trained ", SSL_TYPE, " model...")

ssl_model = AutoModel.from_pretrained(SSL_TYPE)
if args.fuse_type in ["RT", "SCA"]:
    ssl_model.feature_extractor = net.IdentityTransform()
    
    ssl_feat_model = AutoModel.from_pretrained(SSL_TYPE)
    ssl_feat_model = ssl_feat_model.feature_extractor
    ssl_feat_model.eval(); ssl_feat_model.cuda()
if args.ssl_type == "wav2vec2-large-robust":
    del ssl_model.encoder.layers[12:]
ssl_model.load_state_dict(torch.load(MODEL_PATH+"/final_ssl.pt"))
ssl_model.eval(); ssl_model.cuda()

if args.ssl_type in ["wav2vec2-large-robust", "wavlm-large"]:
    feat_dim = 1024
elif args.ssl_type in ["wav2vec2-base", "wavlm-base", "wavlm-base-plus"]:
    feat_dim = 768
    
if args.fuse_type in ["RH", "RH-avg"]:
    txt_feat_dim = feat_dim
else:
    txt_feat_dim = 512

# env_model = net.EnvEmbeddingEncoder(512, DOMAIN_NUM)
env_model = net.OnehotEncoder(512, DOMAIN_NUM)
env_model.load_state_dict(torch.load(MODEL_PATH+"/final_env.pt"))
env_model.eval(); env_model.cuda()
ser_model = net.EmotionRegression(feat_dim, 512, 1, 3, p=0.5)



if args.fuse_type in ["RT", "SCA"]:
    ssl_feat_model = AutoModel.from_pretrained(SSL_TYPE)
    ssl_model.feature_extractor = net.IdentityTransform()
    ssl_feat_model = ssl_feat_model.feature_extractor
    ssl_feat_model.eval(); ssl_feat_model.cuda()
    if args.fuse_type == "SCA":
        sca_model = net.SCA(txt_feat_dim, 256, feat_dim)
        sca_model.eval(); sca_model.cuda()

ser_model.load_state_dict(torch.load(MODEL_PATH+"/final_ser.pt"))
ser_model.eval(); ser_model.cuda()
env_model.eval(); env_model.cuda()

lm = utils.LogManager()
lm.alloc_stat_type_list(["test_aro", "test_dom", "test_val"])

min_epoch=0
min_loss=1e10

ssl_model.eval()
ser_model.eval() 
total_pred = [] 
total_y = []

for xy_pair in tqdm(total_dataloader["test"]):
    noisy_x = xy_pair[0]; noisy_x=noisy_x.cuda(non_blocking=True).float()
    y = xy_pair[1]; y=y.cuda(non_blocking=True).float()
    mask = xy_pair[2]; mask=mask.cuda(non_blocking=True).float()
    env_x = xy_pair[3]; env_x=env_x.cuda(non_blocking=True).long()
    
    with torch.no_grad():
        
        noisy_x = ssl_feat_model(noisy_x)
        txt_x = env_model(env_x)
        noisy_x = torch.cat([txt_x.transpose(1, 2), noisy_x], dim=2)
        ssl = ssl_model(noisy_x, attention_mask=mask).last_hidden_state
        ssl = torch.mean(ssl, dim=1)
        
        emo_pred = ser_model(ssl)

        total_pred.append(emo_pred)
        total_y.append(y)

# CCC calculation
total_pred = torch.cat(total_pred, 0)
total_y = torch.cat(total_y, 0)
ccc = utils.CCC_loss(total_pred, total_y)
# Logging
lm.add_torch_stat("test_aro", ccc[0])
lm.add_torch_stat("test_dom", ccc[1])
lm.add_torch_stat("test_val", ccc[2])

lm.print_stat()

os.makedirs(os.path.dirname(args.result_path), exist_ok=True)
with open(args.result_path, 'w') as f:
    for attr in ["aro", "dom", "val"]:
        f.write(str(lm.get_stat("test_"+attr))+"\n")