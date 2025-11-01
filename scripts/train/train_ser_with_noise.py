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
from transformers import AutoModel

from torch.utils.tensorboard import SummaryWriter

# Self-Written Modules
sys.path.append(os.getcwd())
import net
import utils


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default=None)
parser.add_argument("--seed", type=int, default=100)
parser.add_argument("--ssl_type", type=str, default="wavlm-large")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--accumulation_steps", type=int, default=1)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--lr_decay", type=float, default=0.8)
parser.add_argument("--original_model_path", type=str, default="./temp")
parser.add_argument("--model_path", type=str, default="./temp")
parser.add_argument("--train_type", type=str, default="./temp")
args = parser.parse_args()

utils.set_deterministic(args.seed)
SSL_TYPE = utils.get_ssl_type(args.ssl_type)
assert SSL_TYPE != None, print("Invalid SSL type!")
BATCH_SIZE = args.batch_size
ACCUMULATION_STEP = args.accumulation_steps
assert (ACCUMULATION_STEP > 0) and (BATCH_SIZE % ACCUMULATION_STEP == 0)
EPOCHS=args.epochs
LR=args.lr
LR_DECAY=args.lr_decay
MODEL_PATH = args.model_path
PRE_MODEL_PATH = args.original_model_path
os.makedirs(MODEL_PATH+"/param", exist_ok=True)
tensorboard_path = os.path.join(MODEL_PATH, 'logs')
os.makedirs(tensorboard_path, exist_ok = True)
tb_writer = SummaryWriter(tensorboard_path)

from collections import defaultdict
corpus_path = args.data_path
audio_path = os.path.join(corpus_path, "Audios/Audio")
label_path = os.path.join(corpus_path, "Labels", "labels_consensus.csv")

total_dataset=dict()
total_dataloader=dict()
noise_loader=dict()
for dtype in ["train", "dev"]:
    cur_utts, cur_labs = utils.load_adv_emo_label(label_path, dtype)
    cur_wavs = utils.load_audio(audio_path, cur_utts)
    
    wav_mean, wav_std = utils.load_norm_stat(PRE_MODEL_PATH+"/train_norm_stat.pkl")            
    cur_wav_set = utils.WavSet(cur_wavs, wav_mean=wav_mean, wav_std=wav_std)
    cur_wav_set.save_norm_stat(MODEL_PATH+"/train_norm_stat.pkl")
    
    ## Get noise
    noise_filelist_path=f"data/filelist/{dtype}.txt"
    noise_path_list = []
    with open(noise_filelist_path, "r") as f:
        for line in f:
            cur_path, cur_type = line.strip().split("\t")
            noise_path_list.append(cur_path)
    np.random.shuffle(noise_path_list)
    noise_path_list = noise_path_list[:11000]
    noise_wavs = utils.load_audio(noise_path_list)
    noise_wav_set = utils.WavSet(noise_wavs, wav_mean=wav_mean, wav_std=wav_std)
    
    ########################################################
    cur_bs = BATCH_SIZE // ACCUMULATION_STEP if dtype == "train" else 1
    is_shuffle=True if dtype == "train" else False
    ########################################################
    cur_emo_set = utils.ADV_EmoSet(cur_labs)
    total_dataset[dtype] = utils.CombinedSet([cur_wav_set, cur_emo_set])
    total_dataloader[dtype] = DataLoader(
        total_dataset[dtype], batch_size=cur_bs, shuffle=is_shuffle, 
        pin_memory=True, num_workers=4,
        collate_fn=utils.collate_fn_wav_lab_mask
    )
    noise_loader[dtype] = DataLoader(
        noise_wav_set, batch_size=cur_bs, shuffle=is_shuffle, 
        pin_memory=True, num_workers=4,
        collate_fn=utils.collate_fn_wav)



print("Loading pre-trained ", SSL_TYPE, " model...")

ssl_model = AutoModel.from_pretrained(SSL_TYPE)
if args.ssl_type == "wav2vec2-large-robust":
    del ssl_model.encoder.layers[12:]
ssl_model.load_state_dict(torch.load(PRE_MODEL_PATH+"/final_ssl.pt"))
ssl_model.feature_extractor._freeze_parameters()
ssl_model.eval(); ssl_model.cuda()

if args.ssl_type in ["wav2vec2-large-robust", "wavlm-large"]:
    ser_model = net.EmotionRegression(1024, 512, 1, 3, p=0.5)
elif args.ssl_type in ["wav2vec2-base", "wavlm-base", "wavlm-base-plus"]:
    ser_model = net.EmotionRegression(768, 512, 1, 3, p=0.5)
ser_model.load_state_dict(torch.load(PRE_MODEL_PATH+"/final_ser.pt"))
ser_model.eval(); ser_model.cuda()

ssl_opt = torch.optim.Adam(ssl_model.parameters(), LR)
ser_opt = torch.optim.Adam(ser_model.parameters(), LR)

scaler = GradScaler()
ssl_opt.zero_grad(set_to_none=True)
ser_opt.zero_grad(set_to_none=True)

def warmup_lambda(current_step, warmup_steps):
    if current_step > warmup_steps:
        return 1.0
    else:
        return current_step / warmup_steps
warmup_steps = 1000

ssl_sch = optim.lr_scheduler.LambdaLR(ssl_opt, lr_lambda=lambda step: warmup_lambda(step, warmup_steps))
ser_sch = optim.lr_scheduler.LambdaLR(ser_opt, lr_lambda=lambda step: warmup_lambda(step, warmup_steps))

lm = utils.LogManager()
lm.alloc_stat_type_list(["train_aro", "train_dom", "train_val"])
lm.alloc_stat_type_list(["dev_aro", "dev_dom", "dev_val"])

min_epoch=0
min_loss=1e10

snr_list=[12.5, 7.5, 2.5]
for epoch in range(EPOCHS):
    print("Epoch: ", epoch, "LR: ", ser_sch.get_lr())
    lm.init_stat()
    ssl_model.train()
    ser_model.train()    
    batch_cnt = 0

    data_iter = iter(noise_loader["train"])
    for xy_pair in tqdm(total_dataloader["train"]):
        try:
            noise_batch = next(data_iter)
        except StopIteration:
            data_iter = iter(noise_loader["train"])
            noise_batch = next(data_iter)
        
        x = xy_pair[0]; 
        noise_x = noise_batch[0]
        if noise_x.shape[0] < x.shape[0]:
            data_iter = iter(noise_loader["train"])
            noise_batch = next(data_iter)
            noise_x = noise_batch[0]
        noise_x = noise_x[:x.shape[0]]
        
        noisy_x = utils.make_noisy_train_wavs(x, noise_x, snr_list)
        noisy_x=noisy_x.cuda(non_blocking=True).float()
        
        y = xy_pair[1]; y=y.cuda(non_blocking=True).float()
        mask = xy_pair[2]; mask=mask.cuda(non_blocking=True).float()
        
        if args.train_type == "RH":
            with torch.no_grad():
                ssl = ssl_model(noisy_x, attention_mask=mask).last_hidden_state
        with autocast(enabled=True):
            if args.train_type == "RT":
                ssl = ssl_model(noisy_x, attention_mask=mask).last_hidden_state
            ssl = torch.mean(ssl, dim=1)
            emo_pred = ser_model(ssl)
            ccc = utils.CCC_loss(emo_pred, y)
            loss = 1.0 - ccc
            total_loss = torch.sum(loss) / ACCUMULATION_STEP
        scaler.scale(total_loss).backward()
        if (batch_cnt+1) % ACCUMULATION_STEP == 0 or (batch_cnt+1) == len(total_dataloader["train"]):
            if args.train_type == "RT":
                scaler.step(ssl_opt)
            scaler.step(ser_opt)
            scaler.update()
            if args.train_type == "RT":
                ssl_opt.zero_grad(set_to_none=True)
            ser_opt.zero_grad(set_to_none=True)
            
            if args.train_type == "RT":
                ssl_sch.step()  # Update learning rate
            ser_sch.step()  # Update learning rate

        batch_cnt += 1

        # Logging
        lm.add_torch_stat("train_aro", ccc[0])
        lm.add_torch_stat("train_dom", ccc[1])
        lm.add_torch_stat("train_val", ccc[2])   
        
    # ssl_sch.step()
    # ser_sch.step()
    ssl_model.eval()
    ser_model.eval() 
    total_pred = [] 
    total_y = []
    
    data_iter = iter(noise_loader["dev"])
    for xy_pair in tqdm(total_dataloader["dev"]):
        try:
            noise_batch = next(data_iter)
        except StopIteration:
            data_iter = iter(noise_loader["dev"])
            noise_batch = next(data_iter)
            
        x = xy_pair[0]; 
        noise_x = noise_batch[0]
        if noise_x.shape[0] < x.shape[0]:
            data_iter = iter(noise_loader["train"])
            noise_batch = next(data_iter)
            noise_x = noise_batch[0]
        noise_x = noise_x[:x.shape[0]]
        
        noisy_x = utils.make_noisy_train_wavs(x, noise_x, snr_list)
        noisy_x=noisy_x.cuda(non_blocking=True).float()
        
        y = xy_pair[1]; y=y.cuda(non_blocking=True).float()
        mask = xy_pair[2]; mask=mask.cuda(non_blocking=True).float()
        
        with torch.no_grad():
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
    lm.add_torch_stat("dev_aro", ccc[0])
    lm.add_torch_stat("dev_dom", ccc[1])
    lm.add_torch_stat("dev_val", ccc[2])

    # Save model
    lm.print_stat()
    torch.save(ser_model.state_dict(), \
        os.path.join(MODEL_PATH, "param", str(epoch)+"_ser.pt"))
    torch.save(ssl_model.state_dict(), \
        os.path.join(MODEL_PATH, "param", str(epoch)+"_ssl.pt"))

    tb_writer.add_scalar('Train/Arousal', lm.get_stat("train_aro"), epoch)
    tb_writer.add_scalar('Train/Dominance', lm.get_stat("train_dom"), epoch)
    tb_writer.add_scalar('Train/Valence', lm.get_stat("train_val"), epoch)
    tb_writer.add_scalar('Dev/Arousal', lm.get_stat("dev_aro"), epoch)
    tb_writer.add_scalar('Dev/Dominance', lm.get_stat("dev_dom"), epoch)
    tb_writer.add_scalar('Dev/Valence', lm.get_stat("dev_val"), epoch)
    
        
    dev_loss = 3.0 - lm.get_stat("dev_aro") - lm.get_stat("dev_dom") - lm.get_stat("dev_val")
    if min_loss > dev_loss:
        min_epoch = epoch
        min_loss = dev_loss

        print("Save",min_epoch)
        print("Loss",3.0-min_loss)
        for mtype in ["ser", "ssl"]:
            os.system("cp "+os.path.join(MODEL_PATH, "param", "{}_{}.pt".format(min_epoch, mtype)) + \
            " "+os.path.join(MODEL_PATH, "final_{}.pt".format(mtype)))