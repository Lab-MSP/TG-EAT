import os
import numpy as np
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel

processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")
model = AutoModel.from_pretrained("laion/clap-htsat-unfused")
model.cuda()
template = "This voice is recorded in "

noise_dir="/media/kyunster/ssd1/RawData/freesound-noise"
noise_list = os.listdir(noise_dir)
noise_list += ["clean condition"]
dns_list = ["washroom", "kitchen", "living room", "sports field", "river", "park", 
             "office", "hallway", "meeting", "subway station", "cafeteria", "restaurant", 
             "traffic intersection", "town square", "cafe terrace", "subway", "bus", "car"]
noise_list.extend(dns_list)

emb_list=[]

os.makedirs(os.path.join("data","clap_emb"), exist_ok=True)
for ntype in noise_list:
    out_path = os.path.join("data", "clap_emb", ntype+".npy")
    if os.path.exists(out_path):
        continue
    cur_sentence = template+ntype
    inputs = processor(text=cur_sentence, return_tensors="pt").to(0)
    text_embed = model.get_text_features(**inputs)
    
    text_embed = text_embed.detach().cpu().numpy()
    np.save(out_path, text_embed)
    
# template2 = "This voice is recorded with "
# noise_list2 = ["babble", "music"]
# for ntype in noise_list2:
#     cur_sentence = template2+ntype+" noise"
#     inputs = processor(text=cur_sentence, return_tensors="pt").to(0)
#     text_embed = model.get_text_features(**inputs)
#     out_path = os.path.join("data", "txt_emb", ntype+".npy")
#     text_embed = text_embed.detach().cpu().numpy()
#     np.save(out_path, text_embed)
    