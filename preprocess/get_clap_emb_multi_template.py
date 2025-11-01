import os
import numpy as np
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel

processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")
model = AutoModel.from_pretrained("laion/clap-htsat-unfused")
model.cuda()
template_list = [
    "",
    "The recording condition is ",
    "The type of background noise is ",
    "A speech is mixed with ",
    "This voice is recorded with the sound of ",
    "The input is recorded with a sound of "
]


noise_dir="/media/kyunster/ssd1/RawData/freesound-noise"
noise_list = os.listdir(noise_dir)
noise_list += ["clean condition"]

emb_list=[]

os.makedirs(os.path.join("data","txt_emb"), exist_ok=True)
for tidx, template in enumerate(template_list):
    print(template)
    os.makedirs(os.path.join("data", "multi_template", str(tidx), "clap_emb"), exist_ok=True)
    for ntype in noise_list:

        cur_sentence = template+ntype
        inputs = processor(text=cur_sentence, return_tensors="pt").to(0)
        text_embed = model.get_text_features(**inputs)
        out_path = os.path.join("data", "multi_template", str(tidx), "clap_emb", ntype+".npy")
        text_embed = text_embed.detach().cpu().numpy()
        np.save(out_path, text_embed)
    