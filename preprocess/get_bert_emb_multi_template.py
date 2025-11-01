import os
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("roberta-large")
model = AutoModel.from_pretrained("roberta-large")
model.cuda()

template_list = [
    "",
    "The recording condition is ",
    "The type of background noise is ",
    "A speech is mixed with ",
    "This voice is recorded with the sound of ",
    "The input is recorded with a sound of "
]


# template = "This voice is recorded in "

noise_dir="/media/kyunster/ssd1/RawData/freesound-noise"
noise_list = os.listdir(noise_dir)
noise_list += ["clean condition"]

emb_list=[]

for tidx, template in enumerate(template_list):
    print(template)
    os.makedirs(os.path.join("data", "multi_template", str(tidx), "bert_emb"), exist_ok=True)
    for ntype in noise_list:
        cur_sentence = template+ntype
        inputs = tokenizer(text=cur_sentence, return_tensors="pt").to(0)
        text_embed = model(**inputs).pooler_output
        out_path = os.path.join("data", "multi_template", str(tidx), "bert_emb", ntype+".npy")
        text_embed = text_embed.detach().cpu().numpy()
        np.save(out_path, text_embed)

# template2 = "This voice is recorded with "
# noise_list2 = ["babble", "music"]
# for ntype in noise_list2:
#     cur_sentence = template2+ntype+" noise"
#     inputs = tokenizer(text=cur_sentence, return_tensors="pt").to(0)
#     text_embed = model(**inputs).pooler_output
#     out_path = os.path.join("data", "bert_emb", ntype+".npy")
#     text_embed = text_embed.detach().cpu().numpy()
#     np.save(out_path, text_embed)