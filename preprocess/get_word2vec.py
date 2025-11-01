import os
import numpy as np
import torch.nn.functional as F
from gensim.models import Word2Vec

glove_df = pd.read_csv(glove_path, sep=" ", quoting=3, header=None, index_col=0).astype(float)
print(glove_df.iloc[0])
glove_dict = glove_df.to_dict()

# noise_dir="/media/kyunster/ssd1/RawData/freesound-noise"
# noise_list = os.listdir(noise_dir)
# noise_list += ["clean condition"]

# emb_list=[]

# os.makedirs(os.path.join("data","bert_emb"), exist_ok=True)
# for ntype in noise_list:
#     cur_sentence = template+ntype
#     inputs = tokenizer(text=cur_sentence, return_tensors="pt").to(0)
#     text_embed = model(**inputs).pooler_output
#     out_path = os.path.join("data", "bert_emb", ntype+".npy")
#     text_embed = text_embed.detach().cpu().numpy()
#     np.save(out_path, text_embed)

