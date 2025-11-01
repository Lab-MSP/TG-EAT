import os
import numpy as np
import torch.nn.functional as F

glove_path = "/media/kyunster/ssd1/corpus/GloVe/glove.840B.300d.txt"
noise_dir="/media/kyunster/ssd1/RawData/freesound-noise"
noise_list = os.listdir(noise_dir)

glove_dict = dict()
with open(glove_path, 'r') as f:
    for line in f:
        word = line.split(" ")[0]
        vector = np.array(line.split(" ")[1:], dtype=float)
        glove_dict[word] = vector

os.makedirs(os.path.join("data","glove"), exist_ok=True)
for ntype in noise_list:
    query = ntype.replace("+", " ")
    out_path = os.path.join("data", "glove", ntype+".npy")
    text_embed = glove_dict[query]
    np.save(out_path, text_embed)

