import torch
import torch.nn as nn
import torch.nn.functional as F

class EmotionRegression(nn.Module):
    def __init__(self, *args, **kwargs):
        super(EmotionRegression, self).__init__()
        input_dim = kwargs.get("input_dim", args[0])
        hidden_dim = args[1]
        num_layers = args[2]
        output_dim = args[3]
        p = kwargs.get("dropout", 0.5)

        self.fc=nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(p)
            )
        ])
        for lidx in range(num_layers-1):
            self.fc.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(p)
                )
            )
        self.out = nn.Sequential(
                nn.Linear(hidden_dim, output_dim)
            )
        self.inp_drop = nn.Dropout(p)
    def get_repr(self, x):
        h = self.inp_drop(x)
        for lidx, fc in enumerate(self.fc):
            h=fc(h)
        return h
    
    def forward(self, x):
        h=self.get_repr(x)
        result = self.out(h)
        return result


class ClapTextEmbeddingEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ClapTextEmbeddingEncoder, self).__init__()
        self.emb_dim = kwargs.get("emb_dim", args[0])
        self.out_dim = kwargs.get("out_dim", args[1])
        self.fuse_type = kwargs.get("fuse_type", args[2])
    
        self.text_proj = nn.Linear(self.emb_dim, self.out_dim)
        
            
    def forward(self, txt_emb, inp_ssl=None, mask=None):
        txt_h = self.text_proj(txt_emb).unsqueeze(1)
        if self.fuse_type == "RH":
            assert inp_ssl is not None
            # inp_ssl = inp_ssl*mask
            attn_score = torch.bmm(txt_h, inp_ssl.transpose(1, 2))
            attn_score = F.softmax(attn_score, dim=-1)
            out_h = torch.bmm(attn_score, inp_ssl)
            out_h = out_h.view(out_h.shape[0], out_h.shape[2])
            return out_h
        else:
            return txt_h
class ASTEmbeddingEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ASTEmbeddingEncoder, self).__init__()
        
        self.emb_dim = kwargs.get("emb_dim", args[0])
        self.out_dim = kwargs.get("out_dim", args[1])
        self.fuse_type = kwargs.get("fuse_type", args[2])
    
        self.text_proj = nn.Linear(self.emb_dim, self.out_dim)
        
            
    def forward(self, txt_emb, inp_ssl=None, mask=None):
        txt_h = self.text_proj(txt_emb).unsqueeze(1)
        if self.fuse_type == "RH":
            assert inp_ssl is not None
            # inp_ssl = inp_ssl*mask
            attn_score = torch.bmm(txt_h, inp_ssl.transpose(1, 2))
            attn_score = F.softmax(attn_score, dim=-1)
            out_h = torch.bmm(attn_score, inp_ssl)
            out_h = out_h.view(out_h.shape[0], out_h.shape[2])
            return out_h
        else:
            return txt_h


class TransformerEmbeddingAggregator(nn.Module):
    def __init__(self, *args, **kwargs):
        super(TransformerEmbeddingAggregator, self).__init__()
        self.feat_dim = kwargs.get("feat_dim", 768)
        self.layer_num = kwargs.get("layer_num", 12)
        self.layer_coef = nn.Parameter(torch.ones(self.layer_num+1))
        self.layer_projs = nn.ModuleList([])
        
        for ln in range(self.layer_num+1):
            self.layer_projs.append(nn.LayerNorm(768))
        
    def forward(self, emb_list):
        result_emb = []
        for emb_idx, cur_emb in enumerate(emb_list):
            h = self.layer_projs[emb_idx](cur_emb)
            h = h * (self.layer_coef[emb_idx]/torch.sum(self.layer_coef))
            result_emb.append(torch.mean(h, dim=1))
        result_emb = torch.stack(result_emb, dim=1)
        result_emb = torch.sum(result_emb, dim=1)
        return result_emb
            
class IdentityTransform(nn.Module):
    def __init__(self):
        super(IdentityTransform, self).__init__()
    def forward(self, x):
        return x

import os
import sys
dir_path = os.path.dirname(__file__)
sys.path.append(dir_path)
from .modules.transformer import TransformerEncoder
class SCA(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SCA, self).__init__()
        self.feat_dim = kwargs.get("feat_dim", args[0])
        self.hidden_dim = kwargs.get("hidden_dim", args[1])
        self.out_dim = kwargs.get("out_dim", args[2])
        
        self.num_heads = kwargs.get("num_heads", 16)
        self.num_layers = kwargs.get("num_layers", 1)
        
        self.aud_inp_proj = nn.Linear(self.feat_dim, self.hidden_dim)
        self.txt_inp_proj = nn.Linear(self.feat_dim, self.hidden_dim)
        
        self.sca_encoder = TransformerEncoder(self.hidden_dim, self.num_heads, self.num_layers)
        self.out_proj = nn.Linear(self.hidden_dim, self.out_dim)
        
    def forward(self, x, env_x):
        x = x.transpose(1, 2)
        
        aud_x = self.aud_inp_proj(x)
        txt_x = self.txt_inp_proj(env_x)
        
        aud_x = aud_x.transpose(0, 1)
        txt_x = txt_x.transpose(0, 1)
        
        h = self.sca_encoder(aud_x, x_in_k = txt_x, x_in_v = txt_x)
        h = h.transpose(0, 1)
        o = self.out_proj(h)
        
        return o
    

from torch.autograd import Function


class RevGrad(Function):
    @staticmethod
    def forward(ctx, input_, alpha_):
        ctx.save_for_backward(input_, alpha_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        _, alpha_ = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output * alpha_
        return grad_input, None


revgrad = RevGrad.apply
class GRL(nn.Module):
    def __init__(self, alpha=1., *args, **kwargs):
        """
        A gradient reversal layer.

        This layer has no parameters, and simply reverses the gradient
        in the backward pass.
        """
        super().__init__(*args, **kwargs)

        self._alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, input_):
        return revgrad(input_, self._alpha)
    

class DomainClassifier(nn.Module):
    def __init__(self, *args, **kwargs):
        super(DomainClassifier, self).__init__()
        input_dim = kwargs.get("input_dim", args[0])
        hidden_dim = args[1]
        num_layers = args[2]
        output_dim = args[3]
        p = kwargs.get("dropout", 0.5)

        self.inp_grl = GRL(alpha=0.001)
        self.fc=nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(p)
            )
        ])
        for lidx in range(num_layers-1):
            self.fc.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(p)
                )
            )
        self.out = nn.Sequential(
                nn.Linear(hidden_dim, output_dim), nn.LogSoftmax(dim=1)
            )
        self.inp_drop = nn.Dropout(p)
    def get_repr(self, x):
        h = self.inp_drop(x)
        for lidx, fc in enumerate(self.fc):
            h=fc(h)
        return h
    
    def forward(self, x):
        x = self.inp_grl(x)
        h=self.get_repr(x)
        result = self.out(h)
        return result
    
    
class EnvEmbeddingEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super(EnvEmbeddingEncoder, self).__init__()
        
        self.emb_dim = kwargs.get("emb_dim", args[0])
        self.env_num = kwargs.get("env_num", args[1])
        
        self.env_emb = nn.Embedding(self.env_num, self.emb_dim)
        
    def forward(self, noise_idx):
        return self.env_emb(noise_idx).unsqueeze(1)

class OnehotEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super(OnehotEncoder, self).__init__()
        
        self.emb_dim = kwargs.get("emb_dim", args[0])
        self.env_num = kwargs.get("env_num", args[1])
        self.env_proj = nn.Linear(self.env_num, self.emb_dim)
        
    def forward(self, noise_idx):
        one_hot = torch.zeros(noise_idx.shape[0], self.env_num).cuda()
        one_hot.scatter_(1, noise_idx.unsqueeze(1), 1)
        env_x = self.env_proj(one_hot)
        return env_x.unsqueeze(1)