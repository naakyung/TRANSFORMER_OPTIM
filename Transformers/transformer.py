import numpy as np 

import torch
from torch import nn

from .layers.Embed import DataEmbedding_Transformer
from .layers.Patchformer_EncDec import TSTEncoder, Vanila_TSEncoder
from .layers.SelfAttention_Family import PatchFullAttention, MultiHeadAttention


def z_scoring(x):
    
    x[x == 0]     = np.nan
    x_npy         = x.cpu().numpy()
    
    mean_enc      = x[:, :, :].nanmean(dim = 1, keepdim = True).detach()                                            
    std_enc       = torch.sqrt(torch.from_numpy(np.nanvar(x_npy, axis=1, keepdims = True)) + 1e-5).detach().to('cuda:0',  dtype=torch.double)
    # std_enc       = torch.sqrt(torch.nanvar(x[:, :, :], dim = 1, keepdim = True, unbiased = False) + 1e-5).detach() 
    x[:, :, :4]   = x[:, :,  :4] - torch.nanmean(mean_enc[:,:,:4], dim = 2, keepdim = True)
    x[:, :, :4]   = x[:, :,  :4] / torch.nanmean(std_enc [:,:,:4], dim = 2, keepdim=True)
    x[:, :, 4:7]  = x[:, :, 4:7] - torch.nanmean(mean_enc[:,:,4:7], dim = 2, keepdim = True)
    x[:, :, 4:7]  = x[:, :, 4:7] / torch.nanmean(std_enc [:,:,4:7], dim = 2, keepdim=True)
    x[:, :, -1]   = x[:, :,  -1] - mean_enc[:,:,-1]
    x[:, :, -1]   = x[:, :,  -1] / std_enc [:,:,-1]
    
    x = torch.nan_to_num(x, nan = 0)
    return x

class Transformer_Padding(nn.Module):
    def __init__(self, cfg):
        super(Transformer_Padding, self).__init__()

        self.market_tm = 391 
        self.lookback  = lookback = int(cfg.dataset.time_interval)
        self.lookahead = int(cfg.dataset.f_horizon)

        self.chunk_size = chunk_size = cfg.model.patch_len
        assert(lookback % chunk_size == 0)
        self.num_chunks = lookback // chunk_size

        self.num_node    = int(cfg.model.enc_in)         ## 5
        self.c_dim       = None
        self.dropout     = cfg.model.dropout
        
        self.IStand      = cfg.dataset.interval_standard 
        self.smax_out    = cfg.dataset.smax_out
        self.sigmoid_out = cfg.dataset.sigmoid_out 
        self.num_classes = 1 if self.sigmoid_out else cfg.model.num_classes

        ## Model Configs
        self.d_model  = cfg.model.d_model
        self.pe       = cfg.model.pe
        self.learn_pe = cfg.model.learn_pe

        self.n_layers = cfg.model.e_layers
        self.n_heads  = cfg.model.n_heads
        self.d_model  = cfg.model.d_model

        self.d_k  = cfg.model.d_k
        self.d_v  = cfg.model.d_v
        self.d_ff = cfg.model.d_ff

        self.norm = cfg.model.norm

        self.attn_dropout     = cfg.model.attn_dropout
        self.dropout          = cfg.model.dropout
        self.activation       = cfg.model.activation
        self.res_attention    = cfg.model.res_attention
        self.pre_norm         = cfg.model.pre_norm
        self.output_attention = cfg.model.output_attention
        
        self.cfg = cfg
        self._build()
        
    def _build(self):
        
        self.highway_proj       = nn.Linear(self.market_tm, self.lookahead)
        self.input_embedding    = DataEmbedding_Transformer(d_model     = self.d_model, 
                                                            pe          = self.pe,
                                                            learn_pe    = self.learn_pe, 
                                                            padding_idx = 0,
                                                            dropout     = 0.1)
        
        self.encoder            = Vanila_TSEncoder(MultiHeadAttention, 
                                                   self.d_model, 
                                                   self.n_heads, 
                                                   d_k  = self.d_k, 
                                                   d_v  = self.d_v, 
                                                   d_ff = self.d_ff,
                                                   norm = self.norm, 
                                                   attn_dropout     = self.attn_dropout,
                                                   dropout          = self.dropout,
                                                   pre_norm         = self.pre_norm, 
                                                   activation       = self.activation, 
                                                   res_attention    = self.res_attention, 
                                                   n_layers         = self.n_layers, 
                                                   output_attention = self.output_attention)
     
        self.chunk_proj_patch_1 = nn.Linear(self.num_chunks, 1)
        self.mlp_head           = nn.Sequential(nn.BatchNorm1d(self.lookahead * self.num_node),
                                                nn.Linear(self.lookahead * self.num_node, self.num_classes))
 
    def forward(self, x):   
        '''
        Input Shape  : (BS, time_interval, N_vars)
        '''
        
        #### Interval Standardization 
        if self.IStand:
            x         = z_scoring(x) 
            
        B, T, N       = x.size()     
        x             = x.permute(0, 2, 1)                             
        
        highway       = self.highway_proj(x)              
        highway       = highway.permute(0, 2, 1)  
        
        
        z             = self.input_embedding(x.unsqueeze(-1))            # z: [bs * nvars x patch_num x d_model]
        import pdb; pdb.set_trace()
        z             = self.encoder(z)                                  # z: [bs * nvars x patch_num x d_model]
        z             = z.permute(0,2,1)                                 # z: [bs * nvars x d_model x patch_num]
        z             = self.chunk_proj_patch_1(z).squeeze(dim=-1)       # z: [bs * nvars x d_model x 1]
        
        #### MLP-Head 
        out           = z + highway
        out           = torch.flatten(out,start_dim=1)
        out           = self.mlp_head(out)
        
        if self.smax_out:
            return torch.softmax(out, dim=1)
        elif self.sigmoid_out:
            return torch.sigmoid(out).squeeze(1)
        return out
