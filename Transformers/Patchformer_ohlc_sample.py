import torch
from torch import nn

from .layers.Embed import DataEmbedding_Patchformer
from .layers.Patchformer_EncDec import TSTEncoder, IEBlock
from .layers.SelfAttention_Family import PatchFullAttention

def code_box():
    # MINMAXSCALING TEST
    # scaled_x = (x - 1008.8) / (1293.3 - 1008.8)
    # x = torch.cat((scaled_x, interval_scaled_x), axis = 2)
    
    # [Interval] MINMAXSCALING    
    # interval_min = x[:, :, :].min(1, keepdim = True).values[:, :, 0]
    # interval_max = x[:, :, :].max(1, keepdim = True).values[:, :, 0]
    # interval_scaled_x1     = ((x[:, :, 0] - interval_min) / (interval_max - interval_min)).unsqueeze(2)   
    # interval_scaled_x2     = ((x[:, :, 1] - interval_min) / (interval_max - interval_min)).unsqueeze(2)   
    # interval_scaled_x3     = ((x[:, :, 2] - interval_min) / (interval_max - interval_min)).unsqueeze(2)   

    # x = torch.concat([interval_scaled_x1, interval_scaled_x2, interval_scaled_x3], axis = 2)
    
    pass 

class LightPatchformer_sample(nn.Module):
    def __init__(self, cfg):
        super(LightPatchformer_sample, self).__init__()

        self.lookback  = lookback = int(cfg.dataset.time_interval)
        self.lookahead = int(cfg.dataset.f_horizon)

        self.chunk_size = chunk_size = cfg.model.patch_len
        assert(lookback % chunk_size == 0)
        self.num_chunks = lookback // chunk_size

        self.num_node    = int(cfg.model.enc_in)         ## 5
        self.c_dim       = None
        self.dropout     = cfg.model.dropout
        
        self.smax_out    = cfg.dataset.smax_out
        self.sigmoid_out = cfg.dataset.sigmoid_out 
        self.use_Inorm   = False
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
        self.enc_embedding    = DataEmbedding_Patchformer(patch_num = self.num_chunks, 
                                                            patch_len = self.chunk_size,
                                                            d_model   = self.d_model, 
                                                            pe        = self.pe,
                                                            learn_pe  = self.learn_pe, 
                                                            dropout   = 0.1)
        
        self.enc_embedding_2    = DataEmbedding_Patchformer(patch_num = self.num_chunks, 
                                                            patch_len = self.chunk_size,
                                                            d_model   = self.d_model, 
                                                            pe        = self.pe,
                                                            learn_pe  = self.learn_pe, 
                                                            dropout   = 0.1)

        self.encoder          = TSTEncoder(PatchFullAttention, 
                                             self.num_chunks, 
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
        
        self.encoder_2          = TSTEncoder(PatchFullAttention, 
                                             self.num_chunks, 
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

        self.chunk_proj_1       = nn.Linear(self.num_chunks, 1)
        self.chunk_proj_2       = nn.Linear(self.num_chunks, 1)
        
        self.layer_3            = IEBlock(input_dim  = self.d_model, #  * 2
                                          hid_dim    = self.d_model,
                                          output_dim = self.lookahead,
                                          num_node   = self.num_node,
                                          c_dim      = self.c_dim)
        
        self.act_tanh  = nn.Tanh()
        self.ar        = nn.Linear(self.lookback, self.lookahead)
        
        self.mlp_head = nn.Sequential(nn.BatchNorm1d(self.lookahead * self.num_node),
                                      nn.Linear(self.lookahead * self.num_node, self.num_classes),
                                      # nn.Dropout(0.1),
                                      # nn.ReLU(), 
                                      # nn.Linear(int(cfg.model.d_model/2), self.num_classes)
        )
                                      
   
    def forward(self, x): 
        '''
        Input Shape  : (BS, time_interval, N_vars)
        '''
        
        # [Interval] Standardization 
        mean_enc      = x[:, :, :].mean(dim = 1, keepdim = True).detach()                                            
        std_enc       = torch.sqrt(torch.var(x[:, :, :], dim = 1, keepdim = True, unbiased = False) + 1e-5).detach() 
        x[:, :, :4]   = x[:, :,  :4] - torch.mean(mean_enc[:,:,:4], dim = 2, keepdim = True)
        x[:, :, :4]   = x[:, :,  :4] / torch.mean(std_enc [:,:,:4], dim = 2, keepdim=True)
        x[:, :, 4:7]  = x[:, :, 4:7] - torch.mean(mean_enc[:,:,4:7], dim = 2, keepdim = True)
        x[:, :, 4:7]  = x[:, :, 4:7] / torch.mean(std_enc [:,:,4:7], dim = 2, keepdim=True)
        x[:, :, 7]    = x[:, :,  7] - mean_enc[:,:,7]
        x[:, :, 7]    = x[:, :,  7] / std_enc [:,:,7]
        x[:, :, -1]   = x[:, :,  -1] - mean_enc[:,:,-1]
        x[:, :, -1]   = x[:, :,  -1] / std_enc [:,:,-1]
        
        B, T, N       = x.size()     
        highway = self.ar(x.permute(0, 2, 1))                                # (B,time_interval,4) > (B, 4, 5(ahead)) 
        highway = highway.permute(0, 2, 1)                                   # (B,time_interval,4) > (B, 5(ahead), 4(n_vars))
        
        #### SAMPLING 1 : continuous sampling
        #### Patching
        x1      = x.reshape(B, self.num_chunks, self.chunk_size, N)          ## B x T/C x C x N
        x1      = x1.permute(0, 3, 2, 1)                                     ## B x N   x C x T/C
        #### Projection + Embedding
        n_vars  = x1.shape[1]
        x1      = self.enc_embedding(x1)                                     # z: [bs * nvars x patch_num x d_model]
        #### Transformer Encoder
        x1      = self.encoder(x1)                                           # z: [bs * nvars x patch_num x d_model]
        x1      = x1.permute(0,2,1)                                          # z: [bs * nvars x d_model x patch_num]
        x1      = self.chunk_proj_1(x1).squeeze(dim=-1)                      # z: [bs * nvars x d_model x 1]
        
        #### SAMPLING 2 : interval sampling
        x2      = x.reshape(B, self.chunk_size, self.num_chunks, N)          ## B x C  x T/C x N
        x2      = x2.permute(0, 3, 1, 2)                                     ## B x N  x C   x T/C

        n_vars = x2.shape[1]
        x2 = self.enc_embedding_2(x2)                                        # z: [bs * nvars x patch_num x d_model]
        x2 = self.encoder_2(x2)                                              # z: [bs * nvars x patch_num x d_model]
        x2 = x2.permute(0,2,1)                                               # z: [bs * nvars x d_model x patch_num]
        x2 = self.chunk_proj_2(x2).squeeze(dim=-1)                           # z: [bs * nvars x d_model x 1]

        #### Information Exchange Block
        # x3 = torch.cat([x1, x2], dim=-1)                                  # [36864, 256]
        x3 = x1

        x3 = x3.reshape(B, N, -1)                                            # z: [bs * nvars x 2*d_model]  [4096, 9, 256]
        x3 = x3.permute(0, 2, 1)                                             # [4096, 256, 9]

        out = self.layer_3(x3)                                               # (bs, 5, 30) + (B, 5(ahead), 4(n_vars))
        
        out = out + highway
        
        out = torch.flatten(out,start_dim=1)
        out = self.mlp_head(out)
        
        if self.smax_out:
            return torch.softmax(out, dim=1)
        elif self.sigmoid_out:
            return torch.sigmoid(out).squeeze(1)
        else:
            return out # ,(interval_min, interval_max)
            
