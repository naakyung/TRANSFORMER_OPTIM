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
def z_scoring(x):
    mean_enc      = x[:, :, :].mean(dim = 1, keepdim = True).detach()                                            
    std_enc       = torch.sqrt(torch.var(x[:, :, :], dim = 1, keepdim = True, unbiased = False) + 1e-5).detach() 
    x[:, :, :4]   = x[:, :,  :4] - torch.mean(mean_enc[:,:,:4], dim = 2, keepdim = True)
    x[:, :, :4]   = x[:, :,  :4] / torch.mean(std_enc [:,:,:4], dim = 2, keepdim=True)
    x[:, :, 4:7]  = x[:, :, 4:7] - torch.mean(mean_enc[:,:,4:7], dim = 2, keepdim = True)
    x[:, :, 4:7]  = x[:, :, 4:7] / torch.mean(std_enc [:,:,4:7], dim = 2, keepdim=True)
    x[:, :, -1]   = x[:, :,  -1] - mean_enc[:,:,-1]
    x[:, :, -1]   = x[:, :,  -1] / std_enc [:,:,-1]
    return x

class LightPatchformer_USDFSMB(nn.Module):
    def __init__(self, cfg):
        super(LightPatchformer_USDFSMB, self).__init__()

        self.lookback  = lookback = int(cfg.dataset.time_interval)
        self.lookahead = int(cfg.dataset.f_horizon)

        self.chunk_size = chunk_size = cfg.model.patch_len
        assert(lookback % chunk_size == 0)
        self.num_chunks = lookback // chunk_size

        self.num_node    = int(cfg.model.enc_in)         ## 5
        self.dropout     = cfg.model.dropout
        
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
        self.enc_embedding_1    = DataEmbedding_Patchformer(patch_num = self.num_chunks, 
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
        
        self.encoder_1          = TSTEncoder(PatchFullAttention, 
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
        
        self.layer_3            = IEBlock(input_dim  = self.d_model * 2,
                                          hid_dim    = self.d_model,
                                          output_dim = self.lookahead,
                                          num_node   = self.num_node)
        
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
        # x[:, :, 7]    = x[:, :,  7] - mean_enc[:,:,7]
        # x[:, :, 7]    = x[:, :,  7] / std_enc [:,:,7]
        x[:, :, -1]   = x[:, :,  -1] - mean_enc[:,:,-1]
        x[:, :, -1]   = x[:, :,  -1] / std_enc [:,:,-1]
        
        import pdb; pdb.set_trace()
        B, T, N       = x.size()     
        highway = self.ar(x.permute(0, 2, 1))                                # (B,time_interval,4) > (B, 4, 5(ahead)) 
        highway = highway.permute(0, 2, 1)                                   # (B,time_interval,4) > (B, 5(ahead), 4(n_vars))
        
        #### SAMPLING 1 : continuous sampling
        x1      = x.reshape(B, self.num_chunks, self.chunk_size, N)          ## B x T/C x C x N
        x1      = x1.permute(0, 3, 2, 1)                                     ## B x N   x C x T/C
        
        n_vars  = x1.shape[1]
        x1      = self.enc_embedding_1(x1)                                   # z: [bs * nvars x patch_num x d_model]
        x1      = self.encoder_1(x1)                                         # z: [bs * nvars x patch_num x d_model]
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
        x3 = torch.cat([x1, x2], dim=-1)
        
        x3 = x3.reshape(B, N, -1)                                            # z: [bs * nvars x 2*d_model]
        x3 = x3.permute(0, 2, 1)
        
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

class LightPatchformer_TimePatch(nn.Module):
    def __init__(self, cfg):
        super(LightPatchformer_TimePatch, self).__init__()

        # Large Sample (l), Small Sample (s)
        self.l_lookback = l_lookback = int(cfg.dataset.time_interval)
        self.s_lookback = s_lookback = l_lookback // 3
        self.lookahead = int(cfg.dataset.f_horizon)
        
        self.l_chunk_size = l_chunk_size = cfg.model.patch_len
        assert(l_lookback % l_chunk_size == 0)
        self.s_chunk_size = s_chunk_size = l_chunk_size if s_lookback % l_chunk_size == 0 else 5  
        self.l_num_chunks = l_lookback // l_chunk_size
        self.s_num_chunks = s_lookback // s_chunk_size

        self.num_node    = int(cfg.model.enc_in)         ## 5
        self.c_dim       = None
        self.dropout     = cfg.model.dropout
        
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
        
        self.l_enc_embed_x1 = DataEmbedding_Patchformer(patch_num = self.l_num_chunks, 
                                                        patch_len = self.l_chunk_size,
                                                        d_model   = self.d_model, 
                                                        pe        = self.pe,
                                                        learn_pe  = self.learn_pe, 
                                                        dropout   = 0.1)
        self.l_enc_embed_x2 = DataEmbedding_Patchformer(patch_num = self.l_num_chunks, 
                                                        patch_len = self.l_chunk_size,
                                                        d_model   = self.d_model, 
                                                        pe        = self.pe,
                                                        learn_pe  = self.learn_pe, 
                                                        dropout   = 0.1)
        
        self.s_enc_embed_x1 = DataEmbedding_Patchformer(patch_num = self.s_num_chunks, 
                                                        patch_len = self.s_chunk_size,
                                                        d_model   = self.d_model, 
                                                        pe        = self.pe,
                                                        learn_pe  = self.learn_pe, 
                                                        dropout   = 0.1)
        self.s_enc_embed_x2 = DataEmbedding_Patchformer(patch_num = self.s_num_chunks, 
                                                        patch_len = self.s_chunk_size,
                                                        d_model   = self.d_model, 
                                                        pe        = self.pe,
                                                        learn_pe  = self.learn_pe, 
                                                        dropout   = 0.1)        


        self.l_encoder_x1   = TSTEncoder(PatchFullAttention, 
                                         self.l_num_chunks, 
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
        self.l_encoder_x2   = TSTEncoder(PatchFullAttention, 
                                         self.l_num_chunks, 
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

        self.s_encoder_x1   = TSTEncoder(PatchFullAttention, 
                                         self.s_num_chunks, 
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
        self.s_encoder_x2   = TSTEncoder(PatchFullAttention, 
                                         self.s_num_chunks, 
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

        self.l_chunk_proj_x1= nn.Linear(self.l_num_chunks, 1)
        self.l_chunk_proj_x2= nn.Linear(self.l_num_chunks, 1)
        self.s_chunk_proj_x1= nn.Linear(self.s_num_chunks, 1)
        self.s_chunk_proj_x2= nn.Linear(self.s_num_chunks, 1)
        
        self.IEBlock_layer  = TimePatch_IEBlock(input_dim  = self.d_model * 2,
                                                hid_dim    = self.d_model,
                                                output_dim = self.lookahead,
                                                num_node   = self.num_node,
                                                c_dim      = self.c_dim)

        self.ar             = nn.Linear(self.l_lookback, self.lookahead)
        self.mlp_head       = nn.Sequential(nn.BatchNorm1d(self.lookahead * self.num_node),
                                            nn.Linear(self.lookahead * self.num_node, self.num_classes))
                                        
    def forward(self, x): 
        '''
        Input Shape  : (BS, time_interval, N_vars)
        '''
        
        # [Interval] Standardization 
        mean_enc      = x[:, :, :].mean(dim = 1, keepdim = True).detach()                                            
        std_enc       = torch.sqrt(torch.var(x[:, :, :], dim = 1, keepdim = True, unbiased = False) + 1e-5).detach() 
        x[:, :, :4] = x[:, :, :4] - torch.mean(mean_enc[:,:,:4], dim = 2, keepdim = True)
        x[:, :, :4] = x[:, :, :4] / torch.mean(std_enc [:,:,:4], dim = 2, keepdim=True)
        x[:, :, 4:] = x[:, :, 4:] - torch.mean(mean_enc[:,:,4:], dim = 2, keepdim = True)
        x[:, :, 4:] = x[:, :, 4:] / torch.mean(std_enc [:,:,4:], dim = 2, keepdim=True)
        
        large_sample_x, small_sample_x = x[:, :, :], x[:, -self.s_lookback:, :]
        
        
        #### (Branch-01) Large Time
        l_B, l_T, l_N = large_sample_x.size()     
        highway = self.ar(large_sample_x.permute(0, 2, 1))                              # (B,time_interval,4) > (B, 4, 5(ahead)) 
        highway = highway.permute(0, 2, 1)                                              # (B,time_interval,4) > (B, 5(ahead), 4(n_vars))
        
        #### SAMPLING 1 : continuous sampling
        l_x1      = large_sample_x.reshape(l_B, self.l_num_chunks, self.l_chunk_size, l_N)    ## B x T/C x C x N
        l_x1      = l_x1.permute(0, 3, 2, 1)                                                ## B x N   x C x T/C

        l_x1      = self.l_enc_embed_x1(l_x1)                                             # z: [bs * nvars x patch_num x d_model]
        l_x1      = self.l_encoder_x1(l_x1)                                                 # z: [bs * nvars x patch_num x d_model]
        l_x1      = l_x1.permute(0,2,1)                                                     # z: [bs * nvars x d_model x patch_num]
        l_x1      = self.l_chunk_proj_x1(l_x1).squeeze(dim=-1)                              # z: [bs * nvars x d_model x 1]
        
        #### SAMPLING 2 : interval sampling
        l_x2      = x.reshape(l_B, self.l_chunk_size, self.l_num_chunks, l_N)                       ## B x C  x T/C x N
        l_x2      = l_x2.permute(0, 3, 1, 2)                                                ## B x N  x C   x T/C

        l_x2 = self.l_enc_embed_x2(l_x2)                                                  # z: [bs * nvars x patch_num x d_model]
        l_x2 = self.l_encoder_x2(l_x2)                                                      # z: [bs * nvars x patch_num x d_model]
        l_x2 = l_x2.permute(0,2,1)                                                          # z: [bs * nvars x d_model x patch_num]
        l_x2 = self.l_chunk_proj_x2(l_x2).squeeze(dim=-1)                                    # z: [bs * nvars x d_model x 1]

        l_x = torch.cat([l_x1, l_x2], dim=-1)
        l_x = l_x.reshape(l_B, l_N, -1)                                                     # z: [bs * nvars x 2*d_model]
        l_x = l_x.permute(0, 2, 1)
        
        
        #### (Branch-02) Small Time
        s_B, s_T, s_N = small_sample_x.size()     
        
        #### SAMPLING 1 : continuous sampling
        s_x1      = small_sample_x.reshape(s_B, self.s_num_chunks, self.s_chunk_size, s_N)  ## B x T/C x C x N
        s_x1      = s_x1.permute(0, 3, 2, 1)                                                ## B x N   x C x T/C

        s_x1      = self.s_enc_embed_x1(s_x1)                                               # z: [bs * nvars x patch_num x d_model]
        s_x1      = self.s_encoder_x1(s_x1)                                                 # z: [bs * nvars x patch_num x d_model]
        s_x1      = s_x1.permute(0,2,1)                                                     # z: [bs * nvars x d_model x patch_num]
        s_x1      = self.s_chunk_proj_x1(s_x1).squeeze(dim=-1)                              # z: [bs * nvars x d_model x 1]
        
        #### SAMPLING 2 : interval sampling
        s_x2      = small_sample_x.reshape(s_B, self.s_chunk_size, self.s_num_chunks, s_N)  ## B x C  x T/C x N
        s_x2      = s_x2.permute(0, 3, 1, 2)                                                ## B x N  x C   x T/C

        s_x2      = self.s_enc_embed_x2(s_x2)                                             # z: [bs * nvars x patch_num x d_model]
        s_x2      = self.s_encoder_x2(s_x2)                                                 # z: [bs * nvars x patch_num x d_model]
        s_x2      = s_x2.permute(0,2,1)                                                     # z: [bs * nvars x d_model x patch_num]
        s_x2      = self.s_chunk_proj_x2(s_x2).squeeze(dim=-1)                              # z: [bs * nvars x d_model x 1]
        
        s_x = torch.cat([s_x1, s_x2], dim=-1)
        s_x = s_x.reshape(s_B, s_N, -1)                                                     # z: [bs * nvars x 2*d_model]
        s_x = s_x.permute(0, 2, 1)
        
        #### Information Exchange Block
        concated_output = self.IEBlock_layer(l_x, s_x)                                  # (bs, 5, 30) + (B, 5(ahead), 4(n_vars))        
        
        out = concated_output + highway
        out = torch.flatten(out,start_dim=1)
        out = self.mlp_head(out)
        
        if self.smax_out:
            return torch.softmax(out, dim=1)
        elif self.sigmoid_out:
            return torch.sigmoid(out).squeeze(1)
        else:
            return out # ,(interval_min, interval_max)
            
class LightPatchformer_Padding(nn.Module):
    def __init__(self, cfg):
        super(LightPatchformer_Padding, self).__init__()

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
        
        self.highway_proj       = nn.Linear(self.lookback, self.lookahead)
        
        self.embedding_patch_1  = DataEmbedding_Patchformer(patch_num = self.num_chunks, 
                                                            patch_len = self.chunk_size,
                                                            d_model   = self.d_model, 
                                                            pe        = self.pe,
                                                            learn_pe  = self.learn_pe, 
                                                            padding   = True,
                                                            dropout   = 0.1)
        
        self.embedding_patch_2  = DataEmbedding_Patchformer(patch_num = self.num_chunks, 
                                                            patch_len = self.chunk_size,
                                                            d_model   = self.d_model, 
                                                            pe        = self.pe,
                                                            learn_pe  = self.learn_pe, 
                                                            padding   = True, 
                                                            dropout   = 0.1)
        
        self.encoder_patch_1    = TSTEncoder(PatchFullAttention, 
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
        
        self.encoder_patch_2    = TSTEncoder(PatchFullAttention, 
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
        
        self.chunk_proj_patch_1 = nn.Linear(self.num_chunks, 1)
        self.chunk_proj_patch_2 = nn.Linear(self.num_chunks, 1)
        
        self.IE_layer           = IEBlock(input_dim  = self.d_model * 2,
                                          hid_dim    = self.d_model,
                                          output_dim = self.lookahead,
                                          num_node   = self.num_node)
    
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
        highway       = self.highway_proj(x.permute(0, 2, 1))              
        highway       = highway.permute(0, 2, 1)                       
        
        #### S-Branch : small-Patch
        patch_x1      = x.reshape(B, self.num_chunks, self.chunk_size, N)          ## B x T/C x C x N
        patch_x1      = patch_x1.permute(0, 3, 2, 1)                               ## B x N   x C x T/C

        patch_z1      = self.embedding_patch_1(patch_x1)                          # z: [bs * nvars x patch_num x d_model]
        patch_z1      = self.encoder_patch_1(patch_z1)                            # z: [bs * nvars x patch_num x d_model]
        patch_z1      = patch_z1.permute(0,2,1)                                    # z: [bs * nvars x d_model x patch_num]
        patch_z1      = self.chunk_proj_patch_1(patch_z1).squeeze(dim=-1)          # z: [bs * nvars x d_model x 1]
        
        #### L-Branch : Large-Patch
        patch_x2      = x.reshape(B, self.chunk_size, self.num_chunks, N)          ## B x C  x T/C x N
        patch_x2      = patch_x2.permute(0, 3, 1, 2)                               ## B x N  x C   x T/C

        patch_z2      = self.embedding_patch_2(patch_x2)                             # z: [bs * nvars x patch_num x d_model]
        patch_z2      = self.encoder_patch_2(patch_z2)                            # z: [bs * nvars x patch_num x d_model]
        patch_z2      = patch_z2.permute(0,2,1)                                    # z: [bs * nvars x d_model x patch_num]
        patch_z2      = self.chunk_proj_patch_2(patch_z2).squeeze(dim=-1)          # z: [bs * nvars x d_model x 1]

        #### Information Exchange of Branches 
        z             = torch.cat([patch_z1, patch_z2], dim=-1)
        z             = z.reshape(B, N, -1)                                        # z: [bs * nvars x 2*d_model]
        z             = z.permute(0, 2, 1)
        z             = self.IE_layer(z)                                           # (bs, 5, 30) + (B, 5(ahead), 4(n_vars))
        
        #### MLP-Head 
        out           = z + highway
        out           = torch.flatten(out,start_dim=1)
        out           = self.mlp_head(out)
        
        if self.smax_out:
            return torch.softmax(out, dim=1)
        elif self.sigmoid_out:
            return torch.sigmoid(out).squeeze(1)
        return out
