import torch
import torch.nn as nn
import math


def positional_encoding(pe, learn_pe, q_len, d_model):

    if pe == None:
        W_pos = torch.empty((q_len, d_model)) # pe = None and learn_pe = False can be used to measure impact of pe
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False

    elif pe == 'zero':
        W_pos = torch.empty((q_len, 1))
        nn.init.uniform_(W_pos, -0.02, 0.02)

    elif pe == 'zeros':
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)

    elif pe == 'normal' or pe == 'gauss':
        W_pos = torch.zeros((q_len, 1))
        torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)

    elif pe == 'uniform':
        W_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(W_pos, a=0.0, b=0.1)

    elif pe == 'lin1d': W_pos = self.Coord1dPosEncoding(q_len, exponential=False, normalize=True)

    elif pe == 'exp1d': W_pos = self.Coord1dPosEncoding(q_len, exponential=True, normalize=True)

    elif pe == 'lin2d': W_pos = self.Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True)

    elif pe == 'exp2d': W_pos = self.Coord2dPosEncoding(q_len, d_model, exponential=True, normalize=True)

    elif pe == 'sincos': W_pos = self.PositionalEncoding(q_len, d_model, normalize=True)

    else: raise ValueError(f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
        'zeros', 'zero', uniform', 'lin1d', 'exp1d', 'lin2d', 'exp2d', 'sincos', None.)")
    
    return nn.Parameter(W_pos, requires_grad = learn_pe)

def PositionalEncoding(q_len, d_model, normalize=True):
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe

def Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True, eps=1e-3, verbose=False):
    x = .5 if exponential else 1
    i = 0
    for i in range(100):
        cpe = 2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** x) * (torch.linspace(0, 1, d_model).reshape(1, -1) ** x) - 1
        # pv(f'{i:4.0f}  {x:5.3f}  {cpe.mean():+6.3f}', verbose)
        if abs(cpe.mean()) <= eps: break
        elif cpe.mean() > eps: x += .001
        else: x -= .001
        i += 1
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe

def Coord1dPosEncoding(q_len, exponential=False, normalize=True):
    cpe = (2 * (torch.linspace(0, 1, q_len).reshape(-1, 1)**(.5 if exponential else 1)) - 1)
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe


class DataEmbedding_Patchformer(nn.Module):
    def __init__(self, patch_num, patch_len, d_model, pe='zero', learn_pe=True, padding = False, dropout=0.1):
        super(DataEmbedding_Patchformer, self).__init__()

        q_len        = patch_num
        self.W_proj  = nn.Linear(patch_len, d_model)        # Eq 1: projection of feature vectors onto a d-dim vector space
        self.seq_len = q_len

        # Positional encoding
        self.W_pos   = positional_encoding(pe, learn_pe, q_len, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):                                                            # x: [bs x nvars x patch_len x patch_num]

        # Input encoding
        x = x.permute(0,1,3,2)                                                       # x: [bs x nvars x patch_num x patch_len]
        x = self.W_proj(x)                                                           # x: [bs x nvars x patch_num x d_model]

        u = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))      # u: [bs * nvars x patch_num x d_model]
        u = self.dropout(u + self.W_pos)                                             # u: [bs * nvars x patch_num x d_model]

        return u
    
class DataEmbedding_Transformer(nn.Module):
    def __init__(self, d_model, padding_idx=0, pe='zero', learn_pe=True, dropout=0.1):
        super(DataEmbedding_Transformer, self).__init__()

        market_time   = 391
        # self.W_embed  = nn.Embedding(market_time, d_model, padding_idx=padding_idx)      
        self.W_embed  = nn.Linear(1, d_model)                                            # Eq 1 : Input Encoding  
        self.W_pos    = positional_encoding(pe, learn_pe, market_time, d_model)          # Eq 2 : Positional Encoding
        self.dropout  = nn.Dropout(dropout)
    
    def _create_padding_mask(self, seq:torch.Tensor, pad_idx:int):
        """
            입력 tensor의 Padding Index 부분을 False로 반환
        """
        return (seq != pad_idx)
        
    def forward(self, x):                                                           
        """
            [Embedding Layer]
            Input Shape 
                Batch Size x Val_nums x Market_time x 1
            Output Shape 
                Batch Size * Val_nums X Market_time x d_model
        """

        padding_mask = self._create_padding_mask(seq = x, pad_idx = 0)
        w_pos        = self.W_pos * padding_mask
        w_pos        = torch.reshape(w_pos, (w_pos.shape[0] * w_pos.shape[1], w_pos.shape[2], w_pos.shape[3]))
        
        u            = self.W_embed(x) * padding_mask                                                       # x: [bs x nvars x market_tm x d_model]
        u            = torch.reshape(u, (u.shape[0] * u.shape[1], u.shape[2], u.shape[3]))                  # u: [bs * nvars x market_tm x d_model]
        u            = self.dropout(u + w_pos)                                                              # u: [bs * nvars x market_tm x d_model]

        return u