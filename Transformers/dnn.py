import torch 
from torch import nn 
import torch.nn.functional as F 

class DNN(nn.Module):
    def __init__(self, cfg):
        super(DNN, self).__init__() 
        self.num_node    = 60  
        self.num_classes = cfg.model.num_classes

        self.dropout     = cfg.model.dropout
        self.smax        = cfg.dataset.smax
        self.activation  = self.get_activation_fn(cfg.model.activation)

        self.fc1 = nn.Linear(self.num_node, 128, bias= True)
        self.bn1  = nn.BatchNorm1d(num_features = 128)
        
        self.fc2 = nn.Linear(128, 64)
        self.bn2  = nn.BatchNorm1d(num_features = 64)
        self.fc3 = nn.Linear(64,  self.num_classes) 
        
        self.hidden = nn.Sequential(

        
            self.fc1,
            self.bn1,
            self.activation,
            nn.Dropout(p = 0.1),

            self.fc2, 
            self.bn2,
            self.activation,
            nn.Dropout(p = 0.1),

            self.fc3,
            #nn.Sigmoid()
        )

        
    def get_activation_fn(self, activation):
        if callable(activation): return activation()
        elif activation.lower() == "relu": return nn.Tanh()
        elif activation.lower() == "gelu": return nn.GELU()
        raise ValueError(f'{activation} is not available. You can use "relu", "gelu", or a callable') 
    
    def forward(self, x):

        # interval_min = (x[:, :, :].min(1, keepdim = True).values[:, :, 0]).unsqueeze(1)
        # interval_max = (x[:, :, :].max(1, keepdim = True).values[:, :, 0]).unsqueeze(1)
        # x        = (x - interval_min) / (interval_max - interval_min)
        
        
        mean_enc = x[:, :, :].mean(1, keepdim = True).detach() # B x 1 x E
        std_enc  = torch.sqrt(torch.var(x[:, :, :], dim = 1, keepdim = True, unbiased = False) + 1e-5).detach() # B x 1 x E
        x        = (x - mean_enc) / std_enc
        
        x        = x.squeeze(2)
        out = self.hidden(x)
        
        out = torch.sigmoid(out)

        return out