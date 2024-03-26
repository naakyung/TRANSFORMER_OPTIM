import math 
# import pywt
# import talib
import numpy as np

# from statsmodels.tsa.seasonal import STL 

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset

def mmddyyyy2mmddyy(date_mmddyyyy):
    return date_mmddyyyy[:4] + date_mmddyyyy[-2:]

def mmddyy2yyyymmdd(date_mmddyy):
    return "20" + date_mmddyy[-2:] + date_mmddyy[:4]


##########################
#### Dataset
##########################

class Datasets(Dataset):
    def __init__(self, input, label):
        self.input = input
        self.label = label

    def __getitem__(self, idx):
        return {'X': self.input[idx], 'y': self.label[idx]}
    
    def __len__(self):
        return len(self.input)

class features:
    def _fft(self, price_dat):
        
        x1 = np.arange(0, 1, 1 / 30)
        # pred_x = np.array([x1[1] - x1[0] if i!=0 else 0 for i in range(35)])
        # pred_x = pred_x.cumsum()

        # 주파수 생성
        # nfft = 샘플 개수
        nfft = len(price_dat)
        # df = 주파수 증가량
        df = 30 / nfft
        k = np.arange(nfft)
        # f = 0부터~최대주파수까지의 범위
        f = k * df

        # 스펙트럼은 중앙을 기준으로 대칭이 되기 때문에 절반만 구함
        nfft_half = math.trunc(nfft / 2)
        f0 = f[range(nfft_half)]

        # fft 변환 값을 nfft로 나눠주는 것은 Normalization 을 위한 작업 
        # 증폭값을 두 배로 계산(위에서 1/2 계산으로 인해 에너지가 반으로 줄었기 때문)
        fft_y = np.fft.fft(price_dat) / nfft * 2
        fft_y0 = fft_y[range(nfft_half)]

        # 벡터(복소수)의 norm 측정(신호 강도)
        amp = abs(fft_y0)

        # 상위 10개의 주파수
        idxy = np.argsort(-amp)
        # for i in range(10):
        #     print('freq=', f0[idxy[i]], 'amp=', fft_y[idxy[i]])
        
        # 상위 10개의 주파수로 복원해서 원본이랑 비교해보기
        # 10개의 주파수만 더해도 어느정도 복원된것을 확인할 수 있음

        newy                = np.zeros((30,))
        freq_collect        = []
        fit_collect         = []
        for i in range(10):
            freq            = f0[idxy[i]]
            yx              = fft_y[idxy[i]]
            coec            = yx.real
            coes            = yx.imag * -1
            get_frequency   = coec * np.cos(2 * np.pi * freq * x1) + coes * np.sin(2 * np.pi * freq * x1)
            newy            += get_frequency

            freq_collect.append(get_frequency)
            fit_collect.append(newy)

        get_dat = np.concatenate([i.reshape(-1, 1) for i in freq_collect], axis=1)
        get_x   = get_dat[:,1:]

        ### 상위 10개의 주파수 합산 :: [get_dat - (30,10)]
        fit_price = np.sum(get_dat, axis=1) / 2
        residual  = price_dat - fit_price

        return np.concatenate((get_x, residual.reshape(-1,1)), axis=1)
    
    # def _stl(self, price_dat):

    #     result_mul = STL(price_dat, period = 5).fit()
    #     seasonal   = result_mul.seasonal
    #     resid      = result_mul.resid
  
    #     return np.concatenate((seasonal.reshape(-1,1), resid.reshape(-1,1)), axis=1)

    # def _dwt(self, price_dat, dwt_type):
    #     # level 1 
    #     cA, cD = pywt.dwt(price_dat, 'db8')
    #     # level 2 
    #     cA, cD = pywt.dwt(cA, 'db8')
        
    #     ### 반환 유형 (Low freq, High freq)
    #     if dwt_type == 'L':
    #         return cA
    #     elif dwt_type == 'H':
    #         return cD        

##########################
#### LOSS
##########################

## (1) ECE Loss
class ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

## (2) RMSELoss
class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss
    
## (3) FocalLoss 
class FocalLoss(nn.Module):
    
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([1 - alpha, alpha]) ## positive weight
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)                         # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))    # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()