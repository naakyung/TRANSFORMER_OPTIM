## torch imports
import torch
import torch.nn as nn

## Import Dataloader 
from srcs.utils import * 
from srcs.Dataloader import dataLoader

import Transformers.dnn as dnn
import Transformers.transformer as module_arch_vanila_Transformer
import Transformers.Patchformer_ohlc_USDFSMB as module_arch_Patchformer_ohlc_USDFSMB
import Transformers.Patchformer_ohlc_sample as module_arch_Patchformer_ohlc_sample
import Transformers.Patchmixer_ohlc_USDFSMB as module_arch_Patchmixer_ohlc_USDFSMB

## Get Objectives for Training
### (1) Dataloader
def get_dataloader(cfg, cwd = './'):
    return dataLoader(cfg)
    
### (2) Model
def get_model(cfg, tuning_mode):
    
    model            = None
    pretrained_model = None 

    # (2-1) Model Object
    if cfg.model.model_name == "Patchformer_ohlc_USDFSMB":
        model            = module_arch_Patchformer_ohlc_USDFSMB.LightPatchformer_USDFSMB(cfg)
    elif cfg.model.model_name == "Patchformer_ohlc_TimePatch":
        model            = module_arch_Patchformer_ohlc_USDFSMB.LightPatchformer_TimePatch(cfg)
    elif cfg.model.model_name == "Vanila_Transformer":
        model            = module_arch_vanila_Transformer.Transformer_Padding(cfg)
    elif cfg.model.model_name == "Patchformer_ohlc_sample":
        model            = module_arch_Patchformer_ohlc_sample.LightPatchformer_sample(cfg)
    elif cfg.model.model_name == "Patchmixer_ohlc_USDFSMB" or cfg.model.model_name == "Patchmixer_overlap_ohlc_USDFSMB" or cfg.model.model_name == "Patchmixer_ohlc_pad_USDFSMB":
        model            = module_arch_Patchmixer_ohlc_USDFSMB.Model(cfg)
    elif cfg.model.model_name == 'DNN':
        model = dnn.DNN(cfg)
    else:
        raise NotImplementedError(f"Invalid model name: {cfg.model.name}")
    
    assert model != None
        
    # (2-2) Model Datatype
    if cfg.model.model_path != '':
        print(f'INFO: [get_model] pretrained model from: {cfg.model.model_path}...')
        pretrained_model = torch.load(cfg.model.model_path)
        
        if cfg.dataset.data_type == "float64":
            model, pretrained_model = model.double(), pretrained_model.double()
        elif cfg.dataset.data_type == "float32":
            model, pretrained_model = model.float(), pretrained_model.float()
            
        print(f'INFO: [get_model] Loading State-dict ...')
        new_model_dict  = model.state_dict()
        pretrained_dict = pretrained_model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in new_model_dict}
        
        new_model_dict.update(pretrained_dict)
        model.load_state_dict(new_model_dict)
        
        load_list = []
        for mod in pretrained_dict.keys():
            primary_module = mod.split('.')[0]
            if primary_module not in load_list:
                load_list.append(primary_module)
        print(f'INFO: [get_model] Loaded Weights of Modules : {load_list} ...\n')
        
        return model, pretrained_model
        
    else:
        if cfg.dataset.data_type == "float64":
            return model.double(), pretrained_model
        elif cfg.dataset.data_type == "float32":
            return model.float(), pretrained_model

### (3) Loss Function
def get_loss_fn(cfg):
    criterion = None
    if cfg.dataset.loss_fn in ['CrossEntropy', 'LogitNormLoss']:
        criterion = nn.CrossEntropyLoss()
    elif cfg.dataset.loss_fn == 'BCEWithLogits':
        criterion = nn.BCEWithLogitsLoss()
    elif cfg.dataset.loss_fn == 'BCE':
        criterion = nn.BCELoss()
    elif cfg.dataset.loss_fn == "FocalLoss":
        criterion = FocalLoss(cfg.dataset.focal_loss_gamma)
    elif cfg.dataset.loss_fn == "MSELoss":
        criterion = nn.MSELoss()
    elif cfg.dataset.loss_fn == "RMSELoss":
        criterion = RMSELoss()
    return criterion

### (4)Setting Devices for Training
def prepare_gpus(cfg, model):
    gpus = cfg.dataset.gpus

    first_gpu=0
    if torch.cuda.is_available() and len(gpus)>0:
        first_gpu = gpus[0]
    
    multi_gpu = False
    avail_devices = torch.cuda.device_count()
    device = torch.device(f"cuda:{first_gpu}" if torch.cuda.is_available() else "cpu")

    if len(gpus) == 1:
        device = torch.device(f"cuda:{gpus[0]}" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

    elif len(gpus) == 0:
        device = torch.device("cpu")
        model = model.to(device)

    elif avail_devices > 1 and len(gpus) > 1:
        print(f"INFO: [prepare_gpus] Preparing Multi-GPU setting with GPUS: {gpus}...")
        model = nn.DataParallel(model, device_ids=gpus)
        model = model.to(device)
        multi_gpu = True

    elif len(gpus) > avail_devices:
        print(f"ERROR: [prepare_gpus] Make sure you have enough GPU's")
        raise NotImplementedError
    
    else:
        print(f"ERROR: [prepare_gpus] Check your gpu lists again!")
        raise NotImplementedError
    
    return model, device, multi_gpu
