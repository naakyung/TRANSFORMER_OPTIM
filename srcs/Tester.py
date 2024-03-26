import os
import numpy as np 
import pandas as pd
from omegaconf import DictConfig

import torch
from sklearn.metrics import classification_report, confusion_matrix

import srcs.setting as setting
import srcs.Evaluator as evaluator

def test_model(cfg, model, test_dataloader, device, criterion, 
               epochs, it=-1, output_key = 'smax_out',
               test_with='best', save=False, save_pred=False, use_std_filter=False):
    
    ## Testing framework for FX
    if test_dataloader is None:
        return

    data_type_str    = cfg.dataset.data_type
    float_data_type  = torch.double
    int_data_type    = torch.int64

    if data_type_str == "float64":
        pass
    elif data_type_str == "float32":
        float_data_type = torch.float32
        # int_data_type = torch.int32
        int_data_type = torch.int64
    else:
        print(f"WARNING: [test_model] Given data type is not supported yet: {cfg.dataset.data_type}, automatically set to float64")
        pass

    print(f'INFO: [test_model] Setting data type as {data_type_str}.')

    test_loss            = []
    test_all_probs       = []
    test_all_volats      = []
    test_all_targets     = []
    test_all_predictions = []

    test_n_correct           = 0.
    test_n_total             = 0.

    test_cumsig = cfg.dataset.test_cumsig
    model_name  = cfg.model.model_name

    model.eval()
    with torch.no_grad():
        for batch_idx, dataset in enumerate(test_dataloader):

            ### Setting inputs & targets
            inputs, targets = dataset['X'], dataset['y']
            
            if cfg.dataset.loss_fn in ['CrossEntropy', 'LogitNormLoss', 'FocalLoss']:
                inputs, targets = inputs.to(device, dtype=float_data_type), targets.to(device, dtype=float_data_type)
            elif cfg.dataset.loss_fn in ["BCE", "BCEWithLogits"]:
                inputs, targets = inputs.to(device, dtype=float_data_type), targets.to(device, dtype=int_data_type)
            elif cfg.dataset.loss_fn in ["MSELoss", "RMSELoss"]:
                inputs, targets = inputs.to(device, dtype=float_data_type), targets.to(device, dtype=float_data_type)
                targets = torch.unsqueeze(targets, 1)
                
            
            ### Prediction
            if cfg.dataset.volat_grouping:
                volats_cpu = inputs[:, -1, -1].cpu().numpy()
                test_all_volats      = np.concatenate((test_all_volats, volats_cpu))  
                outputs    = model(inputs[:, :, :-1])
                # g1_idxs   , g10_idxs    = torch.where(inputs[:, -1, -1] == 1)[0], torch.where(inputs[:, -1, -1] == 10)[0]
                # g1_inputs , g10_inputs  = inputs[g1_idxs, :, :4], inputs[g10_idxs, :, :4]
                # g1_targets, g10_targets = targets[g1_idxs], targets[g10_idxs]
                
                # g1_outputs  = model(g1_inputs, volat_group = 1) if (g1_inputs.shape[0] != 0) else g1_targets
                # g10_outputs = model(g10_inputs, volat_group = 10) if (g10_inputs.shape[0] != 0) else g10_targets
                # outputs, targets        = torch.concat((g1_outputs, g10_outputs)), torch.concat((g1_targets, g10_targets))

            else:
                outputs = model(inputs)                               
                
            if cfg.dataset.smax_out:
                lik, preds      = torch.max(outputs, 1)
                lik, preds_cpu  = lik.detach().cpu().numpy(), preds.cpu().numpy()
                targets_cpu     = targets.cpu().numpy()
                
            elif cfg.dataset.sigmoid_out:
                outputs[outputs >= 0.5] = 1
                outputs[outputs < 0.5]  = 0
                preds_cpu   = outputs.detach().cpu().numpy()
                targets_cpu = targets.cpu().numpy().reshape(1, -1)[0]

            test_n_correct += (preds_cpu == targets_cpu).sum().item()
            test_n_total += targets.shape[0]
            
            test_all_probs       = np.concatenate((test_all_probs, lik))
            test_all_targets     = np.concatenate((test_all_targets, targets_cpu))    
            test_all_predictions = np.concatenate((test_all_predictions, preds_cpu))    
        
    ## Print out results
    test_acc = test_n_correct / test_n_total
    conf_test = confusion_matrix(np.array(test_all_targets).flatten(), np.array(test_all_predictions).flatten())
    print(f'\nINFO: [test_model] Test label distribution: {np.unique(test_all_targets, return_counts=True)}')
    print(f'\nINFO: [test_model] Test Confusion Matrix: \n{conf_test}')

    ## Test accuracy for each label
    label_acc_test, label_prec_test = evaluator.print_acc_details('Test', conf_test, test_acc)
    
    if it % 10 == 0:
        evaluator.print_perform_details('Test', md = model_name, it = it, epochs = epochs, targets = test_all_targets, preds = test_all_predictions,  probs = test_all_probs, volats = test_all_volats, acc = test_acc, key = output_key)
        

    # print(f'\nINFO: [Tester] : Epoch : {it}, Test Loss: {loss.item():.4f}\n')

    if save:
        savepath = f'./outputs/test_saved_models/{model_name}/'
        test_with = test_with
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        context = ''
        if os.path.exists(savepath+f'results_{test_with}_model.txt'):
            with open(savepath+f'results_{test_with}_model.txt', 'r') as txt:
                context = txt.readlines()
                
        with open(savepath+f'results_{test_with}_model.txt', 'w') as txt:
            txt.writelines(context)
            txt.writelines(f'Testing with Epoch: {it}\n')
            txt.writelines(classification_report(test_all_targets, test_all_predictions, digits=4))
            txt.writelines(str(conf_test))
            txt.writelines(f'\nTest Accuracy: {test_acc*100:.3f}')
            txt.writelines(f'\nTest Accuracy by label: {label_acc_test}')
            txt.writelines(f'\nTest Precision by label: {label_prec_test}')
            txt.writelines('\n#######################################################################################################\n')

def test(cfg: DictConfig):
    print(f"-----------------------------------------\nconfig: \n{cfg}")
    if cfg.debug:
        print("------------------------------------------------")
        print("Debug mode")
        print("------------------------------------------------")

    if cfg.dataset.name in ["FX", "FX_SMB"]:
        test_dataloader = setting.get_testloader(cfg, cwd='./')
        model = None
        device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

        print(f'INFO: [test] Loading model from: {cfg.model.model_path} ... ')

        if 'traced' in cfg.model.model_path:
            model = torch.jit.load(cfg.model.model_path)
        else:
            model = torch.load(cfg.model.model_path)
        
        if cfg.dataset.data_type == "float64":
            model = model.double()
        elif cfg.dataset.data_type == "float32":
            model = model.float()
        else:
            print(f"ERROR: [train] Given data type is not supported yet: {cfg.dataset.data_type}")
            raise NotImplementedError()
        
        model, device, multi_gpu = setting.prepare_gpus(cfg, model)

        test_model(cfg, model, test_dataloader, device, test_with='saved', save=True, save_pred=cfg.dataset.save_pred)
    else:
        raise NotImplementedError()