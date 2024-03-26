import os 
import copy
import random
from tqdm import tqdm

import numpy as np 
import pandas as pd

import torch
from sklearn.metrics import confusion_matrix

## hydra imports
from ray.air import session

import srcs.setting as setting
import srcs.Tester as testor
import srcs.Evaluator as evaluator 

def custom_loss_fn(output, target):
    tensor = -(target*torch.log(output) + (1-target)*torch.log(1-output)) 
    return torch.mean(tensor)

def train_for_tune(cfg, dataloader, model, criterion, optimizer, scheduler, epochs):
    
    loaded_checkpoint = session.get_checkpoint()
    if loaded_checkpoint:
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(os.path.join(loaded_checkpoint_dir, "checkpoint.pt"))
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)
    
    
    #### Set Variables 
    ## (1) Dataload & Devices
    dataloader.load_data()
    model, device, multi_gpu = setting.prepare_gpus(cfg, model)
    
    ## (2) Datatypes 
    data_type_str   = cfg.dataset.data_type
    float_data_type = torch.double
    int_data_type   = torch.long
    # print(f'INFO: [train_model] Setting data type as {data_type_str}.')

    ## (3) loss.
    train_losses     = np.zeros(epochs)
    val_losses       = np.zeros(epochs)
    best_val_loss    = np.inf
    best_val_epoch   = 0
    best_model       = None
    last_model       = None
    
    
    print("[Setting For Hyperparameter-Tuning.]")
    print(f'[INFO] [Model] Model name : {cfg.model.model_name}')
    print(f'[INFO] [Parameter] epoch : {epochs}, lr : {cfg.model.lr}, input_feature_cnt : {cfg.model.enc_in}')
    print(f'[INFO] [Act-Fn : {cfg.model.activation}] \t [Loss-Fn : {cfg.dataset.loss_fn}].')
    
    ## Save the config file
    save_output_dir = f'{cfg.dataset.system_dir}/outputs/test_saved_models/{cfg.model.model_name}'
    with open(f'{save_output_dir}/hydra_config.txt', 'w') as txt:
        txt.writelines(str(cfg))
        
    #############################################################
    #                         TRAINING                          #
    #############################################################
    
    summarys = None
    for it in tqdm(range(epochs)):
        model.train()
        
        train_loss          = []
        train_total_targets = []
        train_total_preds   = []
        
        # Train accuracy initialization
        train_n_correct           = 0.
        train_n_total             = 0.
        iter_count                = 0.

        for batch_idx, dataset in enumerate(dataloader.train_dataloader):
            
            ### Setting inputs & targets
            inputs, targets = dataset['X'], dataset['y']
            
            if cfg.dataset.loss_fn in ['CrossEntropy', 'LogitNormLoss', 'FocalLoss']:
                inputs, targets = inputs.to(device,  dtype=float_data_type), targets.to(device, dtype=int_data_type)
            elif cfg.dataset.loss_fn in ["BCE", "BCEWithLogits"]:
                inputs, targets = inputs.to(device, dtype=float_data_type), targets.to(device, dtype=float_data_type)
            elif cfg.dataset.loss_fn in ["MSELoss", "RMSELoss"]:
                inputs, targets = inputs.to(device, dtype=float_data_type), targets.to(device, dtype=float_data_type)
                targets = torch.unsqueeze(targets, 1)

            optimizer.zero_grad()

            ### Prediction & Backward  
            outputs = model(inputs)                             # , interval_minmax = model(inputs)
                                                                # interval_min = torch.squeeze(interval_minmax[0], 2)
                                                                # interval_max = torch.squeeze(interval_minmax[1], 2)

                                                                # MINMAXSCALING 
                                                                # scaled_targets = (targets - interval_min) / (interval_max - interval_min)
                                                                # targets = scaled_targets

                                                                # Rescaling 후, loss 계산 
                                                                # outputs = outputs * (interval_max - interval_min) + interval_min
            
            if cfg.dataset.smax_out:
                loss            = criterion(outputs, targets)
            elif cfg.dataset.sigmoid_out: 
                loss            = criterion(outputs.squeeze(1), targets)   
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)    
            optimizer.step()
            train_loss.append(loss.item())

            if cfg.dataset.smax_out:
                lik, preds      = torch.max(outputs, 1)
                lik, preds_cpu  = lik.detach().cpu().numpy(), preds.cpu().numpy()
                targets_cpu     = targets.cpu().numpy()
                
            elif cfg.dataset.sigmoid_out:
                outputs[outputs >= 0.5] = 1
                outputs[outputs < 0.5]  = 0
                preds       = outputs.squeeze(1)
                preds_cpu   = preds.detach().cpu().numpy()
                targets_cpu = targets.cpu().numpy().reshape(1, -1)[0]
            
            
            ### Save Results Data ###
            train_n_correct     += (preds == targets).sum().item()
            train_n_total       += targets.shape[0]
            
            train_total_targets = np.concatenate((train_total_targets, targets_cpu))
            train_total_preds   = np.concatenate((train_total_preds, preds_cpu))
            
            print(f'class_1_len : {targets_cpu.sum()}, total_len : {targets_cpu.shape[0]}, loss : {loss.item()}')
            

            ### Print INFOS ###     
            # if iter_count % 10 == 0:
            #     print("INFO: [train_model] Batch : ", iter_count, "\t Train_acc : ", train_n_correct/train_n_total, f"\t Mean Train_loss : {np.mean(train_loss):.5f}")       
            iter_count += 1

        ### Epoch loss & eval ###
        train_loss       = np.mean(train_loss)
        train_losses[it] = train_loss
        
        # dataloader.suffle_train_dset()
        conf_train = confusion_matrix(train_total_targets, train_total_preds)

        print(f'\n\nINFO: [train_model] Training label distribution: {np.unique(train_total_targets, return_counts=True)}')
        print(f'\nINFO: [train_model] Training Confusion Matrix: \n{conf_train}')

        train_acc = train_n_correct/train_n_total
        label_acc_train, label_prec_train = evaluator.print_acc_details('Train', conf_train, train_acc)
        
        #############################################################
        #                         Validation                        #
        #############################################################
        with torch.no_grad():
            model.eval()
            
            for batch_idx, dataset in enumerate(dataloader.val_dataloader):
                inputs, targets = dataset['X'], dataset['y']
                
                if cfg.dataset.loss_fn in ['CrossEntropy', 'LogitNormLoss', 'FocalLoss']:
                    inputs, targets = inputs.to(device,  dtype=float_data_type), targets.to(device, dtype=int_data_type)
                elif cfg.dataset.loss_fn in ["BCE", "BCEWithLogits"]:
                    inputs, targets = inputs.to(device, dtype=float_data_type), targets.to(device, dtype=float_data_type)
                elif cfg.dataset.loss_fn in ["MSELoss", "RMSELoss"]:
                    inputs, targets = inputs.to(device, dtype=float_data_type), targets.to(device, dtype=float_data_type)
                    targets = torch.unsqueeze(targets, 1)
        
                outputs = model(inputs)                                     # , interval_minmax = model(inputs)    
                                                                            # interval_min = torch.squeeze(interval_minmax[0], 2)
                                                                            # interval_max = torch.squeeze(interval_minmax[1], 2)

                                                                            # MINMAXSCALING 
                                                                            # scaled_targets = (targets - interval_min) / (interval_max - interval_min)
                                                                            # targets = scaled_targets
                                                                            
                                                                            # Rescaling 후, loss 계산 
                                                                            # outputs = outputs * (interval_max - interval_min) + interval_min
            
                if cfg.dataset.smax_out:
                    loss            = criterion(outputs, targets)
                elif cfg.dataset.sigmoid_out: 
                    loss            = criterion(outputs.squeeze(1), targets)   
                
                val_loss.append(loss.item())
            
                if cfg.dataset.smax_out:
                    lik, preds      = torch.max(outputs, 1)
                    lik, preds_cpu  = lik.detach().cpu().numpy(), preds.cpu().numpy()
                    targets_cpu     = targets.cpu().numpy()
                    
                elif cfg.dataset.sigmoid_out:
                    outputs[outputs >= 0.5] = 1
                    outputs[outputs < 0.5]  = 0
                    preds       = outputs.squeeze(1)
                    preds_cpu   = preds.detach().cpu().numpy()
                    targets_cpu = targets.cpu().numpy().reshape(1, -1)[0]


                val_n_correct += (preds == targets).sum().item()
                val_n_total   += targets.shape[0]

                val_total_targets = np.concatenate((val_total_targets, targets_cpu))
                val_total_preds   = np.concatenate((val_total_preds, preds_cpu))

        val_loss       = np.mean(val_loss)
        val_losses[it] = val_loss
        conf_val = confusion_matrix(val_total_targets, val_total_preds)

        print(f'\n\nINFO: [val_model] valing label distribution: {np.unique(val_total_targets, return_counts=True)}')
        print(f'\nINFO: [val_model] valing Confusion Matrix: \n{conf_val}')

        ## val accuracy for each label
        val_acc = val_n_correct/val_n_total
        label_acc_val, label_prec_val = evaluator.print_acc_details('val', conf_val, val_acc)
        
        if hyperparameter_tune:
            os.makedirs("my_model", exist_ok=True)
            torch.save((model.state_dict(), optimizer.state_dict()), "my_model/checkpoint.pt")
            checkpoint = Checkpoint.from_directory("my_model")
            session.report({"loss": (val_loss / val_steps), "accuracy": val_n_correct / val_n_total}, checkpoint=checkpoint)
            pass

        #############################################################
        #                         TESTING                           #
        #############################################################
        testor.test_model(cfg, model, dataloader.test_dataloader, device, criterion, it=it, test_with='in_progress', save=True)


        ## Save the last model and its traced module
        if not os.path.exists(f'{cfg.dataset.system_dir}/models/'):
            os.makedirs(f'{cfg.dataset.system_dir}/models/')
        savename   = f'{cfg.dataset.system_dir}/models/{cfg.model.model_name}/{cfg.dataset.label_mode}_epoch_{it}'
        
        last_model = copy.deepcopy(model)
        last_model.eval()
        last_model = last_model.to(torch.device('cpu'))
        if it%cfg.dataset.model_save_freq==0:
            if multi_gpu:
                torch.save(last_model.module, savename+'.pt')
            else:
                torch.save(last_model, savename+'.pt')
        
        ## Best model update
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_val_epoch = it
            print(f'\nINFO: [train_model] Model saved: Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f},  Best Val Epoch: {best_val_epoch}\n')
            
        # ## Update training progress log and save as txt file
        context = ''
        
        if os.path.exists(f'{cfg.dataset.system_dir}/outputs/test_saved_models/__sample__training_progress.txt'):
            with open(f'{cfg.dataset.system_dir}/outputs/test_saved_models/__sample__training_progress.txt', 'r') as txt:
                context = txt.readlines()
                context = ''.join(context)

        with open(f'{cfg.dataset.system_dir}/outputs/test_saved_models/__sample__training_progress.txt', 'w') as txt:
            training_progress = context + f'Current Epoch: {it}, Best Epoch: {best_val_epoch} / {epochs}, Val Loss: {best_val_loss}'
            if not pretrain_training_mode:
                training_progress += f'\nTrain Accuracy: {train_acc*100:.2f} % Train Accuracy by label: {label_acc_train}, Train Precision by label: {label_prec_train}'
                training_progress += f'\nTrain Confusion Matrix: \n{conf_train}'
                training_progress += f'\n\nVal Accuracy: {val_acc*100:.2f} % Val Accuracy by label: {label_acc_val}, Val Precision by label: {label_prec_val}'
                training_progress += f'\nVal Confusion Matrix: \n{conf_val}'

            training_progress += '\n################################################################################################################################################\n\n'
            txt.writelines(training_progress)
    
        
        ##################################################################### FINISH ###############################################################
        if pretrain_training_mode:
            print(f'\nINFO: [train_model] Epoch {it}/{epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Best Val Epoch: {best_val_epoch}\n\n')
        else:
            epoch_summary = np.array([it, train_loss, train_acc*100, val_loss, val_acc*100])
            if summarys is None: 
                summarys = epoch_summary
            else :
                summarys = np.vstack([summarys, epoch_summary])
                
            print(f'\nINFO: [train_model] Epoch {it}/{epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Train Accuracy: {train_acc*100:.2f} %, Validation Accuracy: {val_acc*100:.2f} %, Best Val Epoch: {best_val_epoch}\n\n')

        ## LR scheduler
        if cfg.dataset.enable_scheduler:
            scheduler.step()
            
        summarys_df = pd.DataFrame(summarys)
        summarys_df.to_csv(f'{cfg.dataset.system_dir}/outputs/summarys/{cfg.model.model_name}/[{it}_{epochs}epochs] {cfg.dataset.predict_type}_{cfg.dataset.label_mode}_{cfg.dataset.feature_mode}_summary.csv', index = False)
        ##############################################################################################################################################
 
    model.eval()
    return best_model, model, train_losses, val_losses

def train_model(cfg, dataloader, model, criterion, optimizer, scheduler, epochs):
    # torch.manual_seed(0)
    # torch.cuda.manual_seed(0)
    # # torch.cuda.manual_seed_all(self.seed)  # if use multi-GPU
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # np.random.seed(0)
    # random.seed(0)
    
    #### Set Variables 
    ## (1) Dataload & Devices
    dataloader.load_data()
    model, device, multi_gpu = setting.prepare_gpus(cfg, model)
    model_name               = cfg.model.model_name
    
    ## (2) Datatypes 
    data_type_str   = cfg.dataset.data_type
    float_data_type = torch.double
    int_data_type   = torch.long
    # print(f'INFO: [train_model] Setting data type as {data_type_str}.')

    ## (3) loss.
    train_losses     = np.zeros(epochs)
    val_losses       = np.zeros(epochs)
    best_val_loss    = np.inf
    best_val_epoch   = 0
    best_model       = None
    last_model       = None

    print(f'INFO: [train_model] [Model] Model name : {model_name} ')
    print(f'INFO: [train_model] [Parameter] epoch : {epochs}, lr : {cfg.model.lr}, input_feature_cnt : {cfg.model.enc_in}')
    print(f'INFO: [train_model] [Act-Fn : {cfg.model.activation}] \t [Loss-Fn : {cfg.dataset.loss_fn}].')
    # assert cfg.model.enc_in == dataloader.train_x.shape[2], '[train_model] "enc_in" parameter != input_features_shape'
    
    ## (4) output shape 
    output_key = 'smax_out' if cfg.dataset.smax_out else 'sigmoid_out'
    
    ## Save the config files
    save_output_dir = f'./outputs/test_saved_models/{model_name}'
    with open(f'{save_output_dir}/hydra_config.txt', 'w') as txt:
        txt.writelines(str(cfg))
    
    #############################################################
    #                         TRAINING                          #
    #############################################################
    
    pretrain_training_mode   = cfg.dataset.pretrain_training_mode
    summarys = None
    for it in tqdm(range(epochs)):

        model.train()
        
        train_loss          = []
        train_total_liks    = []
        train_total_targets = []
        train_total_preds   = []
        train_total_volats  = []
        
        if cfg.dataset.volat_grouping:
            train_vol_g1_total_preds, train_vol_g10_total_preds     = [], []
            train_vol_g1_total_targets, train_vol_g10_total_targets = [], []
 

        # Train accuracy initialization
        train_n_correct           = 0.
        train_n_total             = 0.
        iter_count                = 0.

        for batch_idx, dataset in enumerate(dataloader.train_dataloader):
            
            ### Setting inputs & targets
            inputs, targets = dataset['X'], dataset['y']
            
            if cfg.dataset.loss_fn in ['CrossEntropy', 'LogitNormLoss', 'FocalLoss']:
                inputs, targets = inputs.to(device,  dtype=float_data_type), targets.to(device, dtype=int_data_type)
            elif cfg.dataset.loss_fn in ["BCE", "BCEWithLogits"]:
                inputs, targets = inputs.to(device, dtype=float_data_type), targets.to(device, dtype=float_data_type)
            elif cfg.dataset.loss_fn in ["MSELoss", "RMSELoss"]:
                inputs, targets = inputs.to(device, dtype=float_data_type), targets.to(device, dtype=float_data_type)
                targets = torch.unsqueeze(targets, 1)

            optimizer.zero_grad()
            
            ### Prediction & Backward  
            if cfg.dataset.volat_grouping:
                volats_cpu          = inputs[:, -1, -1].cpu().numpy()
                train_total_volats  = np.concatenate((train_total_volats, volats_cpu))
                outputs             = model(inputs[:, :, :-1])
                # g1_idxs   , g10_idxs    = torch.where(inputs[:, -1, -1] == 1)[0], torch.where(inputs[:, -1, -1] == 10)[0]
                # g1_inputs , g10_inputs = inputs[g1_idxs, :, :4], inputs[g10_idxs, :, :4]
                # g1_targets, g10_targets = targets[g1_idxs], targets[g10_idxs]
                
                # g1_outputs  = model(g1_inputs, volat_group = 1) if (g1_inputs.shape[0] != 0) else None
                # g10_outputs = model(g10_inputs, volat_group = 10) if (g10_inputs.shape[0] != 0) else None
                # outputs, targets        = torch.concat((g1_outputs, g10_outputs)), torch.concat((g1_targets, g10_targets))
           
            else: 
                outputs = model(inputs)                         
            
            if output_key == 'smax_out':
                lik, preds           = torch.max(outputs, 1)
                liks_cpu, preds_cpu  = lik.detach().cpu().numpy(), preds.cpu().numpy()
                targets_cpu          = targets.cpu().numpy()
                
            elif output_key == 'sigmoid_out':
                liks_cpu = outputs.detach().cpu().numpy()
                preds_cpu   = np.squeeze(np.where(liks_cpu>= 0.5, 1, 0))
                targets_cpu = targets.cpu().numpy().reshape(1, -1)[0]

            loss            = criterion(outputs, targets)  
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

            
            ### Save Results Data ###
            train_n_correct     += (preds_cpu == targets_cpu).sum().item()
            train_n_total       += targets.shape[0]
            
            train_total_liks    = np.concatenate((train_total_liks, liks_cpu))
            train_total_targets = np.concatenate((train_total_targets, targets_cpu))
            train_total_preds   = np.concatenate((train_total_preds, preds_cpu))
            
            # print(f'class_1_len : {targets_cpu.sum()}, total_len : {targets_cpu.shape[0]}, loss : {loss.item()}')
            

            ### Print INFOS ###     
            # if iter_count % 10 == 0:
            #     print("INFO: [train_model] Batch : ", iter_count, "\t Train_acc : ", train_n_correct/train_n_total, f"\t Mean Train_loss : {np.mean(train_loss):.5f}")       
            iter_count += 1

        ### Epoch loss & eval ###
        train_loss       = np.mean(train_loss)
        train_losses[it] = train_loss
        
        # dataloader.suffle_train_dset()
        conf_train = confusion_matrix(train_total_targets, train_total_preds)
        print(f'\n\nINFO: [train_model] Training label distribution: {np.unique(train_total_targets, return_counts=True)}')
        print(f'\nINFO: [train_model] Training Confusion Matrix: \n{conf_train}')

        train_acc = train_n_correct/train_n_total
        label_acc_train, label_prec_train = evaluator.print_acc_details('Train', conf_train, train_acc)
        if it % 10 == 0:
            evaluator.print_perform_details('Train', md = model_name, it = it, epochs = epochs, targets = train_total_targets, preds = train_total_preds, probs = train_total_liks, volats = train_total_volats, acc = train_acc, key = output_key)

        #############################################################
        #                         Validation                        #
        #############################################################
        val_loss          = []
        val_total_liks    = []
        val_total_targets = []
        val_total_preds   = []
        val_total_volats  = []
       
        
        # val accuracy initialization
        val_n_correct     = 0.
        val_n_total       = 0.
        iter_count        = 0.

        with torch.no_grad():
            model.eval()
            
            for batch_idx, dataset in enumerate(dataloader.val_dataloader):
                inputs, targets = dataset['X'], dataset['y']
                
                if cfg.dataset.loss_fn in ['CrossEntropy', 'LogitNormLoss', 'FocalLoss']:
                    inputs, targets = inputs.to(device,  dtype=float_data_type), targets.to(device, dtype=int_data_type)
                elif cfg.dataset.loss_fn in ["BCE", "BCEWithLogits"]:
                    inputs, targets = inputs.to(device, dtype=float_data_type), targets.to(device, dtype=float_data_type)
                elif cfg.dataset.loss_fn in ["MSELoss", "RMSELoss"]:
                    inputs, targets = inputs.to(device, dtype=float_data_type), targets.to(device, dtype=float_data_type)
                    targets = torch.unsqueeze(targets, 1)
        
                if cfg.dataset.volat_grouping:
                    volats_cpu        = inputs[:, -1, -1].cpu().numpy()
                    val_total_volats  = np.concatenate((val_total_volats, volats_cpu))
                    outputs           = model(inputs[:, :, :-1])
                    # g1_idxs   , g10_idxs    = torch.where(inputs[:, -1, -1] == 1)[0], torch.where(inputs[:, -1, -1] == 10)[0]
                    # g1_inputs , g10_inputs  = inputs[g1_idxs, :, :4], inputs[g10_idxs, :, :4]
                    # g1_targets, g10_targets = targets[g1_idxs], targets[g10_idxs]
                    
                    # g1_outputs  = model(g1_inputs , volat_group = 1)  if (g1_inputs.shape[0] != 0) else g1_targets
                    # g10_outputs = model(g10_inputs, volat_group = 10) if (g10_inputs.shape[0] != 0) else g10_targets

                    # outputs, targets        = torch.concat((g1_outputs, g10_outputs)), torch.concat((g1_targets, g10_targets))
                
                else:   
                    outputs = model(inputs)                                
                                                                        
            
                if output_key == 'smax_out':
                    lik, preds      = torch.max(outputs, 1)
                    liks_cpu, preds_cpu  = lik.detach().cpu().numpy(), preds.cpu().numpy()
                    targets_cpu     = targets.cpu().numpy()
                    
                elif output_key == 'sigmoid_out':
                    liks_cpu = np.squeeze(outputs.detach().cpu().numpy())
                    preds_cpu   = np.squeeze(np.where(liks_cpu>= 0.5, 1, 0))
                    targets_cpu = targets.cpu().numpy().reshape(1, -1)[0]

                loss            = criterion(outputs, targets)
                val_loss.append(loss.item())

                val_n_correct += (preds_cpu == targets_cpu).sum().item()
                val_n_total   += targets.shape[0]

                val_total_liks    = np.concatenate((val_total_liks, liks_cpu))
                val_total_targets = np.concatenate((val_total_targets, targets_cpu))
                val_total_preds   = np.concatenate((val_total_preds, preds_cpu))
            

        val_loss       = np.mean(val_loss)
        val_losses[it] = val_loss
        conf_val = confusion_matrix(val_total_targets, val_total_preds)

        print(f'\n\nINFO: [val_model] valing label distribution: {np.unique(val_total_targets, return_counts=True)}')
        print(f'\nINFO: [val_model] valing Confusion Matrix: \n{conf_val}')

        ## val accuracy for each label
        val_acc = val_n_correct/val_n_total
        label_acc_val, label_prec_val = evaluator.print_acc_details('val', conf_val, val_acc)
        if it % 10 == 0:
            evaluator.print_perform_details('Val', md = model_name, it = it, epochs = epochs, targets = val_total_targets, preds = val_total_preds,  probs = val_total_liks, volats = val_total_volats, acc = val_acc, key = output_key)
        
        
        #############################################################
        #                         TESTING                           #
        # ###########################################################
        testor.test_model(cfg, model, dataloader.test_dataloader, device, criterion, epochs=epochs, it=it, test_with='in_progress', save=True)

        ## Save the last model and its traced module
        save_model_dir   = f'./models/{model_name}/'
        save_model_name  = f'{cfg.dataset.label_mode}_epoch_{it}'
        
        last_model = copy.deepcopy(model)
        last_model.eval()
        last_model = last_model.to(torch.device('cpu'))
        if it%cfg.dataset.model_save_freq==0:
            if multi_gpu:
                torch.save(last_model.module, save_model_dir + save_model_name+'.pt')
            else:
                torch.save(last_model, save_model_dir + save_model_name+'.pt')
        
        ## Best model update
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_val_epoch = it
            print(f'\nINFO: [train_model] Model saved: Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f},  Best Val Epoch: {best_val_epoch}\n')
            
        ## Update training progress log and save as txt file
        context = ''
        if os.path.exists(f'{save_output_dir}/training_progress.txt'):
            with open(f'{save_output_dir}/training_progress.txt', 'r') as txt:
                context = txt.readlines()
                context = ''.join(context)

        with open(f'{save_output_dir}/training_progress.txt', 'w') as txt:
            training_progress = context + f'Current Epoch: {it}, Best Epoch: {best_val_epoch} / {epochs}, Val Loss: {best_val_loss}'
            if not pretrain_training_mode:
                training_progress += f'\nTrain Accuracy: {train_acc*100:.2f} % Train Accuracy by label: {label_acc_train}, Train Precision by label: {label_prec_train}'
                training_progress += f'\nTrain Confusion Matrix: \n{conf_train}'
                training_progress += f'\n\nVal Accuracy: {val_acc*100:.2f} % Val Accuracy by label: {label_acc_val}, Val Precision by label: {label_prec_val}'
                training_progress += f'\nVal Confusion Matrix: \n{conf_val}'

            training_progress += '\n################################################################################################################################################\n\n'
            txt.writelines(training_progress)
    
        
        ##################################################################### FINISH ###############################################################
        if pretrain_training_mode:
            print(f'\nINFO: [train_model] Epoch {it}/{epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Best Val Epoch: {best_val_epoch}\n\n')
        else:
            epoch_summary = np.array([it, train_loss, train_acc*100, val_loss, val_acc*100])
            if summarys is None: 
                summarys = epoch_summary
            else :
                summarys = np.vstack([summarys, epoch_summary])
                
            print(f'\nINFO: [train_model] Epoch {it}/{epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Train Accuracy: {train_acc*100:.2f} %, Validation Accuracy: {val_acc*100:.2f} %, Best Val Epoch: {best_val_epoch}\n\n')

        ## LR scheduler
        if cfg.dataset.enable_scheduler:
            scheduler.step()
        
        
        summarys_df = pd.DataFrame(summarys)
        summarys_df.to_csv(f'./outputs/summarys/{cfg.model.model_name}/[{it}_{epochs}epochs] {cfg.dataset.predict_type}_{cfg.dataset.label_mode}_{cfg.dataset.feature_mode}_summary.csv', index = False)
        ##############################################################################################################################################
 
    model.eval()
    # return best_model, model, train_losses, val_losses

def train_FX(cfg, tuning_mode = False):
    
    ## (1) Setting Dataloader 
    dataloader = setting.get_dataloader(cfg)

    ## (2) Setting Model Class
    model, pretrained_model = setting.get_model(cfg, tuning_mode)

    ## (3) Setting Loss Function
    criterion = setting.get_loss_fn(cfg)
    
    ## (4) Setting Other parameters
    ## (5) Start Training
    epochs    = cfg.model.n_epochs

    if tuning_mode:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.model.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10) #step_size=5, gamma=0.9)
        best_model, last_model, train_losses, val_losses = train_for_tune(cfg, dataloader, model, criterion, optimizer, scheduler, epochs)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.model.lr, weight_decay=1e-5) 
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10) #step_size=5, gamma=0.9)
        best_model, last_model, train_losses, val_losses = train_model(cfg, dataloader, model, criterion, optimizer, scheduler, epochs)
        
