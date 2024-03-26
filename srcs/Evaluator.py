
import os 
import numpy as np

from datetime import datetime, timedelta

from sklearn.metrics import confusion_matrix

def print_acc_details(mode, conf, acc):
    label_acc = {}
    label_prec = {}
    for i in range(len(conf)):
        row = conf[i]
        col = conf.T[i]
        sum_row = np.sum(row)
        sum_col = np.sum(col)
        tmp_acc = np.round(conf[i][i] / sum_row * 100, 3)
        tmp_prec = np.nan_to_num(np.round(conf[i][i] / sum_col * 100, 3), 0.0)
        label_acc[i] = tmp_acc
        label_prec[i] = tmp_prec
        
    print(f'\nINFO: [print_acc_details] {mode} Accuracy: {acc*100:.3f}')
    print(f'INFO: [print_acc_details] {mode} Accuracy (Bal): {sum(label_acc.values())/2:.3f}')
    print(f'INFO: [print_acc_details] {mode} Accuracy by label: {label_acc}')
    print(f'INFO: [print_acc_details] {mode} Precision by label: {label_prec}')

    return label_acc, label_prec

def print_perform_details(mode, md, it, epochs, targets, preds, probs, acc, key = 'smax_out', volats = None):
    model_name = md
    if volats is not None:
        
        g1_idxs, g2_idxs, g3_idxs, g4_idxs, g5_idxs                 = np.where(volats == 1)[0], np.where(volats == 2)[0], np.where(volats == 3)[0], np.where(volats == 4)[0], np.where(volats == 5)[0] 
        g6_idxs, g7_idxs, g8_idxs, g9_idxs, g10_idxs                = np.where(volats == 6)[0], np.where(volats == 7)[0], np.where(volats == 8)[0], np.where(volats == 9)[0], np.where(volats == 10)[0] 
        
        g1_targets, g2_targets, g3_targets, g4_targets, g5_targets  = targets[g1_idxs], targets[g2_idxs], targets[g3_idxs], targets[g4_idxs], targets[g5_idxs]
        g6_targets, g7_targets, g8_targets, g9_targets, g10_targets = targets[g6_idxs], targets[g7_idxs], targets[g8_idxs], targets[g9_idxs], targets[g10_idxs]
        
        g1_preds, g2_preds, g3_preds, g4_preds, g5_preds            = preds[g1_idxs], preds[g2_idxs], preds[g3_idxs], preds[g4_idxs], preds[g5_idxs]
        g6_preds, g7_preds, g8_preds, g9_preds, g10_preds           = preds[g6_idxs], preds[g7_idxs], preds[g8_idxs], preds[g9_idxs], preds[g10_idxs]
        
        
        g1_conf, g2_conf, g3_conf, g4_conf, g5_conf   =  confusion_matrix(g1_targets, g1_preds), confusion_matrix(g2_targets, g2_preds), confusion_matrix(g3_targets, g3_preds), confusion_matrix(g4_targets, g4_preds), confusion_matrix(g5_targets, g5_preds)
        g6_conf, g7_conf, g8_conf, g9_conf, g10_conf  =  confusion_matrix(g6_targets, g6_preds), confusion_matrix(g7_targets, g7_preds), confusion_matrix(g8_targets, g8_preds), confusion_matrix(g9_targets, g9_preds), confusion_matrix(g10_targets, g10_preds)
        
        
        g1_acc   = round((((g1_preds == g1_targets).sum())  / g1_targets.shape[0]) * 100, 2)
        g2_acc   = round((((g2_preds == g2_targets).sum())  / g2_targets.shape[0]) * 100, 2)
        g3_acc   = round((((g3_preds == g3_targets).sum())  / g3_targets.shape[0]) * 100, 2)
        g4_acc   = round((((g4_preds == g4_targets).sum())  / g4_targets.shape[0]) * 100, 2)
        g5_acc   = round((((g5_preds == g5_targets).sum())  / g5_targets.shape[0]) * 100, 2)
        g6_acc   = round((((g6_preds == g6_targets).sum())  / g6_targets.shape[0]) * 100, 2)
        g7_acc   = round((((g7_preds == g7_targets).sum())  / g7_targets.shape[0]) * 100, 2)
        g8_acc   = round((((g8_preds == g8_targets).sum())  / g8_targets.shape[0]) * 100, 2)
        g9_acc   = round((((g9_preds == g9_targets).sum())  / g9_targets.shape[0]) * 100, 2)
        g10_acc  = round((((g10_preds == g10_targets).sum()) / g10_targets.shape[0]) * 100, 2)
        
        
        print(f"\n[Confusion Matrix by Group]")
        print("\nVolatility GROUP 01 :")
        print(f'INFO: Accuracy : {g1_acc}%\nGroup_01 Confusion Matrix: \n{g1_conf}')
        
        print("\nVolatility GROUP 02")
        print(f'INFO: Accuracy : {g2_acc}%\nGroup_02 Confusion Matrix: \n{g2_conf}')
        
        print("\nVolatility GROUP 03")
        print(f'INFO: Accuracy : {g3_acc}%\nGroup_03 Confusion Matrix: \n{g3_conf}')
        
        print("\nVolatility GROUP 04 :")
        print(f'INFO: Accuracy : {g4_acc}%\nGroup_04 Confusion Matrix: \n{g4_conf}')
        
        print("\nVolatility GROUP 05")
        print(f'INFO: Accuracy : {g5_acc}%\nGroup_05 Confusion Matrix: \n{g5_conf}')
        
        print("\nVolatility GROUP 06")
        print(f'INFO: Accuracy : {g6_acc}%\nGroup_06 Confusion Matrix: \n{g6_conf}')
        
        print("\nVolatility GROUP 07 :")
        print(f'INFO: Accuracy : {g7_acc}%\nGroup_07 Confusion Matrix: \n{g7_conf}')
        
        print("\nVolatility GROUP 08")
        print(f'INFO: Accuracy : {g8_acc}%\nGroup_08 Confusion Matrix: \n{g8_conf}')
        
        print("\nVolatility GROUP 09")
        print(f'INFO: Accuracy : {g9_acc}%\nGroup_09 Confusion Matrix: \n{g9_conf}\n')
        
        print("\nVolatility GROUP 10")
        print(f'INFO: Accuracy : {g10_acc}%\nGroup_10 Confusion Matrix: \n{g10_conf}\n')
        
        
        filename = f'[{mode.upper()}]grouping_by_volatility_of_{mode}set'
        context  = ''
        if os.path.exists(f'./outputs/summarys/{model_name}/{filename}.txt'):
            with open(f'./outputs/summarys/{model_name}/{filename}.txt', 'r') as txt:
                context = txt.readlines()
                context = ''.join(context)

        with open(f'./outputs/summarys/{model_name}/{filename}.txt', 'w') as txt:
            training_progress = context + f'Current Epoch: {it} / {epochs}'
            training_progress += f"\nTotal Accuracy : {round(acc, 2)}%"
            training_progress += f"\n\n[Confusion Matrix by Group]"
            training_progress += f"\nVolatility GROUP 01"
            training_progress += f'INFO: Accuracy : {g1_acc}%\nGroup_01 Confusion Matrix: \n{g1_conf}'
            training_progress += f"\nVolatility GROUP 02"
            training_progress += f'INFO: Accuracy : {g2_acc}%\nGroup_02 Confusion Matrix: \n{g2_conf}'
            training_progress += f"\nVolatility GROUP 03"
            training_progress += f'INFO: Accuracy : {g3_acc}%\nGroup_03 Confusion Matrix: \n{g3_conf}'
            training_progress += f"\nVolatility GROUP 04"
            training_progress += f'INFO: Accuracy : {g4_acc}%\nGroup_04 Confusion Matrix: \n{g4_conf}'
            training_progress += f"\nVolatility GROUP 05"
            training_progress += f'INFO: Accuracy : {g5_acc}%\nGroup_05 Confusion Matrix: \n{g5_conf}'
            training_progress += f"\nVolatility GROUP 06"
            training_progress += f'INFO: Accuracy : {g6_acc}%\nGroup_06 Confusion Matrix: \n{g6_conf}'
            training_progress += f"\nVolatility GROUP 07"
            training_progress += f'INFO: Accuracy : {g7_acc}%\nGroup_07 Confusion Matrix: \n{g7_conf}'
            training_progress += f"\nVolatility GROUP 08"
            training_progress += f'INFO: Accuracy : {g8_acc}%\nGroup_08 Confusion Matrix: \n{g8_conf}'
            training_progress += f"\nVolatility GROUP 09"
            training_progress += f'INFO: Accuracy : {g9_acc}%\nGroup_09 Confusion Matrix: \n{g9_conf}'
            training_progress += f"\nVolatility GROUP 10"
            training_progress += f'INFO: Accuracy : {g10_acc}%\nGroup_010 Confusion Matrix: \n{g10_conf}'
            training_progress += '\n\n################################################################################################################################################\n\n'
            txt.writelines(training_progress)
 
 
        # with open(f'./outputs/summarys/{filename}.txt', 'w') as txt:
        #     training_progress = context + f'Current Epoch: {it} / {epochs}'
        #     training_progress += f"\nTotal Accuracy : {round(acc, 2)}%"
        #     training_progress += f"\n\n[Confusion Matrix by Group]"
        #     training_progress += f"\nVolatility GROUP 01"
        #     training_progress += f'INFO: Accuracy : {g1_acc}%\nGroup_01 Confusion Matrix: \n{g1_conf}'
        #     training_progress += f"\nVolatility GROUP 10"
        #     training_progress += f'INFO: Accuracy : {g10_acc}%\nGroup_010 Confusion Matrix: \n{g10_conf}'
        #     training_progress += '\n\n################################################################################################################################################\n\n'
        #     txt.writelines(training_progress)
    
    if key == 'sigmoid_out':
        total_sample_length = len(probs)
        probs_55_cnt, probs_45_cnt, others  = len(probs[probs >= 0.55]), len(probs[probs <= 0.45]), len(probs[(0.45 < probs) & (probs < 0.55)])
        probs_55_rat, probs_45_rat, others_rat = round((probs_55_cnt / total_sample_length) * 100, 2) , round((probs_45_cnt / total_sample_length) * 100, 2), round((others / total_sample_length) * 100, 2) 

        print(f"Summarys ")
        print(f"INFO: [probs >= 0.55] {probs_55_cnt}({probs_55_rat}%)\t[probs <= 0.45] {probs_45_cnt}({probs_45_rat}%)\t[0.45 < probs < 0.55] {others}({others_rat}%)")
        print(f"Total Accuracy : {round(acc*100, 2)}%")
        
            
        # group 01 : [0.55, 0.60) & (0.40, 0.45]
        group_01_idx_u = np.where((0.55 <= probs) & (probs < 0.60))[0] 
        group_01_idx_b = np.where((0.40 < probs) & (probs <= 0.45))[0]
        group_01_idxs  = np.concatenate([group_01_idx_u, group_01_idx_b])
        group_01_targets, group_01_preds = targets[group_01_idxs], preds[group_01_idxs] 
        
        # group 02 : [0.60, 0.65) & (0.35, 0.40]
        group_02_idx_u = np.where((0.60 <= probs) & (probs < 0.65))[0] 
        group_02_idx_b = np.where((0.35 < probs) & (probs <= 0.40))[0] 
        group_02_idxs  = np.concatenate([group_02_idx_u, group_02_idx_b])
        group_02_targets, group_02_preds = targets[group_02_idxs], preds[group_02_idxs]  
        
        # group 03 : [0.65, 0.70) & (0.30, 0.35]
        group_03_idx_u = np.where((0.65 <= probs) & (probs < 0.70))[0] 
        group_03_idx_b = np.where((0.30 < probs) & (probs <= 0.35))[0]
        group_03_idxs  = np.concatenate([group_03_idx_u, group_03_idx_b])
        group_03_targets, group_03_preds = targets[group_03_idxs], preds[group_03_idxs] 
        
        
        group_01_conf  = confusion_matrix(group_01_targets, group_01_preds)
        group_02_conf  = confusion_matrix(group_02_targets, group_02_preds)
        group_03_conf  = confusion_matrix(group_03_targets, group_03_preds)
        
        
        group_01_acc   = round((((group_01_preds == group_01_targets).sum()) / group_01_targets.shape[0]) * 100, 2)
        group_02_acc   = round((((group_02_preds == group_02_targets).sum()) / group_02_targets.shape[0]) * 100, 2)
        group_03_acc   = round((((group_03_preds == group_03_targets).sum()) / group_03_targets.shape[0]) * 100, 2)
        
        
        print(f"\n[Confusion Matrix by Group]")
        print("\nGROUP 01 : [0.55, 0.60) & (0.40, 0.45]")
        print(f'INFO: Accuracy : {group_01_acc}%\nGroup_01 Confusion Matrix: \n{group_01_conf}')
        
        print("\nGROUP 02 : [0.60, 0.65) & (0.35, 0.40]")
        print(f'INFO: Accuracy : {group_02_acc}%\nGroup_02 Confusion Matrix: \n{group_02_conf}')
        
        print("\nGROUP 03 : [0.65, 0.70) & (0.30, 0.35]")
        print(f'INFO: Accuracy : {group_03_acc}%\nGroup_03 Confusion Matrix: \n{group_03_conf}\n')

        
        filename = f'[{mode.upper()}]grouping_by_probs_of_{mode}set'
        context = ''
        if os.path.exists(f'./outputs/summarys/{model_name}/{filename}.txt'):
            with open(f'./outputs/summarys/{model_name}/{filename}.txt', 'r') as txt:
                context = txt.readlines()
                context = ''.join(context)

        with open(f'./outputs/summarys/{model_name}/{filename}.txt', 'w') as txt:
            training_progress = context + f'Current Epoch: {it} / {epochs}'
            training_progress += f"\n\nINFO: [probs >= 0.55] {probs_55_cnt}({probs_55_rat}%)\t[probs <= 0.45] {probs_45_cnt}({probs_45_rat}%)\t[0.45 < probs < 0.55] {others}({others_rat}%)"
            training_progress += f"\nTotal Accuracy : {round(acc, 2)}%"
            training_progress += f"\n\n[Confusion Matrix by Group]"
            training_progress += f"\nGROUP 01 : [0.55, 0.60) & (0.40, 0.45]"
            training_progress += f'INFO: Accuracy : {group_01_acc}%\nGROUP_01 Confusion Matrix: \n{group_01_conf}'
            training_progress += f"\n\nGROUP 02 : [0.60, 0.65) & (0.35, 0.40]"
            training_progress += f'INFO: Accuracy : {group_02_acc}%\nGROUP_02 Confusion Matrix: \n{group_02_conf}'
            training_progress += f"\n\nGROUP 03 : [0.65, 0.70) & (0.30, 0.35]"
            training_progress += f'INFO: Accuracy : {group_03_acc}%\nGROUP_03 Confusion Matrix: \n{group_03_conf}'
            training_progress += '\n\n################################################################################################################################################\n\n'
            txt.writelines(training_progress)
    
    elif key == 'smax_out':
        total_sample_length      = len(probs)
        probs_55_cnt, others     = len(probs[probs >= 0.55]), len(probs[(probs < 0.55)])
        probs_55_rat, others_rat = round((probs_55_cnt / total_sample_length) * 100, 2) , round((others / total_sample_length) * 100, 2) 

        print(f"\nSummarys ")
        print(f"INFO: [probs >= 0.55] {probs_55_cnt}({probs_55_rat}%)\t[probs < 0.55] {others}({others_rat}%)")
        print(f"Total Accuracy : {round(acc*100, 2)}%")
        
            
        # group 01 : [0.55, 0.60)
        group_01_idxs = np.where((0.55 <= probs) & (probs < 0.60))[0] 
        group_01_targets, group_01_preds = targets[group_01_idxs], preds[group_01_idxs] 
        
        # group 02 : [0.60, 0.65) 
        group_02_idxs = np.where((0.60 <= probs) & (probs < 0.65))[0] 
        group_02_targets, group_02_preds = targets[group_02_idxs], preds[group_02_idxs]  
        
        # group 03 : [0.65, 0.70) 
        group_03_idxs = np.where((0.65 <= probs) & (probs < 0.70))[0] 
        group_03_targets, group_03_preds = targets[group_03_idxs], preds[group_03_idxs] 
        
        # group 04 : [0.70, ) 
        group_04_idxs = np.where((0.70 <= probs))[0] 
        group_04_targets, group_04_preds = targets[group_04_idxs], preds[group_04_idxs] 
        
        group_01_conf  = confusion_matrix(group_01_targets, group_01_preds)
        group_02_conf  = confusion_matrix(group_02_targets, group_02_preds)
        group_03_conf  = confusion_matrix(group_03_targets, group_03_preds)
        group_04_conf  = confusion_matrix(group_04_targets, group_04_preds)
        
        group_01_acc   = round((((group_01_preds == group_01_targets).sum()) / group_01_targets.shape[0]) * 100, 2)
        group_02_acc   = round((((group_02_preds == group_02_targets).sum()) / group_02_targets.shape[0]) * 100, 2)
        group_03_acc   = round((((group_03_preds == group_03_targets).sum()) / group_03_targets.shape[0]) * 100, 2)
        group_04_acc   = round((((group_04_preds == group_04_targets).sum()) / group_04_targets.shape[0]) * 100, 2)
        
        
        print(f"\n[Confusion Matrix by Group]")
        print("\nGROUP 01 : [0.55, 0.60)]")
        print(f'INFO: Accuracy : {group_01_acc}%\nGroup_01 Confusion Matrix: \n{group_01_conf}')
        
        print("\nGROUP 02 : [0.60, 0.65)")
        print(f'INFO: Accuracy : {group_02_acc}%\nGroup_02 Confusion Matrix: \n{group_02_conf}')
        
        print("\nGROUP 03 : [0.65, 0.70)")
        print(f'INFO: Accuracy : {group_03_acc}%\nGroup_03 Confusion Matrix: \n{group_03_conf}\n')
        
        print("\nGROUP 04 : [0.70,     )")
        print(f'INFO: Accuracy : {group_04_acc}%\nGroup_04 Confusion Matrix: \n{group_04_conf}\n')
        
        
        filename = f'[{mode.upper()}]grouping_by_probs_of_{mode}set'
        
        
        context = ''
        if os.path.exists(f'./outputs/summarys/{model_name}/{filename}.txt'):
            with open(f'./outputs/summarys/{model_name}/{filename}.txt', 'r') as txt:
                context = txt.readlines()
                context = ''.join(context)

        with open(f'./outputs/summarys/{model_name}/{filename}.txt', 'w') as txt:
            training_progress = context + f'Current Epoch: {it} / {epochs}'
            training_progress += f"\n\nINFO: [probs >= 0.55] {probs_55_cnt}({probs_55_rat}%)\t[probs < 0.55] {others}({others_rat}%)"
            training_progress += f"\nTotal Accuracy : {round(acc, 2)}%"
            training_progress += f"\n\n[Confusion Matrix by Group]"
            training_progress += f"\nGROUP 01 : [0.55, 0.60)"
            training_progress += f'INFO: Accuracy : {group_01_acc}%\nGROUP_01 Confusion Matrix: \n{group_01_conf}'
            training_progress += f"\n\nGROUP 02 : [0.60, 0.65)"
            training_progress += f'INFO: Accuracy : {group_02_acc}%\nGROUP_02 Confusion Matrix: \n{group_02_conf}'
            training_progress += f"\n\nGROUP 03 : [0.65, 0.70)"
            training_progress += f'INFO: Accuracy : {group_03_acc}%\nGROUP_03 Confusion Matrix: \n{group_03_conf}'
            training_progress += f"\n\nGROUP 04 : [0.70,     )"
            training_progress += f'INFO: Accuracy : {group_04_acc}%\nGROUP_04 Confusion Matrix: \n{group_04_conf}'
            training_progress += '\n\n################################################################################################################################################\n\n'
            txt.writelines(training_progress)
            
