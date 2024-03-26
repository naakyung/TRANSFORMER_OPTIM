import torch
from tqdm import tqdm
from Transformers import * 

import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 

import warnings 
warnings.filterwarnings('ignore')


def plot_inventory_profit(dat1:pd.Series, dat1_label:str, dat2:pd.Series, dat2_label:str, title:str):

    #model_cum_profit

    diff = dat2.squeeze() - dat1.squeeze()

    fig = plt.figure(figsize=(20, 6))
    ax = fig.add_subplot()

    ax.plot(dat1, color='black', label=dat1_label, linewidth=1.0)
    ax.plot(dat2, color='red',  label=dat2_label, linewidth=1.0)
    ax.stackplot(diff.index, diff, color="blue", alpha=0.2, linewidth=0)

    #plt.xlabel("Monthly Maturity Date")
    plt.ylabel("Inventory Profit")
    ax.grid(True)
    plt.title(title, fontdict = {'fontsize' : 15})
    plt.style.use("seaborn-whitegrid")
    plt.legend()
    plt.show()
    plt.savefig(f'./Performance/{title}.png')

def print_performance(prof_model, key):

    cum_ret = prof_model['model_cum_profit'].iloc[-1]
    print(f'\n[{key}] RETURN : {cum_ret}')
    print(f'[{key}] Profit average : ', round(prof_model[prof_model['ret'] > 0]['price_diff'].mean(), 4))
    print(f'[{key}] Loss average   : ', round(prof_model[prof_model['ret'] < 0]['price_diff'].mean(), 4))

    ##### 변동성 단위 모델 평가 ##### 
    volat_df = prof_model[prof_model['volat'].isna() == False]
    prof_sample = volat_df[volat_df['ret'] > 0] 
    loss_sample = volat_df[volat_df['ret'] < 0]
    print(f'[{key}] Profit Volat-weight average : ', round(((prof_sample['volat'] / prof_sample['volat'].sum()) * prof_sample['price_diff']).sum(), 4))
    print(f'[{key}] Loss   Volat-weight average : ', round(((loss_sample['volat'] / loss_sample['volat'].sum()) * loss_sample['price_diff']).sum(), 4))
    
    print(f'[{key}] Profit Probability : ', round(len(prof_sample) / (len(prof_sample) + len(loss_sample)), 4) * 100, "%")
    print(f'[{key}] Loss Probability : ', round(len(loss_sample) / (len(prof_sample) + len(loss_sample)), 4) * 100, "%")


    ##### 변동성 그룹 단위 평가 #####
    volat_25per, volat_50per, volat_75per = volat_df['volat'].describe()['25%'], volat_df['volat'].describe()['50%'], volat_df['volat'].describe()['75%']
    vol_g1 = volat_df[volat_df['volat'] < volat_25per]
    vol_g2 = volat_df[(volat_25per <= volat_df['volat']) & (volat_df['volat'] < volat_50per)]
    vol_g3 = volat_df[(volat_50per <= volat_df['volat']) & (volat_df['volat'] < volat_75per)]
    vol_g4 = volat_df[volat_75per < volat_df['volat']]
    
    vol_g1_prof_sample, vol_g1_loss_sample = vol_g1[vol_g1['ret'] > 0], vol_g1[vol_g1['ret'] < 0]
    vol_g2_prof_sample, vol_g2_loss_sample = vol_g2[vol_g2['ret'] > 0], vol_g2[vol_g2['ret'] < 0]
    vol_g3_prof_sample, vol_g3_loss_sample = vol_g3[vol_g3['ret'] > 0], vol_g3[vol_g3['ret'] < 0]
    vol_g4_prof_sample, vol_g4_loss_sample = vol_g4[vol_g4['ret'] > 0], vol_g4[vol_g4['ret'] < 0]

    print(f'\n[{key}-G1] Profit Avg : ', round(vol_g1_prof_sample['price_diff'].mean(), 4),' Loss Avg : ', round(vol_g1_loss_sample['price_diff'].mean(), 4))
    print(f'[{key}-G2] Profit Avg : ', round(vol_g2_prof_sample['price_diff'].mean(), 4),' Loss Avg : ', round(vol_g2_loss_sample['price_diff'].mean(), 4))
    print(f'[{key}-G3] Profit Avg : ', round(vol_g3_prof_sample['price_diff'].mean(), 4),' Loss Avg : ', round(vol_g3_loss_sample['price_diff'].mean(), 4))
    print(f'[{key}-G4] Profit Avg : ', round(vol_g4_prof_sample['price_diff'].mean(), 4),' Loss Avg : ', round(vol_g4_loss_sample['price_diff'].mean(), 4))

    cnt_g1_prof, cnt_g1_loss = len(vol_g1[vol_g1['ret'] > 0]), len(vol_g1[vol_g1['ret'] < 0])
    cnt_g2_prof, cnt_g2_loss = len(vol_g2[vol_g2['ret'] > 0]), len(vol_g2[vol_g2['ret'] < 0])
    cnt_g3_prof, cnt_g3_loss = len(vol_g3[vol_g3['ret'] > 0]), len(vol_g3[vol_g3['ret'] < 0])
    cnt_g4_prof, cnt_g4_loss = len(vol_g4[vol_g4['ret'] > 0]), len(vol_g4[vol_g4['ret'] < 0])

    print(f'\n[{key}-G1] Profit Prob : {cnt_g1_prof/(cnt_g1_loss + cnt_g1_prof):.4f}', f', Loss Prob : {cnt_g1_loss/(cnt_g1_loss + cnt_g1_prof):.4f}')
    print(f'[{key}-G2] Profit Prob : {cnt_g2_prof/(cnt_g2_loss + cnt_g2_prof):.4f}', f', Loss Prob : {cnt_g2_loss/(cnt_g2_loss + cnt_g2_prof):.4f}')
    print(f'[{key}-G3] Profit Prob : {cnt_g3_prof/(cnt_g3_loss + cnt_g3_prof):.4f}', f', Loss Prob : {cnt_g3_loss/(cnt_g3_loss + cnt_g3_prof):.4f}')
    print(f'[{key}-G4] Profit Prob : {cnt_g4_prof/(cnt_g4_loss + cnt_g4_prof):.4f}', f', Loss Prob : {cnt_g4_loss/(cnt_g4_loss + cnt_g4_prof):.4f}')

class EnsembleResult:
    def __init__(self, file_dict):
        self.price_data_dir = r"/home/jovyan/shared/wealth-solution/FX_modeling/Transformer_optim/Datas/raw_datas/{}"
        self.dl_data_dir    = r"/home/jovyan/shared/wealth-solution/FX_modeling/Transformer_optim/Datas/signals/deeplearning_model_res/{}"
        self.quant_dat_dir  = fr"/home/jovyan/shared/wealth-solution/FX_modeling/Transformer_optim/Datas/signals/quant_model_res/{file_dict['quant_model']}"
        self.prev_dat_dir   = fr"/home/jovyan/shared/wealth-solution/FX_modeling/Transformer_optim/Datas/signals/version1_model_res/{file_dict['prev_model']}"

        self.deeplearning_data = pd.read_csv(self.dl_data_dir.format(file_dict["deeplearning_data"]))
        
        self.data_120m         = pd.read_csv(self.dl_data_dir.format(file_dict["120m_data"]))
        self.data_30m          = pd.read_csv(self.dl_data_dir.format(file_dict["30m_data"]))

        self._dl_data_arrange(thr=0.05)
        self.quant_data = pd.read_csv(self.quant_dat_dir, index_col=0)
        self.prev_data = pd.read_csv(self.prev_dat_dir, index_col=0)
        self.prev_data.columns =[pd.to_datetime(i).strftime("%Y%m%d") for i in self.prev_data.columns]

        self.close_p = pd.read_csv(self.price_data_dir.format(file_dict["close_price"]), index_col=0)
        self.close_p.columns = [pd.to_datetime(i).strftime("%Y%m%d") for i in self.close_p.columns]
        self.ret_1min        = self.close_p / self.close_p.shift(1) - 1

        self.quant_data.columns = [pd.to_datetime(i).strftime("%Y%m%d") for i in self.quant_data.columns]

        self.test_st_date = file_dict["test_st_date"]

        self.test_dates = [i for i in self.close_p.columns if int(i) >= int(self.test_st_date)]
        self.train_dates = [i for i in self.close_p.columns if int(i) < int(self.test_st_date)]

        self.st_time = 1200
        self.en_time = 1300


    def _dl_data_arrange(self, thr=0.05):
        self.deeplearning_data = self.deeplearning_data[self.deeplearning_data["time"].isna() == False]
        self.deeplearning_data = self.deeplearning_data[self.deeplearning_data["liks"] >= (0.5 + thr)].copy()

        self.data_120m = self.data_120m[self.data_120m["time"].isna() == False]
        self.data_120m = self.data_120m[self.data_120m["liks"] >= (0.5 + thr)].copy()

        self.data_30m  = self.data_30m[self.data_30m["time"].isna() == False]
        self.data_30m  = self.data_30m[self.data_30m["liks"] >= (0.5 + thr)].copy()

    def cal_profit_quant(self, freq=30, mode="buy", reversion_overlay=False, dl_overlay=False):

        obj_tm_sets = [i for i in self.close_p.index if (int(i) >= 1000) & (int(i) <= 1500)]
        past_tm_sets = [i for i in self.close_p.index if (int(i) >= 930) & (int(i) <= 1430)]

        profit_collect = []
        for i_d in tqdm(self.test_dates):
            for i_tm, i_ptm in zip(obj_tm_sets, past_tm_sets):

                now_price = self.close_p.loc[:i_tm, i_d].iloc[-2].copy()
                get_quant_sig = self.quant_data.loc[:i_tm, i_d].iloc[-2].copy()
                
                future_price = self.close_p.loc[i_tm:, i_d].iloc[:freq]
                past_price   = self.ret_1min.loc[i_ptm:i_tm, i_d].iloc[:-1]
                volat        = past_price.std() if (past_price.isna() == False).sum() == freq else None
                model = None
               
                state = "instant"
                ret = 0
                diff = 0

                if mode == "sell":
                    ret *= -1
                    diff *= -1

                # Basic dl signal
                get_dl_sig = self.deeplearning_data[(self.deeplearning_data["date"] == int(i_d)) & (self.deeplearning_data["time"] == i_tm)]

                # 시간대별 dl signal 

                # 특정 시간대 확률이 더 높은 모델의 signal 사용
                # if (i_tm >= self.st_time) and (i_tm < self.en_time):
                #     get_120_sig = self.data_120m[(self.data_120m["date"] == int(i_d)) & (self.data_120m["time"] == i_tm)]
                #     get_30_sig  = self.data_30m[(self.data_30m["date"] == int(i_d)) & (self.data_30m["time"] == i_tm)]
                #     if (get_120_sig.size == 0 and get_30_sig.size == 0) or (get_120_sig.size != 0 and get_30_sig.size == 0):
                #         get_dl_sig = get_120_sig
                #     elif get_120_sig.size == 0 and get_30_sig.size != 0:
                #         get_dl_sig = get_30_sig
                #     else:
                #         if get_120_sig['liks'].values[0] < get_30_sig['liks'].values[0]:
                #             get_dl_sig = get_30_sig
                #         else:
                #             get_dl_sig = get_120_sig
                # else:
                #     get_dl_sig = self.data_30m[(self.data_30m["date"] == int(i_d)) & (self.data_30m["time"] == i_tm)]

                if reversion_overlay == True:
                    if get_quant_sig != 0:
                        model = "RV"
                        if (mode == "buy") & (get_quant_sig == -1):
                            state = "maturity"
                            ret = -1 * (future_price.iloc[0] / now_price - 1)
                            diff = -1 * (future_price.iloc[0] - now_price)

                        elif (mode == "sell") & (get_quant_sig == 1):
                            state = "maturity"
                            ret = 1 * (future_price.iloc[0] / now_price - 1)
                            diff = 1 * (future_price.iloc[0] - now_price)


                if dl_overlay == True:

                    if get_dl_sig.empty == False:
                    
                        model = "DL"
                        if (mode=="buy") & (get_dl_sig["DL_sig"].item() == 0):
                            state = "maturity"
                            ret = -1 * (future_price.iloc[4] / now_price - 1)
                            diff = -1 * (future_price.iloc[4] - now_price)

                        elif (mode=="sell") & (get_dl_sig["DL_sig"].item() == 2):
                            state = "maturity"
                            ret = (future_price.iloc[4] / now_price - 1)
                            diff = (future_price.iloc[4] - now_price)


                bm_ret = -1 * (future_price.iloc[-1] / now_price - 1)
                if mode == "sell":
                    bm_ret *= -1

                profit_collect.append([i_d, i_tm, model, state, ret, diff, bm_ret, volat])
        profit_collect = pd.DataFrame(profit_collect,
                                      columns=["date", "tm", "model", "order_result", "ret",
                                               "price_diff", "bm_ret", "volat"])

        invest_amt = 1000000000
        profit_collect["model_profit"] = profit_collect["ret"] * invest_amt
        profit_collect["bm_profit"] = profit_collect["bm_ret"] * invest_amt
        profit_collect["model_cum_profit"] = np.nancumsum(profit_collect["model_profit"])
        profit_collect["bm_cum_profit"] = np.nancumsum(profit_collect["bm_profit"])

        return profit_collect

    def result(self):
        profit_vanilla_model_buy = self.cal_profit_quant(
            mode="buy",
            reversion_overlay=True,
            dl_overlay=True
        )
        profit_vanilla_model_sell = self.cal_profit_quant(
            mode="sell",
            reversion_overlay=True,
            dl_overlay=True
        )

        return profit_vanilla_model_buy, profit_vanilla_model_sell #, self.st_time, self.en_time

def _dt_resampling(data, date):
    """
        Data Formatting : [datetime, DL_signal] 
            datetime : 900 ~ 1530 ( signal이 없는 경우, None )
    """
    

    mk_open_dt, mk_close_dt    = f'{date}900', f'{date}1530'
    idx_first_row_t = data.iloc[0]['mmss']
    if idx_first_row_t != '900':
        open_data = {'datetime': [mk_open_dt], f'optim_real_u' : [np.nan] , 'optim_real_b' : [np.nan], 'optim_pred_u' : [np.nan], 'optim_pred_u' : [np.nan]}
        open_df   = pd.DataFrame(open_data, columns = data.columns)
        data      = pd.concat([open_df, data])

    idx_last_row_t = data.iloc[-1]['mmss']
    if idx_last_row_t != '1530':
        close_data = {'datetime': [mk_close_dt],  f'optim_real_u' : [np.nan] , 'optim_real_b' : [np.nan], 'optim_pred_u' : [np.nan], 'optim_pred_u' : [np.nan]}
        close_df = pd.DataFrame(close_data, columns = data.columns)
        data = pd.concat([data, close_df])

    data['datetime'] = pd.to_datetime(data['datetime'], format = '%Y-%m-%d%H%M')
    data = data.set_index('datetime')
    data = data.resample('T').first()
    return data

def _input_batch_maker(cur_time, time_interval, features, data):
    if features == 'fx_ohlc_dxy_hlc_ks200':
        interval_fx_open    = data[0][ cur_time - time_interval : cur_time]
        interval_fx_high    = data[1][ cur_time - time_interval : cur_time]
        interval_fx_low     = data[2][ cur_time - time_interval : cur_time]
        interval_fx_close   = data[3][ cur_time - time_interval : cur_time]

        interval_dxy_high   = data[4][ cur_time - time_interval : cur_time]
        interval_dxy_low    = data[5][ cur_time - time_interval : cur_time]
        interval_dxy_close  = data[6][ cur_time - time_interval : cur_time]

        interval_ks200      = data[7][ cur_time - time_interval : cur_time]


        interval_fx_open    = np.expand_dims(interval_fx_open, 1)
        interval_fx_high    = np.expand_dims(interval_fx_high, 1)
        interval_fx_low     = np.expand_dims(interval_fx_low, 1)
        interval_fx_close   = np.expand_dims(interval_fx_close, 1)

        interval_dxy_high   = np.expand_dims(interval_dxy_high, 1)
        interval_dxy_low    = np.expand_dims(interval_dxy_low, 1)
        interval_dxy_close  = np.expand_dims(interval_dxy_close, 1)

        interval_ks200      = np.expand_dims(interval_ks200, 1)

        input_batch    = np.concatenate([interval_fx_open, interval_fx_high, interval_fx_low, interval_fx_close, interval_dxy_high, interval_dxy_low, interval_dxy_close, interval_ks200], axis=1)
    
    if np.sum(np.sum(np.isnan(input_batch), axis=1), axis=0) == 0:
        input_batch = np.expand_dims(input_batch, 0) 
        return torch.from_numpy(input_batch)
    else:
        return None 

def make_signal_table(model, time_interval, f_horizon, feature_mode, model_name):
    
    ##### 01. Parameter 설정 #####
    test_start_date = "2021-01-04"
    time_interval   = time_interval
    f_horizon       = 5

    ##### 02. Data Load #####
    test_data_dir      = f'/home/jovyan/shared/wealth-solution/FX_modeling/Transformer_optim/Datas_for_donwload'

    fx_open_dat   = pd.read_csv(f'{test_data_dir}/test_usdkrw_open_reshape.csv'  , index_col=0)
    fx_high_dat   = pd.read_csv(f'{test_data_dir}/test_usdkrw_high_reshape.csv'  , index_col=0)
    fx_low_dat    = pd.read_csv(f'{test_data_dir}/test_usdkrw_low_reshape.csv'   , index_col=0)
    fx_close_dat  = pd.read_csv(f'{test_data_dir}/test_usdkrw_close_reshape.csv' , index_col=0)
    dxy_high_dat  = pd.read_csv(f'{test_data_dir}/test_dxy_high_reshape.csv'     , index_col=0)
    dxy_low_dat   = pd.read_csv(f'{test_data_dir}/test_dxy_low_reshape.csv'      , index_col=0)
    dxy_close_dat = pd.read_csv(f'{test_data_dir}/test_dxy_close_reshape.csv'    , index_col=0)
    # cnh_mid_dat   = pd.read_csv(f'{test_data_dir}/test_cnh_mid_reshape.csv'      , index_col=0)
    kospi200      = pd.read_csv(f'{test_data_dir}/test_ks200_reshape.csv'        , index_col=0)
    
    targets       = pd.read_csv(f'/home/jovyan/shared/wealth-solution/FX_modeling/Transformer_optim/Datas/raw_datas/labels/5bp_after5_ud_binary_classes.csv', index_col=0).loc[:1530]

    sig_save_filename   = f'{model_name}_summary.csv'

    ###### 03. Date Iterate #####
    btest_infos       = None
    iter_dates = list(fx_close_dat.columns)
    for date_key in iter_dates:
        
        iter_date_fx_open     = fx_open_dat[date_key]
        iter_date_fx_high     = fx_high_dat[date_key]
        iter_date_fx_low      = fx_low_dat[date_key]
        iter_date_fx_close    = fx_close_dat[date_key]
        iter_date_dxy_high    = dxy_high_dat[date_key]
        iter_date_dxy_low     = dxy_low_dat[date_key]
        iter_date_dxy_close   = dxy_close_dat[date_key]
        # iter_date_cnh_mid     = cnh_mid_dat[date_key]
        iter_date_ks200_close = kospi200[date_key]

        iter_date_labels      = targets[date_key].reset_index(drop=True)
        proc_times            = iter_date_labels.reset_index().dropna()['index']
 
        print(f"INFO: Making DL SIGNAL of [{date_key}]")
        
        liks, predictions, proc_tm_idxs, labels = list(), list(), list(), list()
        for proc_tm in proc_times:
            
            if (proc_tm < time_interval) or (proc_tm > (391 - f_horizon)):
                continue
            
            batch       = _input_batch_maker(cur_time = int(proc_tm), time_interval = time_interval, features = feature_mode,
                                            data     = (np.array(iter_date_fx_open), np.array(iter_date_fx_high), np.array(iter_date_fx_low), np.array(iter_date_fx_close), 
                                                        np.array(iter_date_dxy_high), np.array(iter_date_dxy_low), np.array(iter_date_dxy_close), np.array(iter_date_ks200_close)))

            lik, pred = np.nan, np.nan
            if batch is not None:
                output = model(batch.type(torch.FloatTensor))
                if np.isnan(output.tolist()).sum() == 0:
                    lik, pred         = torch.max(output, 1)
                    lik               = lik.detach().numpy()[0]
                    pred              = 2 if pred == 1 else 0


            liks.append(lik)
            predictions.append(pred)
            proc_tm_idxs.append(proc_tm)                         
            labels.append(iter_date_labels.iloc[proc_tm])   
        
        ###### 04. Concat Data #####
        tmp_pred         = pd.DataFrame(predictions, columns = [f'DL_sig'], index = proc_tm_idxs)
        tmp_lik          = pd.DataFrame(liks, columns = [f'liks'], index = proc_tm_idxs)
        tmp_label        = pd.DataFrame(labels, columns = [f'labels'], index = proc_tm_idxs)

        tmp_fx_df              = iter_date_fx_close.reset_index()
        tmp_fx_df.columns      = ['mmss', 'Pt']

        tmp_infos              = pd.concat([tmp_lik, tmp_pred, tmp_label, tmp_fx_df], axis =  1)
        tmp_infos['date']      = date_key
        tmp_infos              = tmp_infos.astype({'date' : 'str', 'mmss' : 'str'})
        tmp_infos['datetime']  = tmp_infos['date'] + tmp_infos['mmss']
        
        tmp_infos              = tmp_infos[['datetime', 'date', 'mmss', 'liks', 'DL_sig', 'labels']]
        tmp_infos              = tmp_infos.reset_index().sort_values(by=['index'], axis=0).set_index('index')
        tmp_infos              = _dt_resampling(data = tmp_infos, date = date_key)
        
        tmp_infos['date'] = date_key
        btest_infos = tmp_infos if btest_infos is None else pd.concat([btest_infos, tmp_infos])
    
    btest_infos = btest_infos[['date', 'mmss', 'liks', 'DL_sig']]
    btest_infos.columns = ['date' , 'time', 'liks', 'DL_sig'] 
    btest_infos['date'] = btest_infos['date'].str.replace('-', '')
    btest_infos.to_csv(f'./Datas/signals/deeplearning_model_res/{sig_save_filename}', index = False)
    
def save_and_trace_model(model_path, model_savename, time_interval, enc_in):
    model_savepath = f"/home/jovyan/shared/wealth-solution/FX_modeling/Transformer_optim/models/traced_models/{model_savename}.pt"
    model = torch.load(model_path)
    model = model.float()
    model = model.to(torch.device('cpu'))
    model.eval()

    ## Check dummy input shape
    input_dummy = torch.rand(1, time_interval, enc_in, dtype=torch.float32)
    input_dummy = input_dummy.to(torch.device('cpu'))

    traced_script_module = torch.jit.trace(model, input_dummy)
    traced_script_module.save(model_savepath)
    
    traced_model = torch.load(model_savepath)
    traced_model.eval()
    
    return traced_model

if __name__ == '__main__':
    
    '''
        [save_and_trace_model] 타 프레임워크에서 사용할 수 있도록 모델 Trace 및 Save 
        model_path : 불러올 모델 경로 설정 
        model_savename : 저장할 모델 이름 설정 
        
        [make_signal_table] traced 모델의 시그널 테이블 생성          
        model_path : 불러올 모델 경로 설정 
        model_savename : 저장할 모델 이름 설정 
    '''
    
    time_interval  = 30  # 120
    epoch          = 180 # 220
    lr             = 0.00001

    model_path     = f"/home/jovyan/shared/wealth-solution/FX_modeling/Transformer_optim/models/saveDB/Classification/Patchmixer_input_ohlc_dxy_hlc_ks200_lr0.00001_epoch180_Softmax/5bp_after5_ud_epoch_180.pt"

    feature_mode   = 'fx_ohlc_dxy_hlc_ks200'
    model_savename = f"traced_5bp_after5_ud_fver_{feature_mode}_lr0.00001_epoch{epoch}"

    # 120분 모델
    #  30분 모델
    savename_120m = f"traced_5bp_after5_ud_fver_{feature_mode}_lr0.000001_epoch220"
    savename_30m = f"traced_5bp_after5_ud_fver_{feature_mode}_lr0.00001_epoch200"

    import pdb; pdb.set_trace()
    # model_savepath = f"/home/jovyan/shared/wealth-solution/FX_modeling/Transformer_optim/models/traced_models/{model_savename}.pt"
    # traced_model = torch.load(model_savepath)

    # traced_model   = save_and_trace_model(model_path=model_path, model_savename=model_savename, time_interval=time_interval, enc_in=8)
    # make_signal_table(model=traced_model, time_interval=time_interval, f_horizon=5, feature_mode=feature_mode, model_name=model_savename)
    
    file_dict = {
        "deeplearning_data": f'{model_savename}_summary.csv',
        "quant_model" : "reversion_wr_v1.csv",
        "prev_model": "hana_fx_engine_signal_reshaped.csv",
        "close_price": "usdkrw_futures_12yr_close_reshape.csv",
        "test_st_date": "20210101",

        "120m_data" : f'{savename_120m}_summary.csv',
        "30m_data"  : f'{savename_30m}_summary.csv'
    }

    profit_vanilla_model_buy, profit_vanilla_model_sell = EnsembleResult(file_dict=file_dict).result()
    time_120m_st, time_120m_en = EnsembleResult(file_dict=file_dict).st_time, EnsembleResult(file_dict=file_dict).en_time

    plot_inventory_profit( dat1  = profit_vanilla_model_buy.bm_cum_profit   , dat1_label = 'fx_engine_v1', 
                           dat2  = profit_vanilla_model_buy.model_cum_profit, dat2_label = 'PatchMixer Model',
                           title = f'[BUY CASE] fx-engine v1.0 <> PatchMixer Model') # {time_120m_st} ~ {time_120m_en} 120m model + 30m model
    plot_inventory_profit( dat1  = profit_vanilla_model_sell.bm_cum_profit   , dat1_label = 'fx_engine_v1', 
                           dat2  = profit_vanilla_model_sell.model_cum_profit, dat2_label = 'PatchMixer Model',
                           title = f'[SELL CASE] fx-engine v1.0 <> PatchMixer Model')

    profit_vanilla_model_buy.to_csv(f'./Performance/profit_vanilla_model_buy_PatchMixer Model.csv') # {time_120m_st}_{time_120m_en}_120m_30m
    profit_vanilla_model_buy.to_csv(f'./Performance/profit_vanilla_model_sell_PatchMixer Model.csv')
    print_performance(prof_model = profit_vanilla_model_buy , key = 'BUY')
    print_performance(prof_model = profit_vanilla_model_sell, key = 'SELL')