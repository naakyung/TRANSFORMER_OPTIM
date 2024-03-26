import os, math, random 
from tqdm import tqdm
from datetime import datetime

import torch
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from srcs.utils import Datasets, features

import warnings 
warnings.filterwarnings('ignore')

class dataLoader:
    
    def __init__(self, cfg):
        self._init_cfg(cfg)
        
        if (f'{self.predict_type}_interval{self.time_interval}_{self.feature_mode}_{self.label_mode}_train_data.npz' not in os.listdir(f"{cfg.dataset.fx_data_dir}")) or (f'{self.predict_type}_interval{self.time_interval}_{self.feature_mode}_{self.label_mode}_test_data.npz' not in os.listdir(f"{cfg.dataset.fx_data_dir}")):
            print(f"\nINFO: [__init__] We don't have feature files. Start Processing...")
            self._init_processing()
        else:
            print(f"\nINFO: [__init__] loading ... {self.predict_type}_interval{self.time_interval}_{self.feature_mode}_{self.label_mode}_train_data.npz")
            print(f"INFO: [__init__] loading ... {self.predict_type}_interval{self.time_interval}_{self.feature_mode}_{self.label_mode}_val_data.npz")
            print(f"INFO: [__init__] loading ... {self.predict_type}_interval{self.time_interval}_{self.feature_mode}_{self.label_mode}_test_data.npz\n")
            
            if self.data_padding:
                train_x1_dat_npz                    = np.load(f'{cfg.dataset.fx_data_dir}/{self.predict_type}_interval{self.time_interval}_{self.feature_mode}_{self.label_mode}_train_x1_data.npz', allow_pickle=True)
                train_x2_dat_npz                    = np.load(f'{cfg.dataset.fx_data_dir}/{self.predict_type}_interval{self.time_interval}_{self.feature_mode}_{self.label_mode}_train_x2_data.npz', allow_pickle=True)
            else:
                train_dat_npz                       = np.load(f'{cfg.dataset.fx_data_dir}/{self.predict_type}_interval{self.time_interval}_{self.feature_mode}_{self.label_mode}_train_data.npz', allow_pickle=True)
                
            val_dat_npz                         = np.load(f'{cfg.dataset.fx_data_dir}/{self.predict_type}_interval{self.time_interval}_{self.feature_mode}_{self.label_mode}_val_data.npz' , allow_pickle=True)
            test_dat_npz                        = np.load(f'{cfg.dataset.fx_data_dir}/{self.predict_type}_interval{self.time_interval}_{self.feature_mode}_{self.label_mode}_test_data.npz' , allow_pickle=True)

            if self.feature_mode in ['c', 'momentum']:
                self.train_x_npy, self.train_y_npy  = np.expand_dims(train_dat_npz["x"], 2), train_dat_npz["y"]
                self.test_x_npy, self.test_y_npy    = np.expand_dims(test_dat_npz["x"], 2), test_dat_npz["y"]
            elif self.data_padding:
                self.train_x_npy   = np.concatenate((train_x1_dat_npz["x"], train_x2_dat_npz["x"]), axis = 0)
                self.train_y_npy   = np.concatenate((train_x1_dat_npz["y"], train_x2_dat_npz["y"]), axis = 0)
                self.val_x_npy,  self.val_y_npy     = val_dat_npz["x"], val_dat_npz["y"]
                self.test_x_npy, self.test_y_npy    = test_dat_npz["x"], test_dat_npz["y"]
                train_x1_dat_npz.close()
                train_x2_dat_npz.close()
            else:
                self.train_x_npy, self.train_y_npy  = train_dat_npz["x"], train_dat_npz["y"]
                self.val_x_npy,  self.val_y_npy     = val_dat_npz["x"], val_dat_npz["y"]
                self.test_x_npy, self.test_y_npy    = test_dat_npz["x"], test_dat_npz["y"]

            val_dat_npz.close()
            test_dat_npz.close()
    
    def _init_cfg(self, cfg):
        
        ##### (1) 필요한 파라미터 정의 #####
        self.seed               = cfg.dataset.seed
        self.scaling            = cfg.dataset.scaling 
        self.volat_grouping     = cfg.dataset.volat_grouping
        self.data_padding       = cfg.dataset.data_padding
        self.data_patching      = cfg.dataset.data_patching 
        self.feature_mode       = cfg.dataset.feature_mode
        # import pdb; pdb.set_trace()
        if self.data_padding:
            self.feature_mode   = 'padding_' + self.feature_mode

        self.predict_type       = cfg.dataset.predict_type                                                        # "Classification", "Regression"
        self.label_mode         = cfg.dataset.label_mode                                                          # "optim_u", "optim_b", "ub_spread" 
        self.label_class        = cfg.model.num_classes

        self.time_interval      = int(cfg.dataset.time_interval)

        self.trn_batch_size     = cfg.model.batch_size
        self.val_batch_size     = cfg.dataset.val_batch_size

        self.srt_mrk_t          = int(cfg.dataset.market_start_time)                                              # Market Open Time  : 900
        self.end_mrk_t          = int(cfg.dataset.market_end_time)                                                # Market Close Time : 1530
        self.trn_split_point    = datetime.strptime(cfg.dataset.train_split_point, "%m%d%y").strftime("%Y-%m-%d") # "2016-07-29"
        self.val_st_dt          = datetime.strptime(cfg.dataset.val_start_date, "%m%d%y").strftime("%Y-%m-%d")    # "2019-01-02"
        self.test_st_dt         = datetime.strptime(cfg.dataset.test_start_date, "%m%d%y").strftime("%Y-%m-%d")   # "2021-01-04"

        ##### (2) Feature/Target에 따라 필요한 데이터 로드 #####      
        print(f"INFO: [_init_cfg] predict_type = {self.predict_type},  label mode = {self.label_mode},  label class = {self.label_class}")
        if self.predict_type == 'Regression':
            self.target_dat                 = pd.read_csv(os.path.join(cfg.dataset.fx_data_dir, "raw_datas", "labels",  f"{self.label_mode}_Regression_ver1.csv"), index_col=0)
        elif self.predict_type == 'Classification':
            if self.label_class == 2:
               self.target_dat              = pd.read_csv(os.path.join(cfg.dataset.fx_data_dir, "raw_datas", "labels",  f'{self.label_mode}_binary_classes.csv'), index_col = 0)
            elif self.label_class > 2:
                self.target_dat             = pd.read_csv(os.path.join(cfg.dataset.fx_data_dir, "raw_datas", "labels",  f"{self.label_mode}_multi_classes.csv"), index_col=0)
        
        print(f"INFO: [_init_cfg] feature mode = {self.feature_mode}")
        self.fx_close_dat           = pd.read_csv(os.path.join(cfg.dataset.fx_data_dir, "raw_datas", "usdkrw_futures_12yr_close_reshape.csv"), index_col=0).loc[:1530].fillna(method="ffill")
        if self.feature_mode  in ['cub', 'scaled_cub']:
            self.interval_u         = pd.read_csv(os.path.join(cfg.dataset.fx_data_dir, "raw_datas", "optim_u_ver1_shift.csv"), index_col=0)
            self.interval_b         = pd.read_csv(os.path.join(cfg.dataset.fx_data_dir, "raw_datas", "optim_b_ver1_shift.csv"), index_col=0)
        elif self.feature_mode in ['cfreq', 'scaled_cub']:
            self.freq_by_sin        = pd.read_csv(os.path.join(cfg.dataset.fx_data_dir, "raw_datas", "sin_freqs_features.csv"), index_col=0)
        elif (self.feature_mode in ['mvf', 'cmvf', 'scaled_mvf', 'scaled_cmvf']):
            self.momentum           = pd.read_csv(os.path.join(cfg.dataset.fx_data_dir, "raw_datas", f"momentum_feature_interval{self.time_interval}.csv"), index_col=0)
            self.volatility         = pd.read_csv(os.path.join(cfg.dataset.fx_data_dir, "raw_datas", f"volatility_feature_interval{self.time_interval}.csv"), index_col=0)
            self.freq_by_sin        = pd.read_csv(os.path.join(cfg.dataset.fx_data_dir, "raw_datas", f"sin_freq_feature_interval{self.time_interval}.csv"), index_col=0)
        elif (self.feature_mode in ['scaled_mamv_kospi', 'scaled_mamv_kospi_ret', 'scaled_mamv_kospi_ret_5wr']):
            self.kospi200           = pd.read_csv(os.path.join(cfg.dataset.fx_data_dir, "raw_datas", "kospi200_futures_agg.csv"), index_col=0).fillna(method="ffill") 
            self.kospi200.columns   = self.kospi200.columns.str[:4] + '-' + self.kospi200.columns.str[4:6] + '-' + self.kospi200.columns.str[6:]
            avaliable_columns       = sorted(set(self.fx_close_dat.columns) & set(self.kospi200.columns))
            
            self.fx_close_dat       = self.fx_close_dat.loc[: , avaliable_columns]
            self.kospi200           = self.kospi200.loc[: , avaliable_columns]
        elif (self.feature_mode in ['scaled_mamv_kospi_ret_5wr_dxy_ret']):
            self.kospi200           = pd.read_csv(os.path.join(cfg.dataset.fx_data_dir, "raw_datas", "kospi200_futures_agg.csv"), index_col=0).loc[:1530]
            self.dxy_close_dat      = pd.read_csv(os.path.join(cfg.dataset.fx_data_dir, "raw_datas", "DXY_close_9yr_reshape.csv"), index_col=0).loc[:1530].fillna(method="ffill") 
            self.kospi200.columns   = self.kospi200.columns.str[:4] + '-' + self.kospi200.columns.str[4:6] + '-' + self.kospi200.columns.str[6:]
            
            avaliable_columns       = sorted(set(self.fx_close_dat.columns) & set(self.kospi200.columns) & set(self.dxy_close_dat.columns))
            
            self.fx_close_dat       = self.fx_close_dat.loc [: , avaliable_columns]
            self.kospi200           = self.kospi200.loc     [: , avaliable_columns]
            self.dxy_close_dat      = self.dxy_close_dat.loc[: , avaliable_columns]
        elif (self.feature_mode in ['scaled_hlcma_mv_kospi_ret_5wr']):
            self.fx_high_dat        = pd.read_csv(os.path.join(cfg.dataset.fx_data_dir, "raw_datas", "usdkrw_futures_12yr_high_reshape.csv"), index_col=0).loc[:1530].fillna(method="ffill")
            self.fx_low_dat         = pd.read_csv(os.path.join(cfg.dataset.fx_data_dir, "raw_datas", "usdkrw_futures_12yr_low_reshape.csv"), index_col=0).loc[:1530].fillna(method="ffill")
            
            self.kospi200           = pd.read_csv(os.path.join(cfg.dataset.fx_data_dir, "raw_datas", "kospi200_futures_agg.csv"), index_col=0).loc[:1530]
            self.kospi200.columns   = self.kospi200.columns.str[:4] + '-' + self.kospi200.columns.str[4:6] + '-' + self.kospi200.columns.str[6:]
            avaliable_columns       = sorted(set(self.fx_close_dat.columns) & set(self.kospi200.columns))
            
            self.fx_close_dat, self.fx_high_dat, self.fx_low_dat        = self.fx_close_dat.loc[: , avaliable_columns], self.fx_high_dat.loc[: , avaliable_columns], self.fx_low_dat.loc[: , avaliable_columns]
            self.kospi200                                               = self.kospi200.loc[: , avaliable_columns]
        elif self.feature_mode in ['ohlc_dxy_hlc']:
            self.fx_open_dat       = pd.read_csv(os.path.join(cfg.dataset.fx_data_dir, "raw_datas", "usdkrw_futures_12yr_open_reshape.csv"), index_col=0).loc[900:1530].fillna(method="ffill")
            self.fx_high_dat       = pd.read_csv(os.path.join(cfg.dataset.fx_data_dir, "raw_datas", "usdkrw_futures_12yr_high_reshape.csv"), index_col=0).loc[900:1530].fillna(method="ffill")
            self.fx_low_dat        = pd.read_csv(os.path.join(cfg.dataset.fx_data_dir, "raw_datas", "usdkrw_futures_12yr_low_reshape.csv"), index_col=0).loc[900:1530].fillna(method="ffill")

            self.dxy_high_dat      = pd.read_csv(os.path.join(cfg.dataset.fx_data_dir, "raw_datas", "DXY_9yr_high_reshape.csv"), index_col=0).loc[900:1530].fillna(method="ffill") 
            self.dxy_low_dat       = pd.read_csv(os.path.join(cfg.dataset.fx_data_dir, "raw_datas", "DXY_9yr_low_reshape.csv"), index_col=0).loc[900:1530].fillna(method="ffill")
            self.dxy_close_dat     = pd.read_csv(os.path.join(cfg.dataset.fx_data_dir, "raw_datas", "DXY_9yr_close_reshape.csv"), index_col=0).loc[900:1530].fillna(method="ffill")   
        elif self.feature_mode in ['ohlc_dxy_hlc_ks', 'padding_ohlc_dxy_hlc_ks']:
            self.fx_open_dat       = pd.read_csv(os.path.join(cfg.dataset.fx_data_dir, "raw_datas", "usdkrw_futures_12yr_open_reshape.csv"), index_col=0).loc[900:1530].fillna(method="ffill")
            self.fx_high_dat       = pd.read_csv(os.path.join(cfg.dataset.fx_data_dir, "raw_datas", "usdkrw_futures_12yr_high_reshape.csv"), index_col=0).loc[900:1530].fillna(method="ffill")
            self.fx_low_dat        = pd.read_csv(os.path.join(cfg.dataset.fx_data_dir, "raw_datas", "usdkrw_futures_12yr_low_reshape.csv"), index_col=0).loc[900:1530].fillna(method="ffill")

            self.dxy_high_dat      = pd.read_csv(os.path.join(cfg.dataset.fx_data_dir, "raw_datas", "DXY_9yr_high_reshape.csv"), index_col=0).loc[900:1530].fillna(method="ffill") 
            self.dxy_low_dat       = pd.read_csv(os.path.join(cfg.dataset.fx_data_dir, "raw_datas", "DXY_9yr_low_reshape.csv"), index_col=0).loc[900:1530].fillna(method="ffill")
            self.dxy_close_dat     = pd.read_csv(os.path.join(cfg.dataset.fx_data_dir, "raw_datas", "DXY_9yr_close_reshape.csv"), index_col=0).loc[900:1530].fillna(method="ffill")     
            
            self.kospi200_dat      = pd.read_csv(os.path.join(cfg.dataset.fx_data_dir, "raw_datas", "kospi200_futures_agg.csv"), index_col=0).loc[900:1530].fillna(method='ffill')
            self.kospi200_dat.columns  = self.kospi200_dat.columns.str[:4] + '-' + self.kospi200_dat.columns.str[4:6] + '-' + self.kospi200_dat.columns.str[6:]
            
            avaliable_columns                     = sorted(set(self.fx_close_dat.columns) & set(self.dxy_close_dat.columns) & set(self.kospi200_dat.columns) & set(self.target_dat.columns))

            self.fx_open_dat, self.fx_close_dat   = self.fx_open_dat.loc  [: , avaliable_columns], self.fx_close_dat.loc[: , avaliable_columns]
            self.fx_high_dat, self.fx_low_dat     = self.fx_high_dat.loc  [: , avaliable_columns], self.fx_low_dat.loc  [: , avaliable_columns]
            self.dxy_high_dat, self.dxy_low_dat   = self.dxy_high_dat.loc [: , avaliable_columns], self.dxy_low_dat.loc [: , avaliable_columns] 
            self.dxy_close_dat, self.kospi200_dat = self.dxy_close_dat.loc[: , avaliable_columns], self.kospi200_dat.loc[: , avaliable_columns]
            self.target_dat                       = self.target_dat.loc   [: , avaliable_columns]              
        elif self.feature_mode in ['ohlc_dxy_hlc_cnh_ks']:
            self.fx_open_dat       = pd.read_csv(os.path.join(cfg.dataset.fx_data_dir, "raw_datas", "usdkrw_futures_12yr_open_reshape.csv"), index_col=0).loc[900:1530].fillna(method="ffill")
            self.fx_high_dat       = pd.read_csv(os.path.join(cfg.dataset.fx_data_dir, "raw_datas", "usdkrw_futures_12yr_high_reshape.csv"), index_col=0).loc[900:1530].fillna(method="ffill")
            self.fx_low_dat        = pd.read_csv(os.path.join(cfg.dataset.fx_data_dir, "raw_datas", "usdkrw_futures_12yr_low_reshape.csv"), index_col=0).loc[900:1530].fillna(method="ffill")

            self.dxy_high_dat      = pd.read_csv(os.path.join(cfg.dataset.fx_data_dir, "raw_datas", "DXY_9yr_high_reshape.csv"), index_col=0).loc[900:1530].fillna(method="ffill") 
            self.dxy_low_dat       = pd.read_csv(os.path.join(cfg.dataset.fx_data_dir, "raw_datas", "DXY_9yr_low_reshape.csv"), index_col=0).loc[900:1530].fillna(method="ffill")
            self.dxy_close_dat     = pd.read_csv(os.path.join(cfg.dataset.fx_data_dir, "raw_datas", "DXY_9yr_close_reshape.csv"), index_col=0).loc[900:1530].fillna(method="ffill")     
            
            self.cnh_mid_dat       = pd.read_csv(os.path.join(cfg.dataset.fx_data_dir, "raw_datas", "CNH_9yr_mid_reshape.csv"), index_col=0).loc[900:1530].fillna(method='ffill')
            
            self.kospi200_dat      = pd.read_csv(os.path.join(cfg.dataset.fx_data_dir, "raw_datas", "kospi200_futures_agg.csv"), index_col=0).loc[900:1530].fillna(method='ffill')
            self.kospi200_dat.columns  = self.kospi200_dat.columns.str[:4] + '-' + self.kospi200_dat.columns.str[4:6] + '-' + self.kospi200_dat.columns.str[6:]
            
            avaliable_columns                     = sorted(set(self.fx_close_dat.columns) & set(self.dxy_close_dat.columns) & set(self.cnh_mid_dat.columns)& set(self.kospi200_dat.columns) & set(self.target_dat.columns))
            self.fx_open_dat, self.fx_close_dat   = self.fx_open_dat.loc  [: , avaliable_columns], self.fx_close_dat.loc[: , avaliable_columns]
            self.fx_high_dat, self.fx_low_dat     = self.fx_high_dat.loc  [: , avaliable_columns], self.fx_low_dat.loc  [: , avaliable_columns]
            self.dxy_high_dat, self.dxy_low_dat   = self.dxy_high_dat.loc [: , avaliable_columns], self.dxy_low_dat.loc [: , avaliable_columns] 
            self.dxy_close_dat, self.kospi200_dat = self.dxy_close_dat.loc[: , avaliable_columns], self.kospi200_dat.loc[: , avaliable_columns]
            self.cnh_mid_dat                      = self.cnh_mid_dat.loc  [: , avaliable_columns]
            self.target_dat                       = self.target_dat.loc   [: , avaliable_columns]   
            
    def _init_processing(self):
        
        ##### (1) Data Split & Make Features #####
        trn_target_bfr    = self.target_dat.loc[ :1500,                       :  self.trn_split_point].iloc[:,:-1]
        trn_target_aft    = self.target_dat.loc[ :1530, self.trn_split_point  :         self.val_st_dt]
        val_target_dat    = self.target_dat.loc[ :1530, self.val_st_dt        :        self.test_st_dt]
        test_target_dat   = self.target_dat.loc[ :1530, self.test_st_dt       :                       ]
                
        trn_bfr_input_sets, trn_aft_input_sets, val_input_sets, test_input_sets = self._make_features()
        
        ##### (2) Make Input-Frame #####
        train_x_1, train_y_1  = self._base_arrange_data(input_dat=trn_bfr_input_sets, target_dat=trn_target_bfr)
        train_x_2, train_y_2  = self._base_arrange_data(input_dat=trn_aft_input_sets, target_dat=trn_target_aft)
        train_x, train_y      = np.concatenate((train_x_1, train_x_2), axis=0), np.concatenate((train_y_1, train_y_2), axis=0)
        val_x, val_y          = self._base_arrange_data(input_dat=val_input_sets    , target_dat=val_target_dat)
        test_x, test_y        = self._base_arrange_data(input_dat=test_input_sets   , target_dat=test_target_dat)

        ##### (3) Data Sorting #####
        ##### (3-1) Data Sorting : get available_loc #####   
        if self.volat_grouping:
            available_train_x_loc           = np.where(np.sum(np.sum(np.isnan(train_x[:,:,:-1]), axis=1), axis=1) == 0)[0]
            available_val_x_loc             = np.where(np.sum(np.sum(np.isnan(val_x[:,:,:-1]), axis=1), axis=1) == 0)[0]
            available_test_x_loc            = np.where(np.sum(np.sum(np.isnan(test_x[:,:,:-1]), axis=1), axis=1) == 0)[0]
        else:
            available_train_x_loc           = np.where(np.sum(np.sum(np.isnan(train_x[:,:,:]), axis=1), axis=1) == 0)[0]
            available_val_x_loc             = np.where(np.sum(np.sum(np.isnan(val_x[:,:,:]), axis=1), axis=1) == 0)[0]
            available_test_x_loc            = np.where(np.sum(np.sum(np.isnan(test_x[:,:,:]), axis=1), axis=1) == 0)[0]
        
        available_train_y_loc               = np.where(np.isnan(train_y) == False)[0]
        available_val_y_loc                 = np.where(np.isnan(val_y) == False)[0]
        available_test_y_loc                = np.where(np.isnan(test_y) == False)[0]

        available_train_loc                 = sorted(list(set(available_train_x_loc) & set(available_train_y_loc)))
        available_val_loc                   = sorted(list(set(available_val_x_loc) & set(available_val_y_loc)))
        available_test_loc                  = sorted(list(set(available_test_x_loc) & set(available_test_y_loc)))

        ##### (3-2) Data Sorting : available_loc extraction #####
        import pdb; pdb.set_trace()
        self.train_x_npy, self.train_y_npy  = train_x[available_train_loc, :, :], train_y[available_train_loc]    
        self.val_x_npy  , self.val_y_npy    = val_x  [available_val_loc  , :, :], val_y  [available_val_loc]
        self.test_x_npy , self.test_y_npy   = test_x [available_test_loc , :, :], test_y [available_test_loc]
        
        ##### (* Options) #####
        ##### (*5) Data Patching : 30분 >> 5분 씩 6개의 조각으로 보기 위함 #####
        if self.data_patching:
            self.train_x_npy = self.train_x_npy[:, 4::5, :]
            self.val_x_npy   = self.val_x_npy[:, 4::5, :]
            self.test_x_npy  = self.test_x_npy[:, 4::5, :]
    
        ##### (*6) Data Grouping  #####
        if self.volat_grouping:
            self._data_grouping(key = 'volatility')
            
        import pdb; pdb.set_trace()
        ##### (5) Data 저장 #####
        np.savez_compressed(f"./Datas/{self.predict_type}_interval{self.time_interval}_{self.feature_mode}_{self.label_mode}_train_data.npz", x = self.train_x_npy, y = self.train_y_npy)
        np.savez_compressed(f"./Datas/{self.predict_type}_interval{self.time_interval}_{self.feature_mode}_{self.label_mode}_val_data.npz", x = self.val_x_npy, y = self.val_y_npy)
        np.savez_compressed(f"./Datas/{self.predict_type}_interval{self.time_interval}_{self.feature_mode}_{self.label_mode}_test_data.npz" , x = self.test_x_npy , y = self.test_y_npy)
        
    def _make_features(self):
        
        ##### (1-1) Split Data #####
        if self.feature_mode  in ['cub']:                                                                                        
            trn_iu_bfr         = self.interval_u.loc[ :1500,                       :  self.trn_split_point].iloc[:,:-1]
            trn_iu_aft         = self.interval_u.loc[ :1530, self.trn_split_point  :       self.test_st_dt]
            test_iu_dat        = self.interval_u.loc[ :1530, self.test_st_dt       :                      ]

            trn_ib_bfr         = self.interval_b.loc[ :1500,                       :  self.trn_split_point].iloc[:,:-1]
            trn_ib_aft         = self.interval_b.loc[ :1530, self.trn_split_point  :       self.test_st_dt]
            test_ib_dat        = self.interval_b.loc[ :1530, self.test_st_dt       :                      ]
        elif self.feature_mode in ['cfreq']: 
            trn_freq_bfr       = self.freq_by_sin.loc[ :1500,                       :  self.trn_split_point].iloc[:,:-1]
            trn_freq_aft       = self.freq_by_sin.loc[ :1530, self.trn_split_point  :       self.test_st_dt]
            test_freq_dat      = self.freq_by_sin.loc[ :1530, self.test_st_dt       :                      ]
        elif self.feature_mode in ['mvf', 'cmvf', 'scaled_mvf', 'scaled_cmvf'] :
            trn_momentum_bfr   = self.momentum.loc[ :1500,                       :  self.trn_split_point].iloc[:,:-1]
            trn_momentum_aft   = self.momentum.loc[ :1530, self.trn_split_point  :           self.val_st_dt]
            val_momentum_dat   = self.momentum.loc[ :1530, self.val_st_dt        :          self.test_st_dt]
            test_momentum_dat  = self.momentum.loc[ :1530, self.test_st_dt       :                         ]

            trn_volat_bfr      = self.volatility.loc[ :1500,                       :  self.trn_split_point].iloc[:,:-1]
            trn_volat_aft      = self.volatility.loc[ :1530, self.trn_split_point  :         self.val_st_dt]
            val_volat_dat      = self.volatility.loc[ :1530, self.val_st_dt        :        self.test_st_dt]
            test_volat_dat     = self.volatility.loc[ :1530, self.test_st_dt       :                       ]

            trn_freq_bfr       = self.freq_by_sin.loc[ :1500,                       :  self.trn_split_point].iloc[:,:-1]
            trn_freq_aft       = self.freq_by_sin.loc[ :1530, self.trn_split_point  :        self.val_st_dt]
            val_freq_dat       = self.freq_by_sin.loc[ :1530, self.val_st_dt        :       self.test_st_dt]
            test_freq_dat      = self.freq_by_sin.loc[ :1530, self.test_st_dt       :                      ]
        
            print("\nINFO: [_init_processing] DATASET SPLIT DATA RANGE is ...")
            print(f"[Momentum Feature :: Train] {trn_momentum_bfr.iloc[:, 0].name} ~ {trn_momentum_aft.iloc[:, -1].name}")
            print(f"[Momentum Feature ::   Val] {val_momentum_dat.iloc[:, 0].name} ~ {val_momentum_dat.iloc[:, -1].name}")
            print(f"[Momentum Feature ::  Test] {test_momentum_dat.iloc[:, 0].name} ~ {test_momentum_dat.iloc[:, -1].name}\n")

            print(f"[volatility Feature :: Train] {trn_volat_bfr.iloc[:, 0].name} ~ {trn_volat_aft.iloc[:, -1].name}")
            print(f"[volatility Feature ::   Val] {val_volat_dat.iloc[:, 0].name} ~ {val_volat_dat.iloc[:, -1].name}")
            print(f"[volatility Feature ::  Test] {test_volat_dat.iloc[:, 0].name} ~ {test_volat_dat.iloc[:, -1].name}\n")
            
            print(f"[freqency Feature :: Train] {trn_freq_bfr.iloc[:, 0].name} ~ {trn_freq_aft.iloc[:, -1].name}")
            print(f"[freqency Feature ::   Val] {val_freq_dat.iloc[:, 0].name} ~ {val_freq_dat.iloc[:, -1].name}")
            print(f"[freqency Feature ::  Test] {test_freq_dat.iloc[:, 0].name} ~ {test_freq_dat.iloc[:, -1].name}\n")
        elif self.feature_mode in ['scaled_mamv', 'scaled_mamv_kospi', 'scaled_mamv_kospi_ret', 'scaled_mamv_kospi_ret_5wr', 'scaled_mamv_kospi_ret_5wr_dxy_ret', 'scaled_hlcma_mv_kospi_ret_5wr']:
            fx_close_ma         = self.fx_close_dat.rolling(5).mean()
            momentum_dat        = self.fx_close_dat / self.fx_close_dat.shift(5) - 1
            volat_dat           = (self.fx_close_dat / self.fx_close_dat.shift(1) - 1).rolling(5).std()
            volat_grp_dat       = (self.fx_close_dat / self.fx_close_dat.shift(1) - 1).rolling(30).std()
            
            trn_cma_bfr         = fx_close_ma.loc[ :1500,                       :  self.trn_split_point].iloc[:,:-1]
            trn_cma_aft         = fx_close_ma.loc[ :1530, self.trn_split_point  :           self.val_st_dt]
            val_cma_dat         = fx_close_ma.loc[ :1530, self.val_st_dt        :          self.test_st_dt]
            test_cma_dat        = fx_close_ma.loc[ :1530, self.test_st_dt       :                         ]

            trn_momentum_bfr    = momentum_dat.loc[ :1500,                       :  self.trn_split_point].iloc[:,:-1]
            trn_momentum_aft    = momentum_dat.loc[ :1530, self.trn_split_point  :         self.val_st_dt]
            val_momentum_dat    = momentum_dat.loc[ :1530, self.val_st_dt        :        self.test_st_dt]
            test_momentum_dat   = momentum_dat.loc[ :1530, self.test_st_dt       :                       ]

            trn_volat_bfr       = volat_dat.loc[ :1500,                       :  self.trn_split_point].iloc[:,:-1]
            trn_volat_aft       = volat_dat.loc[ :1530, self.trn_split_point  :        self.val_st_dt]
            val_volat_dat       = volat_dat.loc[ :1530, self.val_st_dt        :       self.test_st_dt]
            test_volat_dat      = volat_dat.loc[ :1530, self.test_st_dt       :                      ]
            
            trn_volat_grp_bfr   = volat_grp_dat.loc[ :1500,                       :  self.trn_split_point].iloc[:,:-1]
            trn_volat_grp_aft   = volat_grp_dat.loc[ :1530, self.trn_split_point  :        self.val_st_dt]
            val_volat_grp_dat   = volat_grp_dat.loc[ :1530, self.val_st_dt        :       self.test_st_dt]
            test_volat_grp_dat  = volat_grp_dat.loc[ :1530, self.test_st_dt       :                      ]
            
            ### Feature 01 : Price 관련 (high, low 추가) ###  
            if 'hl' in self.feature_mode :
                fx_high_ma          = self.fx_high_dat.rolling(5).mean()
                trn_hma_bfr         = fx_high_ma.loc[ :1500,                       :     self.trn_split_point].iloc[:,:-1]
                trn_hma_aft         = fx_high_ma.loc[ :1530, self.trn_split_point  :           self.val_st_dt]
                val_hma_dat         = fx_high_ma.loc[ :1530, self.val_st_dt        :          self.test_st_dt]
                test_hma_dat        = fx_high_ma.loc[ :1530, self.test_st_dt       :                         ]
                
                fx_low_ma           = self.fx_low_dat.rolling(5).mean()
                trn_lma_bfr         = fx_low_ma.loc[ :1500,                       :     self.trn_split_point].iloc[:,:-1]
                trn_lma_aft         = fx_low_ma.loc[ :1530, self.trn_split_point  :           self.val_st_dt]
                val_lma_dat         = fx_low_ma.loc[ :1530, self.val_st_dt        :          self.test_st_dt]
                test_lma_dat        = fx_low_ma.loc[ :1530, self.test_st_dt       :                         ]

            ### Feature 04 : kospi 관련 ###  
            if 'kospi_ret' in self.feature_mode :
                kospi_ret       = self.kospi200 / self.kospi200.shift(5) - 1
                trn_ksp200_bfr  = kospi_ret.loc[ :1500,                       :     self.trn_split_point].iloc[:,:-1]
                trn_ksp200_aft  = kospi_ret.loc[ :1530, self.trn_split_point  :           self.val_st_dt]
                val_ksp200_dat  = kospi_ret.loc[ :1530, self.val_st_dt        :          self.test_st_dt]
                test_ksp200_dat = kospi_ret.loc[ :1530, self.test_st_dt       :                         ]           
            
            ### Feature 05 : 최근 5일 Range ###         
            if '5wr' in self.feature_mode :
                day5_min_price, day5_max_price  = self.fx_close_dat.min().rolling(5).min().shift(1), self.fx_close_dat.max().rolling(5).max().shift(1)
                day5_williams_R                 = ((day5_max_price - self.fx_close_dat) / (day5_max_price - day5_min_price))
                
                trn_5days_wr_bfr  = day5_williams_R.loc[ :1500,                       :     self.trn_split_point].iloc[:,:-1]
                trn_5days_wr_aft  = day5_williams_R.loc[ :1530, self.trn_split_point  :           self.val_st_dt]
                val_5days_wr_dat  = day5_williams_R.loc[ :1530, self.val_st_dt        :          self.test_st_dt]
                test_5days_wr_dat = day5_williams_R.loc[ :1530, self.test_st_dt       :                         ]   

            ### Feature 06 : 달러인덱스 ### 
            if 'dxy_ret' in self.feature_mode :
                dxy_5min_ret      = self.dxy_close_dat / self.dxy_close_dat.shift(5) - 1
                trn_dxy_bfr       = dxy_5min_ret.loc[ :1500,                       :     self.trn_split_point].iloc[:,:-1]
                trn_dxy_aft       = dxy_5min_ret.loc[ :1530, self.trn_split_point  :           self.val_st_dt]
                val_dxy_dat       = dxy_5min_ret.loc[ :1530, self.val_st_dt        :          self.test_st_dt]
                test_dxy_dat      = dxy_5min_ret.loc[ :1530, self.test_st_dt       :                         ]  
        elif self.feature_mode in ['ohlc_dxy_hlc']:
            
            trn_open_bfr         = self.fx_open_dat.loc[ :1500,                       :  self.trn_split_point].iloc[:,:-1]
            trn_open_aft         = self.fx_open_dat.loc[ :1530, self.trn_split_point  :           self.val_st_dt]
            val_open_dat         = self.fx_open_dat.loc[ :1530, self.val_st_dt        :          self.test_st_dt]
            test_open_dat        = self.fx_open_dat.loc[ :1530, self.test_st_dt       :                         ]
            
            trn_high_bfr         = self.fx_high_dat.loc[ :1500,                       :  self.trn_split_point].iloc[:,:-1]
            trn_high_aft         = self.fx_high_dat.loc[ :1530, self.trn_split_point  :           self.val_st_dt]
            val_high_dat         = self.fx_high_dat.loc[ :1530, self.val_st_dt        :          self.test_st_dt]
            test_high_dat        = self.fx_high_dat.loc[ :1530, self.test_st_dt       :                         ]
            
            trn_low_bfr          = self.fx_low_dat.loc[ :1500,                       :  self.trn_split_point].iloc[:,:-1]
            trn_low_aft          = self.fx_low_dat.loc[ :1530, self.trn_split_point  :           self.val_st_dt]
            val_low_dat          = self.fx_low_dat.loc[ :1530, self.val_st_dt        :          self.test_st_dt]
            test_low_dat         = self.fx_low_dat.loc[ :1530, self.test_st_dt       :                         ]
            
            trn_close_bfr        = self.fx_close_dat.loc[ :1500,                       :  self.trn_split_point].iloc[:,:-1]
            trn_close_aft        = self.fx_close_dat.loc[ :1530, self.trn_split_point  :           self.val_st_dt]
            val_close_dat        = self.fx_close_dat.loc[ :1530, self.val_st_dt        :          self.test_st_dt]
            test_close_dat       = self.fx_close_dat.loc[ :1530, self.test_st_dt       :                         ]
            
            trn_dxy_high_bfr     = self.dxy_high_dat.loc[ :1500,                       :  self.trn_split_point].iloc[:,:-1]
            trn_dxy_high_aft     = self.dxy_high_dat.loc[ :1530, self.trn_split_point  :           self.val_st_dt]
            val_dxy_high_dat     = self.dxy_high_dat.loc[ :1530, self.val_st_dt        :          self.test_st_dt]
            test_dxy_high_dat    = self.dxy_high_dat.loc[ :1530, self.test_st_dt       :                         ]
            
            trn_dxy_low_bfr      = self.dxy_low_dat.loc[ :1500,                       :  self.trn_split_point].iloc[:,:-1]
            trn_dxy_low_aft      = self.dxy_low_dat.loc[ :1530, self.trn_split_point  :           self.val_st_dt]
            val_dxy_low_dat      = self.dxy_low_dat.loc[ :1530, self.val_st_dt        :          self.test_st_dt]
            test_dxy_low_dat     = self.dxy_low_dat.loc[ :1530, self.test_st_dt       :                         ]
            
            trn_dxy_close_bfr    = self.dxy_close_dat.loc[ :1500,                       :  self.trn_split_point].iloc[:,:-1]
            trn_dxy_close_aft    = self.dxy_close_dat.loc[ :1530, self.trn_split_point  :           self.val_st_dt]
            val_dxy_close_dat    = self.dxy_close_dat.loc[ :1530, self.val_st_dt        :          self.test_st_dt]
            test_dxy_close_dat   = self.dxy_close_dat.loc[ :1530, self.test_st_dt       :                         ]
            
            volat_grp_dat       = (self.fx_close_dat / self.fx_close_dat.shift(1) - 1).rolling(30).std()
            trn_volat_grp_bfr   = volat_grp_dat.loc[ :1500,                       :  self.trn_split_point].iloc[:,:-1]
            trn_volat_grp_aft   = volat_grp_dat.loc[ :1530, self.trn_split_point  :        self.val_st_dt]
            val_volat_grp_dat   = volat_grp_dat.loc[ :1530, self.val_st_dt        :       self.test_st_dt]
            test_volat_grp_dat  = volat_grp_dat.loc[ :1530, self.test_st_dt       :                      ]
        elif self.feature_mode in ['ohlc_dxy_hlc_ks', 'padding_ohlc_dxy_hlc_ks']:
            
            trn_open_bfr         = self.fx_open_dat.loc[ :1500,                       :  self.trn_split_point].iloc[:,:-1]
            trn_open_aft         = self.fx_open_dat.loc[ :1530, self.trn_split_point  :           self.val_st_dt]
            val_open_dat         = self.fx_open_dat.loc[ :1530, self.val_st_dt        :          self.test_st_dt]
            test_open_dat        = self.fx_open_dat.loc[ :1530, self.test_st_dt       :                         ]
            
            trn_high_bfr         = self.fx_high_dat.loc[ :1500,                       :  self.trn_split_point].iloc[:,:-1]
            trn_high_aft         = self.fx_high_dat.loc[ :1530, self.trn_split_point  :           self.val_st_dt]
            val_high_dat         = self.fx_high_dat.loc[ :1530, self.val_st_dt        :          self.test_st_dt]
            test_high_dat        = self.fx_high_dat.loc[ :1530, self.test_st_dt       :                         ]
            
            trn_low_bfr          = self.fx_low_dat.loc[ :1500,                       :  self.trn_split_point].iloc[:,:-1]
            trn_low_aft          = self.fx_low_dat.loc[ :1530, self.trn_split_point  :           self.val_st_dt]
            val_low_dat          = self.fx_low_dat.loc[ :1530, self.val_st_dt        :          self.test_st_dt]
            test_low_dat         = self.fx_low_dat.loc[ :1530, self.test_st_dt       :                         ]
            
            trn_close_bfr        = self.fx_close_dat.loc[ :1500,                       :  self.trn_split_point].iloc[:,:-1]
            trn_close_aft        = self.fx_close_dat.loc[ :1530, self.trn_split_point  :           self.val_st_dt]
            val_close_dat        = self.fx_close_dat.loc[ :1530, self.val_st_dt        :          self.test_st_dt]
            test_close_dat       = self.fx_close_dat.loc[ :1530, self.test_st_dt       :                         ]
            
            trn_dxy_high_bfr     = self.dxy_high_dat.loc[ :1500,                       :  self.trn_split_point].iloc[:,:-1]
            trn_dxy_high_aft     = self.dxy_high_dat.loc[ :1530, self.trn_split_point  :           self.val_st_dt]
            val_dxy_high_dat     = self.dxy_high_dat.loc[ :1530, self.val_st_dt        :          self.test_st_dt]
            test_dxy_high_dat    = self.dxy_high_dat.loc[ :1530, self.test_st_dt       :                         ]
            
            trn_dxy_low_bfr      = self.dxy_low_dat.loc[ :1500,                       :  self.trn_split_point].iloc[:,:-1]
            trn_dxy_low_aft      = self.dxy_low_dat.loc[ :1530, self.trn_split_point  :           self.val_st_dt]
            val_dxy_low_dat      = self.dxy_low_dat.loc[ :1530, self.val_st_dt        :          self.test_st_dt]
            test_dxy_low_dat     = self.dxy_low_dat.loc[ :1530, self.test_st_dt       :                         ]
            
            trn_dxy_close_bfr    = self.dxy_close_dat.loc[ :1500,                       :  self.trn_split_point].iloc[:,:-1]
            trn_dxy_close_aft    = self.dxy_close_dat.loc[ :1530, self.trn_split_point  :           self.val_st_dt]
            val_dxy_close_dat    = self.dxy_close_dat.loc[ :1530, self.val_st_dt        :          self.test_st_dt]
            test_dxy_close_dat   = self.dxy_close_dat.loc[ :1530, self.test_st_dt       :                         ]
            
            trn_ks200_bfr        = self.kospi200_dat.loc[ :1500,                       :  self.trn_split_point].iloc[:,:-1]
            trn_ks200_aft        = self.kospi200_dat.loc[ :1530, self.trn_split_point  :           self.val_st_dt]
            val_ks200_dat        = self.kospi200_dat.loc[ :1530, self.val_st_dt        :          self.test_st_dt]
            test_ks200_dat       = self.kospi200_dat.loc[ :1530, self.test_st_dt       :                         ]
            
            volat_grp_dat        = (self.fx_close_dat / self.fx_close_dat.shift(1) - 1).rolling(30).std()
            trn_volat_grp_bfr    = volat_grp_dat.loc[ :1500,                       :  self.trn_split_point].iloc[:,:-1]
            trn_volat_grp_aft    = volat_grp_dat.loc[ :1530, self.trn_split_point  :        self.val_st_dt]
            val_volat_grp_dat    = volat_grp_dat.loc[ :1530, self.val_st_dt        :       self.test_st_dt]
            test_volat_grp_dat   = volat_grp_dat.loc[ :1530, self.test_st_dt       :                      ]
        elif self.feature_mode in ['ohlc_dxy_hlc_cnh_ks']:
            
            trn_open_bfr         = self.fx_open_dat.loc[ :1500,                       :  self.trn_split_point].iloc[:,:-1]
            trn_open_aft         = self.fx_open_dat.loc[ :1530, self.trn_split_point  :           self.val_st_dt]
            val_open_dat         = self.fx_open_dat.loc[ :1530, self.val_st_dt        :          self.test_st_dt]
            test_open_dat        = self.fx_open_dat.loc[ :1530, self.test_st_dt       :                         ]
            
            trn_high_bfr         = self.fx_high_dat.loc[ :1500,                       :  self.trn_split_point].iloc[:,:-1]
            trn_high_aft         = self.fx_high_dat.loc[ :1530, self.trn_split_point  :           self.val_st_dt]
            val_high_dat         = self.fx_high_dat.loc[ :1530, self.val_st_dt        :          self.test_st_dt]
            test_high_dat        = self.fx_high_dat.loc[ :1530, self.test_st_dt       :                         ]
            
            trn_low_bfr          = self.fx_low_dat.loc[ :1500,                       :  self.trn_split_point].iloc[:,:-1]
            trn_low_aft          = self.fx_low_dat.loc[ :1530, self.trn_split_point  :           self.val_st_dt]
            val_low_dat          = self.fx_low_dat.loc[ :1530, self.val_st_dt        :          self.test_st_dt]
            test_low_dat         = self.fx_low_dat.loc[ :1530, self.test_st_dt       :                         ]
            
            trn_close_bfr        = self.fx_close_dat.loc[ :1500,                       :  self.trn_split_point].iloc[:,:-1]
            trn_close_aft        = self.fx_close_dat.loc[ :1530, self.trn_split_point  :           self.val_st_dt]
            val_close_dat        = self.fx_close_dat.loc[ :1530, self.val_st_dt        :          self.test_st_dt]
            test_close_dat       = self.fx_close_dat.loc[ :1530, self.test_st_dt       :                         ]
            
            trn_dxy_high_bfr     = self.dxy_high_dat.loc[ :1500,                       :  self.trn_split_point].iloc[:,:-1]
            trn_dxy_high_aft     = self.dxy_high_dat.loc[ :1530, self.trn_split_point  :           self.val_st_dt]
            val_dxy_high_dat     = self.dxy_high_dat.loc[ :1530, self.val_st_dt        :          self.test_st_dt]
            test_dxy_high_dat    = self.dxy_high_dat.loc[ :1530, self.test_st_dt       :                         ]
            
            trn_dxy_low_bfr      = self.dxy_low_dat.loc[ :1500,                       :  self.trn_split_point].iloc[:,:-1]
            trn_dxy_low_aft      = self.dxy_low_dat.loc[ :1530, self.trn_split_point  :           self.val_st_dt]
            val_dxy_low_dat      = self.dxy_low_dat.loc[ :1530, self.val_st_dt        :          self.test_st_dt]
            test_dxy_low_dat     = self.dxy_low_dat.loc[ :1530, self.test_st_dt       :                         ]
            
            trn_dxy_close_bfr    = self.dxy_close_dat.loc[ :1500,                       :  self.trn_split_point].iloc[:,:-1]
            trn_dxy_close_aft    = self.dxy_close_dat.loc[ :1530, self.trn_split_point  :           self.val_st_dt]
            val_dxy_close_dat    = self.dxy_close_dat.loc[ :1530, self.val_st_dt        :          self.test_st_dt]
            test_dxy_close_dat   = self.dxy_close_dat.loc[ :1530, self.test_st_dt       :                         ]

            trn_cnh_mid_bfr      = self.cnh_mid_dat.loc[ :1500,                       :  self.trn_split_point].iloc[:,:-1]
            trn_cnh_mid_aft      = self.cnh_mid_dat.loc[ :1530, self.trn_split_point  :           self.val_st_dt]
            val_cnh_mid_dat      = self.cnh_mid_dat.loc[ :1530, self.val_st_dt        :          self.test_st_dt]
            test_cnh_mid_dat     = self.cnh_mid_dat.loc[ :1530, self.test_st_dt       :                         ]
            
            trn_ks200_bfr        = self.kospi200_dat.loc[ :1500,                       :  self.trn_split_point].iloc[:,:-1]
            trn_ks200_aft        = self.kospi200_dat.loc[ :1530, self.trn_split_point  :           self.val_st_dt]
            val_ks200_dat        = self.kospi200_dat.loc[ :1530, self.val_st_dt        :          self.test_st_dt]
            test_ks200_dat       = self.kospi200_dat.loc[ :1530, self.test_st_dt       :                         ]
            
            volat_grp_dat        = (self.fx_close_dat / self.fx_close_dat.shift(1) - 1).rolling(30).std()
            trn_volat_grp_bfr    = volat_grp_dat.loc[ :1500,                       :  self.trn_split_point].iloc[:,:-1]
            trn_volat_grp_aft    = volat_grp_dat.loc[ :1530, self.trn_split_point  :        self.val_st_dt]
            val_volat_grp_dat    = volat_grp_dat.loc[ :1530, self.val_st_dt        :       self.test_st_dt]
            test_volat_grp_dat   = volat_grp_dat.loc[ :1530, self.test_st_dt       :                      ]

        ##### (1-2) Make Data-Sets #####
        trn_bfr_input_sets, trn_aft_input_sets, val_input_sets, test_input_sets = None, None, None, None
        if self.feature_mode in ['c'] :                                                 # only Close 
            trn_bfr_input_sets, trn_aft_input_sets, test_input_sets = trn_dat_bfr, trn_dat_aft, test_dat                            
        elif self.feature_mode in ['cub']:                                              # Close, bfr_u, bfr_b
            trn_bfr_input_sets = (trn_dat_bfr, trn_iu_bfr , trn_ib_bfr)
            trn_aft_input_sets = (trn_dat_aft, trn_iu_aft , trn_ib_aft)
            test_input_sets    = (test_dat   , test_iu_dat, test_ib_dat)
        elif self.feature_mode in ['cfreq']:                                            # Close, Freq(sin) 
            trn_bfr_input_sets = (trn_dat_bfr, trn_freq_bfr)
            trn_aft_input_sets = (trn_dat_aft, trn_freq_aft)
            test_input_sets    = (test_dat   , test_freq_dat)
        elif self.feature_mode in ['mvf', 'scaled_mvf']:                                # Momentum, Volatility, Frequency(sin)
            trn_bfr_input_sets = (trn_momentum_bfr, trn_volat_bfr, trn_freq_bfr)
            trn_aft_input_sets = (trn_momentum_aft, trn_volat_aft, trn_freq_aft)
            val_input_sets     = (val_momentum_dat, val_volat_dat, val_freq_dat)
            test_input_sets    = (test_momentum_dat,test_volat_dat, test_freq_dat)
        elif self.feature_mode in ['cmvf', 'scaled_cmvf']:                              # Close, Momentum, Volatility, Frequency 
            trn_bfr_input_sets = (trn_dat_bfr, trn_momentum_bfr, trn_volat_bfr, trn_freq_bfr)
            trn_aft_input_sets = (trn_dat_aft, trn_momentum_aft, trn_volat_aft, trn_freq_aft)
            val_input_sets     = (val_dat    , val_momentum_dat, val_volat_dat, val_freq_dat)
            test_input_sets    = (test_dat   , test_momentum_dat,test_volat_dat, test_freq_dat)
        elif self.feature_mode in ['scaled_mamv']:                                      # Close_Moving_Avg, Momentum, Volatility
            trn_bfr_input_sets = (trn_cma_bfr,  trn_momentum_bfr, trn_volat_bfr, trn_volat_grp_bfr)
            trn_aft_input_sets = (trn_cma_aft,  trn_momentum_aft, trn_volat_aft, trn_volat_grp_aft)
            val_input_sets     = (val_cma_dat,  val_momentum_dat, val_volat_dat, val_volat_grp_bfr)
            test_input_sets    = (test_cma_dat, test_momentum_dat,test_volat_dat, test_volat_grp_bfr)
        elif self.feature_mode in ['scaled_mamv_kospi', 'scaled_mamv_kospi_ret']:       # Close_Moving_Avg, Momentum, Volatility, KS200
            trn_bfr_input_sets = (trn_cma_bfr,  trn_momentum_bfr, trn_volat_bfr, trn_ksp200_bfr, trn_volat_grp_bfr)
            trn_aft_input_sets = (trn_cma_aft,  trn_momentum_aft, trn_volat_aft, trn_ksp200_aft, trn_volat_grp_aft)
            val_input_sets     = (val_cma_dat,  val_momentum_dat, val_volat_dat, val_ksp200_dat, val_volat_grp_dat)
            test_input_sets    = (test_cma_dat, test_momentum_dat,test_volat_dat, test_ksp200_dat, test_volat_grp_dat)
        elif self.feature_mode in ['scaled_mamv_kospi_ret_5wr']:
            trn_bfr_input_sets = (trn_cma_bfr,  trn_momentum_bfr, trn_volat_bfr, trn_ksp200_bfr, trn_5days_wr_bfr, trn_volat_grp_bfr)
            trn_aft_input_sets = (trn_cma_aft,  trn_momentum_aft, trn_volat_aft, trn_ksp200_aft, trn_5days_wr_aft, trn_volat_grp_aft)
            val_input_sets     = (val_cma_dat,  val_momentum_dat, val_volat_dat, val_ksp200_dat, val_5days_wr_dat ,val_volat_grp_dat)
            test_input_sets    = (test_cma_dat, test_momentum_dat,test_volat_dat, test_ksp200_dat, test_5days_wr_dat, test_volat_grp_dat)
        elif self.feature_mode in ['scaled_mamv_kospi_ret_5wr_dxy_ret']:
            trn_bfr_input_sets = (trn_cma_bfr ,  trn_momentum_bfr , trn_volat_bfr , trn_ksp200_bfr , trn_5days_wr_bfr , trn_dxy_bfr , trn_volat_grp_bfr)
            trn_aft_input_sets = (trn_cma_aft ,  trn_momentum_aft , trn_volat_aft , trn_ksp200_aft , trn_5days_wr_aft , trn_dxy_aft , trn_volat_grp_aft)
            val_input_sets     = (val_cma_dat ,  val_momentum_dat , val_volat_dat , val_ksp200_dat , val_5days_wr_dat , val_dxy_dat , val_volat_grp_dat)
            test_input_sets    = (test_cma_dat,  test_momentum_dat, test_volat_dat, test_ksp200_dat, test_5days_wr_dat, test_dxy_dat, test_volat_grp_dat)
        elif self.feature_mode in ['scaled_hlcma_mv_kospi_ret_5wr']:
            trn_bfr_input_sets = (trn_cma_bfr , trn_hma_bfr , trn_lma_bfr , trn_momentum_bfr , trn_volat_bfr , trn_ksp200_bfr , trn_5days_wr_bfr , trn_volat_grp_bfr )
            trn_aft_input_sets = (trn_cma_aft , trn_hma_aft , trn_lma_aft , trn_momentum_aft , trn_volat_aft , trn_ksp200_aft , trn_5days_wr_aft , trn_volat_grp_aft )
            val_input_sets     = (val_cma_dat , val_hma_dat , val_lma_dat , val_momentum_dat , val_volat_dat , val_ksp200_dat , val_5days_wr_dat , val_volat_grp_dat )
            test_input_sets    = (test_cma_dat, test_hma_dat, test_lma_dat, test_momentum_dat, test_volat_dat, test_ksp200_dat, test_5days_wr_dat, test_volat_grp_dat)
        elif self.feature_mode in ['ohlc_dxy_hlc']:
            trn_bfr_input_sets = (trn_open_bfr , trn_high_bfr , trn_low_bfr , trn_close_bfr , trn_dxy_high_bfr , trn_dxy_low_bfr , trn_dxy_close_bfr , trn_volat_grp_bfr )
            trn_aft_input_sets = (trn_open_aft , trn_high_aft , trn_low_aft , trn_close_aft , trn_dxy_high_aft , trn_dxy_low_aft , trn_dxy_close_aft , trn_volat_grp_aft )
            val_input_sets     = (val_open_dat , val_high_dat , val_low_dat , val_close_dat , val_dxy_high_dat , val_dxy_low_dat , val_dxy_close_dat , val_volat_grp_dat )
            test_input_sets    = (test_open_dat, test_high_dat, test_low_dat, test_close_dat, test_dxy_high_dat, test_dxy_low_dat, test_dxy_close_dat, test_volat_grp_dat)
        elif self.feature_mode in ['ohlc_dxy_hlc_ks', 'padding_ohlc_dxy_hlc_ks']:
            trn_bfr_input_sets = (trn_open_bfr , trn_high_bfr , trn_low_bfr , trn_close_bfr , trn_dxy_high_bfr , trn_dxy_low_bfr , trn_dxy_close_bfr , trn_ks200_bfr ,  trn_volat_grp_bfr )
            trn_aft_input_sets = (trn_open_aft , trn_high_aft , trn_low_aft , trn_close_aft , trn_dxy_high_aft , trn_dxy_low_aft , trn_dxy_close_aft , trn_ks200_aft ,  trn_volat_grp_aft )
            val_input_sets     = (val_open_dat , val_high_dat , val_low_dat , val_close_dat , val_dxy_high_dat , val_dxy_low_dat , val_dxy_close_dat , val_ks200_dat ,  val_volat_grp_dat )
            test_input_sets    = (test_open_dat, test_high_dat, test_low_dat, test_close_dat, test_dxy_high_dat, test_dxy_low_dat, test_dxy_close_dat, test_ks200_dat,  test_volat_grp_dat)
        elif self.feature_mode in ['ohlc_dxy_hlc_cnh_ks']:
            trn_bfr_input_sets = (trn_open_bfr , trn_high_bfr , trn_low_bfr , trn_close_bfr , trn_dxy_high_bfr , trn_dxy_low_bfr , trn_dxy_close_bfr , trn_cnh_mid_bfr, trn_ks200_bfr ,  trn_volat_grp_bfr )
            trn_aft_input_sets = (trn_open_aft , trn_high_aft , trn_low_aft , trn_close_aft , trn_dxy_high_aft , trn_dxy_low_aft , trn_dxy_close_aft , trn_cnh_mid_aft, trn_ks200_aft ,  trn_volat_grp_aft )
            val_input_sets     = (val_open_dat , val_high_dat , val_low_dat , val_close_dat , val_dxy_high_dat , val_dxy_low_dat , val_dxy_close_dat , val_cnh_mid_dat, val_ks200_dat ,  val_volat_grp_dat )
            test_input_sets    = (test_open_dat, test_high_dat, test_low_dat, test_close_dat, test_dxy_high_dat, test_dxy_low_dat, test_dxy_close_dat, test_cnh_mid_dat, test_ks200_dat,  test_volat_grp_dat)
            
        ##### (*1-3) Statistics for scaling ##### 
        if (self.scaling) and ('scaled' in self.feature_mode):

            self.sta_values = dict()         
            
            trn_cma_bfr_tmp      = np.squeeze(np.array(trn_cma_bfr).reshape(-1, 1))   
            trn_cma_aft_tmp      = np.squeeze(np.array(trn_cma_aft).reshape(-1, 1))   
            trn_cma              = np.concatenate([trn_cma_bfr_tmp, trn_cma_aft_tmp])
            
            trn_momentum_bfr_tmp = np.squeeze(np.array(trn_momentum_bfr).reshape(-1, 1))   
            trn_momentum_aft_tmp = np.squeeze(np.array(trn_momentum_aft).reshape(-1, 1))   
            trn_momentum         = np.concatenate([trn_momentum_bfr_tmp, trn_momentum_aft_tmp])

            trn_volat_bfr_tmp    = np.squeeze(np.array(trn_volat_bfr).reshape(-1, 1))   
            trn_volat_aft_tmp    = np.squeeze(np.array(trn_volat_aft).reshape(-1, 1))  
            trn_volat            = np.concatenate([trn_volat_bfr_tmp, trn_volat_aft_tmp]) 
            
            trn_cma              = trn_cma[~np.isnan(trn_cma)]
            trn_momentum         = trn_momentum[~np.isnan(trn_momentum)]
            trn_volat            = trn_volat[~np.isnan(trn_volat)]
            
            self.sta_values['cma']        = (trn_cma.mean(), trn_cma.std())
            
            if 'hl' in self.feature_mode:
                
                trn_hma_bfr_tmp      = np.squeeze(np.array(trn_hma_bfr).reshape(-1, 1))   
                trn_hma_aft_tmp      = np.squeeze(np.array(trn_hma_aft).reshape(-1, 1))   
                trn_hma              = np.concatenate([trn_hma_bfr_tmp, trn_hma_aft_tmp])
                
                trn_lma_bfr_tmp      = np.squeeze(np.array(trn_lma_bfr).reshape(-1, 1))   
                trn_lma_aft_tmp      = np.squeeze(np.array(trn_lma_aft).reshape(-1, 1))   
                trn_lma              = np.concatenate([trn_lma_bfr_tmp, trn_lma_aft_tmp])
                
                trn_hma, trn_lma     = trn_hma[~np.isnan(trn_hma)], trn_lma[~np.isnan(trn_lma)]
                
                self.sta_values['hma']        = (trn_hma.mean(), trn_hma.std())
                self.sta_values['lma']        = (trn_lma.mean(), trn_lma.std())
                
            self.sta_values['momentum']   = (trn_momentum.mean(), trn_momentum.std())
            self.sta_values['volatility'] = (trn_volat.mean(), trn_volat.std())
            
            if 'kospi' in self.feature_mode:
                trn_ksp200_bfr_tmp = np.squeeze(np.array(trn_ksp200_bfr).reshape(-1, 1))   
                trn_ksp200_aft_tmp = np.squeeze(np.array(trn_ksp200_aft).reshape(-1, 1))   
                trn_ksp200         = np.concatenate([trn_ksp200_bfr_tmp, trn_ksp200_aft_tmp])
                
                trn_ksp200         = trn_ksp200[~np.isnan(trn_ksp200)]
                
                self.sta_values['ks200']  = trn_ksp200.mean(), trn_ksp200.std()
                
            if '5wr' in self.feature_mode:
                trn_5days_wr_bfr_tmp = np.squeeze(np.array(trn_5days_wr_bfr).reshape(-1, 1))   
                trn_5days_wr_aft_tmp = np.squeeze(np.array(trn_5days_wr_aft).reshape(-1, 1))   
                trn_5days_wr         = np.concatenate([trn_5days_wr_bfr_tmp, trn_5days_wr_aft_tmp])
                
                trn_5days_wr         = trn_5days_wr[~np.isnan(trn_5days_wr)]
                
                self.sta_values['5wr']  = trn_5days_wr.mean(), trn_5days_wr.std()
                
            if 'dxy' in self.feature_mode:
                trn_dxy_bfr_tmp = np.squeeze(np.array(trn_dxy_bfr).reshape(-1, 1))   
                trn_dxy_aft_tmp = np.squeeze(np.array(trn_dxy_aft).reshape(-1, 1))   
                trn_dxy         = np.concatenate([trn_dxy_bfr_tmp, trn_dxy_aft_tmp])
                
                trn_dxy         = trn_dxy[~np.isnan(trn_dxy)]
                
                self.sta_values['dxy']  = trn_dxy.mean(), trn_dxy.std()

        # feature_names = list(self.sta_values.keys())
        feature_names = ['test_usdkrw_open', 'test_usdkrw_high', 'test_usdkrw_Low', 'test_usdkrw_close', 'test_dxy_high', 'test_dxy_low', 'test_dxy_close', 'test_ks200']
        for i, key in enumerate(feature_names):
            feature_df = test_input_sets[i]
            feature_df.to_csv(f'./Datas_for_download/{key}_reshape.csv')   
        
        return trn_bfr_input_sets, trn_aft_input_sets, val_input_sets, test_input_sets

    def _data_grouping(self, key = None):
        trn_volat_30min_by_sets, val_volat_30min_by_sets, test_volat_30min_by_sets = self.train_x_npy[:,-1, -1], self.val_x_npy[:,-1, -1], self.test_x_npy[:,-1, -1]
        
        ##### Train Data Sets Grouping #####
        trn_volat_10per, trn_volat_20per, trn_volat_30per, trn_volat_40per, trn_volat_50per = np.nanpercentile(self.train_x_npy[:,-1, -1], 10), np.nanpercentile(self.train_x_npy[:,-1, -1], 20), np.nanpercentile(self.train_x_npy[:,-1, -1], 30), np.nanpercentile(self.train_x_npy[:,-1, -1], 40), np.nanpercentile(self.train_x_npy[:,-1, -1], 50)
        trn_volat_60per, trn_volat_70per, trn_volat_80per, trn_volat_90per                  = np.nanpercentile(self.train_x_npy[:,-1, -1], 60), np.nanpercentile(self.train_x_npy[:,-1, -1], 70), np.nanpercentile(self.train_x_npy[:,-1, -1], 80), np.nanpercentile(self.train_x_npy[:,-1, -1], 90)    
        
        trn_g1_idxs  = np.where(trn_volat_30min_by_sets < trn_volat_10per)[0]
        trn_g2_idxs  = np.where((trn_volat_10per <= trn_volat_30min_by_sets) & ( trn_volat_30min_by_sets < trn_volat_20per))[0]
        trn_g3_idxs  = np.where((trn_volat_20per <= trn_volat_30min_by_sets) & ( trn_volat_30min_by_sets < trn_volat_30per))[0]
        trn_g4_idxs  = np.where((trn_volat_30per <= trn_volat_30min_by_sets) & ( trn_volat_30min_by_sets < trn_volat_40per))[0]
        trn_g5_idxs  = np.where((trn_volat_40per <= trn_volat_30min_by_sets) & ( trn_volat_30min_by_sets < trn_volat_50per))[0]
        trn_g6_idxs  = np.where((trn_volat_50per <= trn_volat_30min_by_sets) & ( trn_volat_30min_by_sets < trn_volat_60per))[0]
        trn_g7_idxs  = np.where((trn_volat_60per <= trn_volat_30min_by_sets) & ( trn_volat_30min_by_sets < trn_volat_70per))[0]
        trn_g8_idxs  = np.where((trn_volat_70per <= trn_volat_30min_by_sets) & ( trn_volat_30min_by_sets < trn_volat_80per))[0]
        trn_g9_idxs  = np.where((trn_volat_80per <= trn_volat_30min_by_sets) & ( trn_volat_30min_by_sets < trn_volat_90per))[0]
        trn_g10_idxs = np.where((trn_volat_90per <= trn_volat_30min_by_sets))[0]
        
        self.train_x_npy[trn_g1_idxs, -1, -1], self.train_x_npy[trn_g2_idxs, -1, -1], self.train_x_npy[trn_g3_idxs, -1, -1], self.train_x_npy[trn_g4_idxs, -1, -1], self.train_x_npy[trn_g5_idxs, -1, -1] = 1, 2, 3, 4, 5
        self.train_x_npy[trn_g6_idxs, -1, -1], self.train_x_npy[trn_g7_idxs, -1, -1], self.train_x_npy[trn_g8_idxs, -1, -1], self.train_x_npy[trn_g9_idxs, -1, -1], self.train_x_npy[trn_g10_idxs, -1, -1] = 6, 7, 8, 9, 10

        
        ##### Val Data Sets Grouping #####
        val_volat_10per, val_volat_20per, val_volat_30per, val_volat_40per, val_volat_50per = np.nanpercentile(self.val_x_npy[:,-1, -1], 10), np.nanpercentile(self.val_x_npy[:,-1, -1], 20), np.nanpercentile(self.val_x_npy[:,-1, -1], 30), np.nanpercentile(self.val_x_npy[:,-1, -1], 40), np.nanpercentile(self.val_x_npy[:,-1, -1], 50)
        val_volat_60per, val_volat_70per, val_volat_80per, val_volat_90per                  = np.nanpercentile(self.val_x_npy[:,-1, -1], 60), np.nanpercentile(self.val_x_npy[:,-1, -1], 70), np.nanpercentile(self.val_x_npy[:,-1, -1], 80), np.nanpercentile(self.val_x_npy[:,-1, -1], 90)    
        
        val_g1_idxs  = np.where(val_volat_30min_by_sets < val_volat_10per)[0]
        val_g2_idxs  = np.where((val_volat_10per <= val_volat_30min_by_sets) & ( val_volat_30min_by_sets < val_volat_20per))[0]
        val_g3_idxs  = np.where((val_volat_20per <= val_volat_30min_by_sets) & ( val_volat_30min_by_sets < val_volat_30per))[0]
        val_g4_idxs  = np.where((val_volat_30per <= val_volat_30min_by_sets) & ( val_volat_30min_by_sets < val_volat_40per))[0]
        val_g5_idxs  = np.where((val_volat_40per <= val_volat_30min_by_sets) & ( val_volat_30min_by_sets < val_volat_50per))[0]
        val_g6_idxs  = np.where((val_volat_50per <= val_volat_30min_by_sets) & ( val_volat_30min_by_sets < val_volat_60per))[0]
        val_g7_idxs  = np.where((val_volat_60per <= val_volat_30min_by_sets) & ( val_volat_30min_by_sets < val_volat_70per))[0]
        val_g8_idxs  = np.where((val_volat_70per <= val_volat_30min_by_sets) & ( val_volat_30min_by_sets < val_volat_80per))[0]
        val_g9_idxs  = np.where((val_volat_80per <= val_volat_30min_by_sets) & ( val_volat_30min_by_sets < val_volat_90per))[0]
        val_g10_idxs = np.where((val_volat_90per <= val_volat_30min_by_sets))[0]
        
        self.val_x_npy[val_g1_idxs, -1, -1], self.val_x_npy[val_g2_idxs, -1, -1], self.val_x_npy[val_g3_idxs, -1, -1], self.val_x_npy[val_g4_idxs, -1, -1], self.val_x_npy[val_g5_idxs, -1, -1] = 1, 2, 3, 4, 5
        self.val_x_npy[val_g6_idxs, -1, -1], self.val_x_npy[val_g7_idxs, -1, -1], self.val_x_npy[val_g8_idxs, -1, -1], self.val_x_npy[val_g9_idxs, -1, -1], self.val_x_npy[val_g10_idxs, -1, -1] = 6, 7, 8, 9, 10
        
        ##### Test Data Sets Grouping #####
        test_volat_10per, test_volat_20per, test_volat_30per, test_volat_40per, test_volat_50per = np.nanpercentile(self.test_x_npy[:,-1, -1], 10), np.nanpercentile(self.test_x_npy[:,-1, -1], 20), np.nanpercentile(self.test_x_npy[:,-1, -1], 30), np.nanpercentile(self.test_x_npy[:,-1, -1], 40), np.nanpercentile(self.test_x_npy[:,-1, -1], 50)
        test_volat_60per, test_volat_70per, test_volat_80per, test_volat_90per                   = np.nanpercentile(self.test_x_npy[:,-1, -1], 60), np.nanpercentile(self.test_x_npy[:,-1, -1], 70), np.nanpercentile(self.test_x_npy[:,-1, -1], 80), np.nanpercentile(self.test_x_npy[:,-1, -1], 90)    
        
        
        test_g1_idxs  = np.where(test_volat_30min_by_sets < test_volat_10per)[0]
        test_g2_idxs  = np.where((test_volat_10per <= test_volat_30min_by_sets) & ( test_volat_30min_by_sets < test_volat_20per))[0]
        test_g3_idxs  = np.where((test_volat_20per <= test_volat_30min_by_sets) & ( test_volat_30min_by_sets < test_volat_30per))[0]
        test_g4_idxs  = np.where((test_volat_30per <= test_volat_30min_by_sets) & ( test_volat_30min_by_sets < test_volat_40per))[0]
        test_g5_idxs  = np.where((test_volat_40per <= test_volat_30min_by_sets) & ( test_volat_30min_by_sets < test_volat_50per))[0]
        test_g6_idxs  = np.where((test_volat_50per <= test_volat_30min_by_sets) & ( test_volat_30min_by_sets < test_volat_60per))[0]
        test_g7_idxs  = np.where((test_volat_60per <= test_volat_30min_by_sets) & ( test_volat_30min_by_sets < test_volat_70per))[0]
        test_g8_idxs  = np.where((test_volat_70per <= test_volat_30min_by_sets) & ( test_volat_30min_by_sets < test_volat_80per))[0]
        test_g9_idxs  = np.where((test_volat_80per <= test_volat_30min_by_sets) & ( test_volat_30min_by_sets < test_volat_90per))[0]
        test_g10_idxs = np.where((test_volat_90per <= test_volat_30min_by_sets))[0]
        
        self.test_x_npy[test_g1_idxs, -1, -1], self.test_x_npy[test_g2_idxs, -1, -1], self.test_x_npy[test_g3_idxs, -1, -1], self.test_x_npy[test_g4_idxs, -1, -1], self.test_x_npy[test_g5_idxs, -1, -1] = 1, 2, 3, 4, 5
        self.test_x_npy[test_g6_idxs, -1, -1], self.test_x_npy[test_g7_idxs, -1, -1], self.test_x_npy[test_g8_idxs, -1, -1], self.test_x_npy[test_g9_idxs, -1, -1], self.test_x_npy[test_g10_idxs, -1, -1] = 6, 7, 8, 9, 10
        
    def _base_arrange_data(self, input_dat, target_dat, indexer=None, m_factor = None):
        if self.feature_mode in ['c']:
            input_data_npy  = self._convert_frame_to_numpy(data = input_dat , seq_len = self.time_interval, timestep = 1)
            target_data_npy = self._convert_frame_to_numpy(data = target_dat, seq_len = self.time_interval, timestep = 1, type='target')
            target_data_npy = target_data_npy[:, -1, :]

            input_data    = np.concatenate([input_data_npy[i, :, j] for j in range(input_data_npy.shape[2])
                                        for i in range(input_data_npy.shape[0])])

            input_data    = input_data.reshape(-1, self.time_interval)
            target_data   = target_data_npy.transpose().reshape(-1)
        
        elif self.feature_mode in ['cub']:
            input_close_data_npy = self._convert_frame_to_numpy(data = input_dat[0] , seq_len = self.time_interval, timestep = 1)
            input_pre_u_data_npy = self._convert_frame_to_numpy(data = input_dat[1] , seq_len = self.time_interval, timestep = 1)
            input_pre_b_data_npy = self._convert_frame_to_numpy(data = input_dat[2] , seq_len = self.time_interval, timestep = 1)
            target_data_npy      = self._convert_frame_to_numpy(data = target_dat, seq_len = self.time_interval, timestep = 1, type='target')
            target_data_npy      = target_data_npy[:, -1, :]

            input_close_data    = np.concatenate([input_close_data_npy[i, :, j] for j in range(input_close_data_npy.shape[2])
                                                for i in range(input_close_data_npy.shape[0])]).reshape(-1, self.time_interval)
            input_pre_u_data    = np.concatenate([input_pre_u_data_npy[i, :, j] for j in range(input_pre_u_data_npy.shape[2])
                                                for i in range(input_pre_u_data_npy.shape[0])]).reshape(-1, self.time_interval)
            input_pre_b_data    = np.concatenate([input_pre_b_data_npy[i, :, j] for j in range(input_pre_b_data_npy.shape[2])
                                                for i in range(input_pre_b_data_npy.shape[0])]).reshape(-1, self.time_interval)
            
            input_close_data, input_pre_u_data, input_pre_b_data =  np.expand_dims(input_close_data, 2), np.expand_dims(input_pre_u_data, 2), np.expand_dims(input_pre_b_data, 2)
            input_data                = np.concatenate([input_close_data, input_pre_u_data, input_pre_b_data], axis=2)
            target_data               = target_data_npy.transpose().reshape(-1)

        elif self.feature_mode in ['cfreq']:
            input_close_data_npy    = self._convert_frame_to_numpy(data = input_dat[0] , seq_len = self.time_interval, timestep = 1)
            input_freq_data_npy     = self._convert_frame_to_numpy(data = input_dat[1] , seq_len = self.time_interval, timestep = 1)
            target_data_npy         = self._convert_frame_to_numpy(data = target_dat, seq_len = self.time_interval, timestep = 1, type='target')
            target_data_npy         = target_data_npy[:, -1, :]

            input_close_data        = np.concatenate([input_close_data_npy[i, :, j] for j in range(input_close_data_npy.shape[2])
                                                    for i in range(input_close_data_npy.shape[0])]).reshape(-1, self.time_interval)
            input_freq_data         = np.concatenate([input_freq_data_npy[i, :, j] for j in range(input_freq_data_npy.shape[2])
                                                    for i in range(input_freq_data_npy.shape[0])]).reshape(-1, self.time_interval)
           
            input_close_data, input_freq_data = np.expand_dims(input_close_data, 2), np.expand_dims(input_freq_data, 2)
            input_data                        = np.concatenate([input_close_data, input_freq_data], axis=2)
            target_data                       = target_data_npy.transpose().reshape(-1)
        
        elif self.feature_mode in ['mvf', 'scaled_mvf']:
            input_moment_data_npy   = self._convert_frame_to_numpy(data = input_dat[0] , seq_len = self.time_interval, timestep = 1)
            input_volat_data_npy    = self._convert_frame_to_numpy(data = input_dat[1] , seq_len = self.time_interval, timestep = 1)
            input_freq_data_npy     = self._convert_frame_to_numpy(data = input_dat[2] , seq_len = self.time_interval, timestep = 1)
            target_data_npy         = self._convert_frame_to_numpy(data = target_dat, seq_len = self.time_interval, timestep = 1, type='target')
            target_data_npy         = target_data_npy[:, -1, :]

            input_moment_data       = np.concatenate([input_moment_data_npy[i, :, j] for j in range(input_moment_data_npy.shape[2])
                                                    for i in range(input_moment_data_npy.shape[0])]).reshape(-1, self.time_interval)
            input_volat_data        = np.concatenate([input_volat_data_npy[i, :, j] for j in range(input_volat_data_npy.shape[2])
                                                    for i in range(input_volat_data_npy.shape[0])]).reshape(-1, self.time_interval)
            input_freq_data         = np.concatenate([input_freq_data_npy[i, :, j] for j in range(input_freq_data_npy.shape[2])
                                                    for i in range(input_freq_data_npy.shape[0])]).reshape(-1, self.time_interval)
           
            input_moment_data, input_volat_data, input_freq_data = np.expand_dims(input_moment_data, 2), np.expand_dims(input_volat_data, 2), np.expand_dims(input_freq_data, 2)
            input_data                                           = np.concatenate([input_moment_data, input_volat_data, input_freq_data], axis=2)
            target_data                                          = target_data_npy.transpose().reshape(-1)
        
        elif self.feature_mode in ['cmvf', 'scaled_cmvf']:
            input_close_data_npy    = self._convert_frame_to_numpy(data = input_dat[0] , seq_len = self.time_interval, timestep = 1)
            input_moment_data_npy   = self._convert_frame_to_numpy(data = input_dat[1] , seq_len = self.time_interval, timestep = 1)
            input_volat_data_npy    = self._convert_frame_to_numpy(data = input_dat[2] , seq_len = self.time_interval, timestep = 1)
            input_freq_data_npy     = self._convert_frame_to_numpy(data = input_dat[3] , seq_len = self.time_interval, timestep = 1)
            target_data_npy         = self._convert_frame_to_numpy(data = target_dat, seq_len = self.time_interval, timestep = 1, type='target')
            target_data_npy         = target_data_npy[:, -1, :]

            input_close_data        = np.concatenate([input_close_data_npy[i, :, j] for j in range(input_close_data_npy.shape[2])
                                                    for i in range(input_close_data_npy.shape[0])]).reshape(-1, self.time_interval)
            input_moment_data       = np.concatenate([input_moment_data_npy[i, :, j] for j in range(input_moment_data_npy.shape[2])
                                                    for i in range(input_moment_data_npy.shape[0])]).reshape(-1, self.time_interval)
            input_volat_data        = np.concatenate([input_volat_data_npy[i, :, j] for j in range(input_volat_data_npy.shape[2])
                                                    for i in range(input_volat_data_npy.shape[0])]).reshape(-1, self.time_interval)
            input_freq_data         = np.concatenate([input_freq_data_npy[i, :, j] for j in range(input_freq_data_npy.shape[2])
                                                    for i in range(input_freq_data_npy.shape[0])]).reshape(-1, self.time_interval)

            input_close_data, input_moment_data, input_volat_data, input_freq_data = np.expand_dims(input_close_data, 2), np.expand_dims(input_moment_data, 2), np.expand_dims(input_volat_data, 2), np.expand_dims(input_freq_data, 2)
            input_data                                           = np.concatenate([input_close_data, input_moment_data, input_volat_data, input_freq_data], axis=2)
            target_data                                          = target_data_npy.transpose().reshape(-1)
        
        elif self.feature_mode in ['mamv', 'scaled_mamv']:
            input_close_data_npy    = self._convert_frame_to_numpy(data = input_dat[0] , seq_len = self.time_interval, timestep = 1)
            input_moment_data_npy   = self._convert_frame_to_numpy(data = input_dat[1] , seq_len = self.time_interval, timestep = 1)
            input_volat_data_npy    = self._convert_frame_to_numpy(data = input_dat[2] , seq_len = self.time_interval, timestep = 1)
            input_volat_group_npy   = self._convert_frame_to_numpy(data = input_dat[4] , seq_len = self.time_interval, timestep = 1)
            target_data_npy         = self._convert_frame_to_numpy(data = target_dat, seq_len = self.time_interval, timestep = 1, type='target')
            target_data_npy         = target_data_npy[:, -1, :]

            input_close_data        = np.concatenate([input_close_data_npy[i, :, j] for j in range(input_close_data_npy.shape[2])
                                                    for i in range(input_close_data_npy.shape[0])]).reshape(-1, self.time_interval)
            input_moment_data       = np.concatenate([input_moment_data_npy[i, :, j] for j in range(input_moment_data_npy.shape[2])
                                                    for i in range(input_moment_data_npy.shape[0])]).reshape(-1, self.time_interval)
            input_volat_data        = np.concatenate([input_volat_data_npy[i, :, j] for j in range(input_volat_data_npy.shape[2])
                                                    for i in range(input_volat_data_npy.shape[0])]).reshape(-1, self.time_interval)
            input_volat_grp         = np.concatenate([input_volat_group_npy[i, :, j] for j in range(input_volat_group_npy.shape[2])
                                                    for i in range(input_volat_group_npy.shape[0])]).reshape(-1, self.time_interval)

            input_close_data, input_moment_data, input_volat_data= np.expand_dims(input_close_data, 2), np.expand_dims(input_moment_data, 2), np.expand_dims(input_volat_data, 2)
            input_data                                           = np.concatenate([input_close_data, input_moment_data, input_volat_data], axis=2)
            target_data                                          = target_data_npy.transpose().reshape(-1)

        elif self.feature_mode in ['scaled_mamv_kospi', 'scaled_mamv_kospi_ret']:
            input_close_data_npy    = self._convert_frame_to_numpy(data = input_dat[0] , seq_len = self.time_interval, timestep = 1)
            input_moment_data_npy   = self._convert_frame_to_numpy(data = input_dat[1] , seq_len = self.time_interval, timestep = 1)
            input_volat_data_npy    = self._convert_frame_to_numpy(data = input_dat[2] , seq_len = self.time_interval, timestep = 1)
            input_kospi_data_npy    = self._convert_frame_to_numpy(data = input_dat[3] , seq_len = self.time_interval, timestep = 1)
            input_volat_group_npy   = self._convert_frame_to_numpy(data = input_dat[4] , seq_len = self.time_interval, timestep = 1)
            target_data_npy         = self._convert_frame_to_numpy(data = target_dat, seq_len = self.time_interval, timestep = 1, type='target')
            target_data_npy         = target_data_npy[:, -1, :]

            input_close_data        = np.concatenate([input_close_data_npy[i, :, j] for j in range(input_close_data_npy.shape[2])
                                                    for i in range(input_close_data_npy.shape[0])]).reshape(-1, self.time_interval)
            input_moment_data       = np.concatenate([input_moment_data_npy[i, :, j] for j in range(input_moment_data_npy.shape[2])
                                                    for i in range(input_moment_data_npy.shape[0])]).reshape(-1, self.time_interval)
            input_volat_data        = np.concatenate([input_volat_data_npy[i, :, j] for j in range(input_volat_data_npy.shape[2])
                                                    for i in range(input_volat_data_npy.shape[0])]).reshape(-1, self.time_interval)
            input_kospi_data        = np.concatenate([input_kospi_data_npy[i, :, j] for j in range(input_kospi_data_npy.shape[2])
                                                    for i in range(input_kospi_data_npy.shape[0])]).reshape(-1, self.time_interval)
            input_volat_grp         = np.concatenate([input_volat_group_npy[i, :, j] for j in range(input_volat_group_npy.shape[2])
                                                    for i in range(input_volat_group_npy.shape[0])]).reshape(-1, self.time_interval)

            input_close_data, input_moment_data, input_volat_data, input_kospi_data, input_volat_grp = np.expand_dims(input_close_data, 2), np.expand_dims(input_moment_data, 2), np.expand_dims(input_volat_data, 2), np.expand_dims(input_kospi_data, 2), np.expand_dims(input_volat_grp, 2)
            input_data                                           = np.concatenate([input_close_data, input_moment_data, input_volat_data, input_kospi_data, input_volat_grp], axis=2)
            target_data                                          = target_data_npy.transpose().reshape(-1)
        
        elif self.feature_mode in ['scaled_mamv_kospi_ret_5wr']:
            input_close_data_npy    = self._convert_frame_to_numpy(data = input_dat[0] , seq_len = self.time_interval, timestep = 1)
            input_moment_data_npy   = self._convert_frame_to_numpy(data = input_dat[1] , seq_len = self.time_interval, timestep = 1)
            input_volat_data_npy    = self._convert_frame_to_numpy(data = input_dat[2] , seq_len = self.time_interval, timestep = 1)
            input_kospi_data_npy    = self._convert_frame_to_numpy(data = input_dat[3] , seq_len = self.time_interval, timestep = 1)
            input_5wr_data_npy      = self._convert_frame_to_numpy(data = input_dat[4] , seq_len = self.time_interval, timestep = 1)
            input_volat_group_npy   = self._convert_frame_to_numpy(data = input_dat[5] , seq_len = self.time_interval, timestep = 1)

            target_data_npy         = self._convert_frame_to_numpy(data = target_dat, seq_len = self.time_interval, timestep = 1, type='target')
            target_data_npy         = target_data_npy[:, -1, :]

            input_close_data        = np.concatenate([input_close_data_npy[i, :, j] for j in range(input_close_data_npy.shape[2])
                                                    for i in range(input_close_data_npy.shape[0])]).reshape(-1, self.time_interval)
            input_moment_data       = np.concatenate([input_moment_data_npy[i, :, j] for j in range(input_moment_data_npy.shape[2])
                                                    for i in range(input_moment_data_npy.shape[0])]).reshape(-1, self.time_interval)
            input_volat_data        = np.concatenate([input_volat_data_npy[i, :, j] for j in range(input_volat_data_npy.shape[2])
                                                    for i in range(input_volat_data_npy.shape[0])]).reshape(-1, self.time_interval)
            input_kospi_data        = np.concatenate([input_kospi_data_npy[i, :, j] for j in range(input_kospi_data_npy.shape[2])
                                                    for i in range(input_kospi_data_npy.shape[0])]).reshape(-1, self.time_interval)
            input_5wr_data          = np.concatenate([input_5wr_data_npy[i, :, j] for j in range(input_5wr_data_npy.shape[2])
                                                    for i in range(input_5wr_data_npy.shape[0])]).reshape(-1, self.time_interval)       
            input_volat_grp         = np.concatenate([input_volat_group_npy[i, :, j] for j in range(input_volat_group_npy.shape[2])
                                                    for i in range(input_volat_group_npy.shape[0])]).reshape(-1, self.time_interval)
            
            input_close_data, input_moment_data, input_volat_data, input_kospi_data, input_5wr_data, input_volat_grp = np.expand_dims(input_close_data, 2), np.expand_dims(input_moment_data, 2), np.expand_dims(input_volat_data, 2), np.expand_dims(input_kospi_data, 2),  np.expand_dims(input_5wr_data, 2), np.expand_dims(input_volat_grp, 2)
            input_data                                           = np.concatenate([input_close_data, input_moment_data, input_volat_data, input_kospi_data, input_5wr_data, input_volat_grp], axis=2)
            target_data                                          = target_data_npy.transpose().reshape(-1)

        elif self.feature_mode in ['scaled_hlcma_mv_kospi_ret_5wr']:
            
            input_close_data_npy    = self._convert_frame_to_numpy(data = input_dat[0] , seq_len = self.time_interval, timestep = 1)
            input_high_data_npy     = self._convert_frame_to_numpy(data = input_dat[1] , seq_len = self.time_interval, timestep = 1)
            input_low_data_npy      = self._convert_frame_to_numpy(data = input_dat[2] , seq_len = self.time_interval, timestep = 1)
            input_moment_data_npy   = self._convert_frame_to_numpy(data = input_dat[3] , seq_len = self.time_interval, timestep = 1)
            input_volat_data_npy    = self._convert_frame_to_numpy(data = input_dat[4] , seq_len = self.time_interval, timestep = 1)
            input_kospi_data_npy    = self._convert_frame_to_numpy(data = input_dat[5] , seq_len = self.time_interval, timestep = 1)
            input_5wr_data_npy      = self._convert_frame_to_numpy(data = input_dat[6] , seq_len = self.time_interval, timestep = 1)
            input_volat_group_npy   = self._convert_frame_to_numpy(data = input_dat[7] , seq_len = self.time_interval, timestep = 1)

            target_data_npy         = self._convert_frame_to_numpy(data = target_dat, seq_len = self.time_interval, timestep = 1, type='target')
            target_data_npy         = target_data_npy[:, -1, :]

            input_close_data        = np.concatenate([input_close_data_npy[i, :, j] for j in range(input_close_data_npy.shape[2])
                                                    for i in range(input_close_data_npy.shape[0])]).reshape(-1, self.time_interval)
            input_high_data         = np.concatenate([input_high_data_npy[i, :, j]  for j in range(input_high_data_npy.shape[2])
                                                    for i in range(input_high_data_npy.shape[0])]).reshape(-1, self.time_interval)
            input_low_data          = np.concatenate([input_low_data_npy[i, :, j]   for j in range(input_low_data_npy.shape[2])
                                                    for i in range(input_low_data_npy.shape[0])]).reshape(-1, self.time_interval)            
            input_moment_data       = np.concatenate([input_moment_data_npy[i, :, j] for j in range(input_moment_data_npy.shape[2])
                                                    for i in range(input_moment_data_npy.shape[0])]).reshape(-1, self.time_interval)
            input_volat_data        = np.concatenate([input_volat_data_npy[i, :, j] for j in range(input_volat_data_npy.shape[2])
                                                    for i in range(input_volat_data_npy.shape[0])]).reshape(-1, self.time_interval)
            input_kospi_data        = np.concatenate([input_kospi_data_npy[i, :, j] for j in range(input_kospi_data_npy.shape[2])
                                                    for i in range(input_kospi_data_npy.shape[0])]).reshape(-1, self.time_interval)
            input_5wr_data          = np.concatenate([input_5wr_data_npy[i, :, j] for j in range(input_5wr_data_npy.shape[2])
                                                    for i in range(input_5wr_data_npy.shape[0])]).reshape(-1, self.time_interval)       
            input_volat_grp         = np.concatenate([input_volat_group_npy[i, :, j] for j in range(input_volat_group_npy.shape[2])
                                                    for i in range(input_volat_group_npy.shape[0])]).reshape(-1, self.time_interval)
            
            input_close_data, input_high_data, input_low_data                 = np.expand_dims(input_close_data , 2), np.expand_dims(input_high_data, 2), np.expand_dims(input_low_data, 2)
            input_moment_data, input_volat_data                               = np.expand_dims(input_moment_data, 2), np.expand_dims(input_volat_data, 2)
            input_kospi_data, input_5wr_data, input_volat_grp                 = np.expand_dims(input_kospi_data , 2), np.expand_dims(input_5wr_data, 2), np.expand_dims(input_volat_grp, 2)
            input_data                                                        = np.concatenate([input_close_data, input_high_data, input_low_data, input_moment_data, input_volat_data, input_kospi_data, input_5wr_data, input_volat_grp], axis=2)
            target_data                                                       = target_data_npy.transpose().reshape(-1)
    
        elif self.feature_mode in ['scaled_mamv_kospi_ret_5wr_dxy_ret']:
            input_close_data_npy    = self._convert_frame_to_numpy(data = input_dat[0] , seq_len = self.time_interval, timestep = 1)
            input_moment_data_npy   = self._convert_frame_to_numpy(data = input_dat[1] , seq_len = self.time_interval, timestep = 1)
            input_volat_data_npy    = self._convert_frame_to_numpy(data = input_dat[2] , seq_len = self.time_interval, timestep = 1)
            input_kospi_data_npy    = self._convert_frame_to_numpy(data = input_dat[3] , seq_len = self.time_interval, timestep = 1)
            input_5wr_data_npy      = self._convert_frame_to_numpy(data = input_dat[4] , seq_len = self.time_interval, timestep = 1)
            input_dxy_data_npy      = self._convert_frame_to_numpy(data = input_dat[6] , seq_len = self.time_interval, timestep = 1)
            input_volat_group_npy   = self._convert_frame_to_numpy(data = input_dat[5] , seq_len = self.time_interval, timestep = 1)
            
            target_data_npy         = self._convert_frame_to_numpy(data = target_dat, seq_len = self.time_interval, timestep = 1, type='target')
            target_data_npy         = target_data_npy[:, -1, :]

            input_close_data        = np.concatenate([input_close_data_npy[i, :, j] for j in range(input_close_data_npy.shape[2])
                                                    for i in range(input_close_data_npy.shape[0])]).reshape(-1, self.time_interval)
            input_moment_data       = np.concatenate([input_moment_data_npy[i, :, j] for j in range(input_moment_data_npy.shape[2])
                                                    for i in range(input_moment_data_npy.shape[0])]).reshape(-1, self.time_interval)
            input_volat_data        = np.concatenate([input_volat_data_npy[i, :, j] for j in range(input_volat_data_npy.shape[2])
                                                    for i in range(input_volat_data_npy.shape[0])]).reshape(-1, self.time_interval)
            input_kospi_data        = np.concatenate([input_kospi_data_npy[i, :, j] for j in range(input_kospi_data_npy.shape[2])
                                                    for i in range(input_kospi_data_npy.shape[0])]).reshape(-1, self.time_interval)
            input_5wr_data          = np.concatenate([input_5wr_data_npy[i, :, j] for j in range(input_5wr_data_npy.shape[2])
                                                    for i in range(input_5wr_data_npy.shape[0])]).reshape(-1, self.time_interval) 
            input_dxy_data          = np.concatenate([input_dxy_data_npy[i, :, j] for j in range(input_dxy_data_npy.shape[2])
                                                    for i in range(input_dxy_data_npy.shape[0])]).reshape(-1, self.time_interval)         
            input_volat_grp         = np.concatenate([input_volat_group_npy[i, :, j] for j in range(input_volat_group_npy.shape[2])
                                                    for i in range(input_volat_group_npy.shape[0])]).reshape(-1, self.time_interval)

            input_close_data, input_moment_data, input_volat_data             = np.expand_dims(input_close_data, 2), np.expand_dims(input_moment_data, 2), np.expand_dims(input_volat_data, 2)
            input_kospi_data, input_5wr_data, input_dxy_data, input_volat_grp = np.expand_dims(input_kospi_data, 2),    np.expand_dims(input_5wr_data, 2),   np.expand_dims(input_dxy_data, 2), np.expand_dims(input_volat_grp, 2)
            input_data                                                        = np.concatenate([input_close_data, input_moment_data, input_volat_data, input_kospi_data, input_5wr_data, input_dxy_data, input_volat_grp], axis=2)
            target_data                                                       = target_data_npy.transpose().reshape(-1)

        elif self.feature_mode in ['ohlc_dxy_hlc']:
            input_open_data_npy     = self._convert_frame_to_numpy(data = input_dat[0] , seq_len = self.time_interval, timestep = 1)
            input_high_data_npy     = self._convert_frame_to_numpy(data = input_dat[1] , seq_len = self.time_interval, timestep = 1)
            input_low_data_npy      = self._convert_frame_to_numpy(data = input_dat[2] , seq_len = self.time_interval, timestep = 1)
            input_close_data_npy    = self._convert_frame_to_numpy(data = input_dat[3] , seq_len = self.time_interval, timestep = 1)
            input_dxy_high_data_npy = self._convert_frame_to_numpy(data = input_dat[4] , seq_len = self.time_interval, timestep = 1)
            input_dxy_low_data_npy       = self._convert_frame_to_numpy(data = input_dat[5] , seq_len = self.time_interval, timestep = 1)
            input_dxy_close_data_npy     = self._convert_frame_to_numpy(data = input_dat[6] , seq_len = self.time_interval, timestep = 1)
            input_volat_group_npy   = self._convert_frame_to_numpy(data = input_dat[7] , seq_len = self.time_interval, timestep = 1)
            
            target_data_npy         = self._convert_frame_to_numpy(data = target_dat, seq_len = self.time_interval, timestep = 1, type='target')
            target_data_npy         = target_data_npy[:, -1, :]

            input_open_data         = np.concatenate([input_open_data_npy[i, :, j] for j in range(input_open_data_npy.shape[2])
                                                    for i in range(input_open_data_npy.shape[0])]).reshape(-1, self.time_interval)
            input_high_data         = np.concatenate([input_high_data_npy[i, :, j] for j in range(input_high_data_npy.shape[2])
                                                    for i in range(input_high_data_npy.shape[0])]).reshape(-1, self.time_interval)
            input_low_data          = np.concatenate([input_low_data_npy[i, :, j] for j in range(input_low_data_npy.shape[2])
                                                    for i in range(input_low_data_npy.shape[0])]).reshape(-1, self.time_interval)
            input_close_data        = np.concatenate([input_close_data_npy[i, :, j] for j in range(input_close_data_npy.shape[2])
                                                    for i in range(input_close_data_npy.shape[0])]).reshape(-1, self.time_interval)
            input_dxy_high_data     = np.concatenate([input_dxy_high_data_npy[i, :, j] for j in range(input_dxy_high_data_npy.shape[2])
                                                    for i in range(input_dxy_high_data_npy.shape[0])]).reshape(-1, self.time_interval) 
            input_dxy_low_data      = np.concatenate([input_dxy_low_data_npy[i, :, j] for j in range(input_dxy_low_data_npy.shape[2])
                                                    for i in range(input_dxy_low_data_npy.shape[0])]).reshape(-1, self.time_interval)  
            input_dxy_close_data    = np.concatenate([input_dxy_close_data_npy[i, :, j] for j in range(input_dxy_close_data_npy.shape[2])
                                                    for i in range(input_dxy_close_data_npy.shape[0])]).reshape(-1, self.time_interval)         
            input_volat_grp         = np.concatenate([input_volat_group_npy[i, :, j] for j in range(input_volat_group_npy.shape[2])
                                                    for i in range(input_volat_group_npy.shape[0])]).reshape(-1, self.time_interval)
            
            input_open_data, input_high_data, input_low_data, input_close_data               = np.expand_dims(input_open_data, 2), np.expand_dims(input_high_data, 2), np.expand_dims(input_low_data, 2), np.expand_dims(input_close_data, 2)
            input_dxy_high_data, input_dxy_low_data, input_dxy_close_data, input_volat_grp   = np.expand_dims(input_dxy_high_data, 2),    np.expand_dims(input_dxy_low_data, 2),   np.expand_dims(input_dxy_close_data, 2), np.expand_dims(input_volat_grp, 2)
            input_data                                                          = np.concatenate([input_open_data, input_high_data, input_low_data, input_close_data, input_dxy_high_data, input_dxy_low_data, input_dxy_close_data, input_volat_grp], axis=2)
            target_data                                                         = target_data_npy.transpose().reshape(-1)            

        elif self.feature_mode in ['ohlc_dxy_hlc_ks', 'padding_ohlc_dxy_hlc_ks']:
            input_open_data_npy      = self._convert_frame_to_numpy(data = input_dat[0] , seq_len = self.time_interval, timestep = 1)
            input_high_data_npy      = self._convert_frame_to_numpy(data = input_dat[1] , seq_len = self.time_interval, timestep = 1)
            input_low_data_npy       = self._convert_frame_to_numpy(data = input_dat[2] , seq_len = self.time_interval, timestep = 1)
            input_close_data_npy     = self._convert_frame_to_numpy(data = input_dat[3] , seq_len = self.time_interval, timestep = 1)
            input_dxy_high_data_npy  = self._convert_frame_to_numpy(data = input_dat[4] , seq_len = self.time_interval, timestep = 1)
            input_dxy_low_data_npy   = self._convert_frame_to_numpy(data = input_dat[5] , seq_len = self.time_interval, timestep = 1)
            input_dxy_close_data_npy = self._convert_frame_to_numpy(data = input_dat[6] , seq_len = self.time_interval, timestep = 1)
            input_ks200_data_npy     = self._convert_frame_to_numpy(data = input_dat[7] , seq_len = self.time_interval, timestep = 1)
            input_volat_group_npy    = self._convert_frame_to_numpy(data = input_dat[8] , seq_len = self.time_interval, timestep = 1)
            
            target_data_npy         = self._convert_frame_to_numpy(data = target_dat, seq_len = self.time_interval, timestep = 1, convert_type='target')
            
            if self.data_padding: 
                target_data_npy         = np.concatenate([np.expand_dims(target_data_npy[i, i, :], 0) for i in range(target_data_npy.shape[0])], axis = 0)
                
                input_open_data         = np.concatenate([input_open_data_npy[i, :, j] for j in range(input_open_data_npy.shape[2]) for i in range(input_open_data_npy.shape[0])]).reshape(-1, 391)
                input_high_data         = np.concatenate([input_high_data_npy[i, :, j] for j in range(input_high_data_npy.shape[2]) for i in range(input_high_data_npy.shape[0])]).reshape(-1, 391)
                input_low_data          = np.concatenate([input_low_data_npy[i, :, j] for j in range(input_low_data_npy.shape[2]) for i in range(input_low_data_npy.shape[0])]).reshape(-1, 391)
                input_close_data        = np.concatenate([input_close_data_npy[i, :, j] for j in range(input_close_data_npy.shape[2]) for i in range(input_close_data_npy.shape[0])]).reshape(-1, 391)
                
                input_dxy_high_data     = np.concatenate([input_dxy_high_data_npy[i, :, j] for j in range(input_dxy_high_data_npy.shape[2]) for i in range(input_dxy_high_data_npy.shape[0])]).reshape(-1, 391)
                input_dxy_low_data      = np.concatenate([input_dxy_low_data_npy[i, :, j] for j in range(input_dxy_low_data_npy.shape[2]) for i in range(input_dxy_low_data_npy.shape[0])]).reshape(-1, 391)
                input_dxy_close_data    = np.concatenate([input_dxy_close_data_npy[i, :, j] for j in range(input_dxy_close_data_npy.shape[2]) for i in range(input_dxy_close_data_npy.shape[0])]).reshape(-1, 391)
                
                input_ks200_data        = np.concatenate([input_ks200_data_npy[i, :, j] for j in range(input_ks200_data_npy.shape[2]) for i in range(input_ks200_data_npy.shape[0])]).reshape(-1, 391)
                
                input_volat_grp         = np.concatenate([input_volat_group_npy[i, :, j] for j in range(input_volat_group_npy.shape[2]) for i in range(input_volat_group_npy.shape[0])]).reshape(-1, 391)
                
            else:
                target_data_npy         = target_data_npy[:, -1, :]

                input_open_data         = np.concatenate([input_open_data_npy[i, :, j] for j in range(input_open_data_npy.shape[2])
                                                        for i in range(input_open_data_npy.shape[0])]).reshape(-1, self.time_interval)
                input_high_data         = np.concatenate([input_high_data_npy[i, :, j] for j in range(input_high_data_npy.shape[2])
                                                        for i in range(input_high_data_npy.shape[0])]).reshape(-1, self.time_interval)
                input_low_data          = np.concatenate([input_low_data_npy[i, :, j] for j in range(input_low_data_npy.shape[2])
                                                        for i in range(input_low_data_npy.shape[0])]).reshape(-1, self.time_interval)
                input_close_data        = np.concatenate([input_close_data_npy[i, :, j] for j in range(input_close_data_npy.shape[2])
                                                        for i in range(input_close_data_npy.shape[0])]).reshape(-1, self.time_interval)
                input_dxy_high_data     = np.concatenate([input_dxy_high_data_npy[i, :, j] for j in range(input_dxy_high_data_npy.shape[2])
                                                        for i in range(input_dxy_high_data_npy.shape[0])]).reshape(-1, self.time_interval) 
                input_dxy_low_data      = np.concatenate([input_dxy_low_data_npy[i, :, j] for j in range(input_dxy_low_data_npy.shape[2])
                                                        for i in range(input_dxy_low_data_npy.shape[0])]).reshape(-1, self.time_interval)  
                input_dxy_close_data    = np.concatenate([input_dxy_close_data_npy[i, :, j] for j in range(input_dxy_close_data_npy.shape[2])
                                                        for i in range(input_dxy_close_data_npy.shape[0])]).reshape(-1, self.time_interval)   
                input_ks200_data        = np.concatenate([input_ks200_data_npy[i, :, j] for j in range(input_ks200_data_npy.shape[2])
                                                        for i in range(input_ks200_data_npy.shape[0])]).reshape(-1, self.time_interval)        
                input_volat_grp         = np.concatenate([input_volat_group_npy[i, :, j] for j in range(input_volat_group_npy.shape[2])
                                                        for i in range(input_volat_group_npy.shape[0])]).reshape(-1, self.time_interval)
            
            input_open_data, input_high_data, input_low_data, input_close_data    = np.expand_dims(input_open_data, 2), np.expand_dims(input_high_data, 2), np.expand_dims(input_low_data, 2), np.expand_dims(input_close_data, 2)
            input_dxy_high_data, input_dxy_low_data, input_dxy_close_data         = np.expand_dims(input_dxy_high_data, 2),    np.expand_dims(input_dxy_low_data, 2),   np.expand_dims(input_dxy_close_data, 2)
            input_ks200_data                                                      = np.expand_dims(input_ks200_data, 2)
            input_volat_grp                                                       = np.expand_dims(input_volat_grp, 2)
            
            input_data                                                            = np.concatenate([input_open_data, input_high_data, input_low_data, input_close_data, input_dxy_high_data, input_dxy_low_data, input_dxy_close_data, input_ks200_data, input_volat_grp], axis=2)
            target_data                                                           = target_data_npy.transpose().reshape(-1) 

        elif self.feature_mode in ['ohlc_dxy_hlc_cnh_ks']:
            input_open_data_npy      = self._convert_frame_to_numpy(data = input_dat[0] , seq_len = self.time_interval, timestep = 1)
            input_high_data_npy      = self._convert_frame_to_numpy(data = input_dat[1] , seq_len = self.time_interval, timestep = 1)
            input_low_data_npy       = self._convert_frame_to_numpy(data = input_dat[2] , seq_len = self.time_interval, timestep = 1)
            input_close_data_npy     = self._convert_frame_to_numpy(data = input_dat[3] , seq_len = self.time_interval, timestep = 1)
            input_dxy_high_data_npy  = self._convert_frame_to_numpy(data = input_dat[4] , seq_len = self.time_interval, timestep = 1)
            input_dxy_low_data_npy   = self._convert_frame_to_numpy(data = input_dat[5] , seq_len = self.time_interval, timestep = 1)
            input_dxy_close_data_npy = self._convert_frame_to_numpy(data = input_dat[6] , seq_len = self.time_interval, timestep = 1)
            input_cnh_mid_data_npy   = self._convert_frame_to_numpy(data = input_dat[7] , seq_len = self.time_interval, timestep = 1)
            input_ks200_data_npy     = self._convert_frame_to_numpy(data = input_dat[8] , seq_len = self.time_interval, timestep = 1)
            input_volat_group_npy    = self._convert_frame_to_numpy(data = input_dat[9] , seq_len = self.time_interval, timestep = 1)
            
            target_data_npy         = self._convert_frame_to_numpy(data = target_dat, seq_len = self.time_interval, timestep = 1, type='target')
            target_data_npy         = target_data_npy[:, -1, :]

            input_open_data         = np.concatenate([input_open_data_npy[i, :, j] for j in range(input_open_data_npy.shape[2])
                                                    for i in range(input_open_data_npy.shape[0])]).reshape(-1, self.time_interval)
            input_high_data         = np.concatenate([input_high_data_npy[i, :, j] for j in range(input_high_data_npy.shape[2])
                                                    for i in range(input_high_data_npy.shape[0])]).reshape(-1, self.time_interval)
            input_low_data          = np.concatenate([input_low_data_npy[i, :, j] for j in range(input_low_data_npy.shape[2])
                                                    for i in range(input_low_data_npy.shape[0])]).reshape(-1, self.time_interval)
            input_close_data        = np.concatenate([input_close_data_npy[i, :, j] for j in range(input_close_data_npy.shape[2])
                                                    for i in range(input_close_data_npy.shape[0])]).reshape(-1, self.time_interval)
            input_dxy_high_data     = np.concatenate([input_dxy_high_data_npy[i, :, j] for j in range(input_dxy_high_data_npy.shape[2])
                                                    for i in range(input_dxy_high_data_npy.shape[0])]).reshape(-1, self.time_interval) 
            input_dxy_low_data      = np.concatenate([input_dxy_low_data_npy[i, :, j] for j in range(input_dxy_low_data_npy.shape[2])
                                                    for i in range(input_dxy_low_data_npy.shape[0])]).reshape(-1, self.time_interval)  
            input_dxy_close_data    = np.concatenate([input_dxy_close_data_npy[i, :, j] for j in range(input_dxy_close_data_npy.shape[2])
                                                    for i in range(input_dxy_close_data_npy.shape[0])]).reshape(-1, self.time_interval)   
            input_cnh_mid_data      = np.concatenate([input_cnh_mid_data_npy[i, :, j] for j in range(input_cnh_mid_data_npy.shape[2])
                                                    for i in range(input_cnh_mid_data_npy.shape[0])]).reshape(-1, self.time_interval)   
            input_ks200_data        = np.concatenate([input_ks200_data_npy[i, :, j] for j in range(input_ks200_data_npy.shape[2])
                                                    for i in range(input_ks200_data_npy.shape[0])]).reshape(-1, self.time_interval)        
            input_volat_grp         = np.concatenate([input_volat_group_npy[i, :, j] for j in range(input_volat_group_npy.shape[2])
                                                    for i in range(input_volat_group_npy.shape[0])]).reshape(-1, self.time_interval)
            
            input_open_data, input_high_data, input_low_data, input_close_data    = np.expand_dims(input_open_data, 2)    , np.expand_dims(input_high_data, 2)   , np.expand_dims(input_low_data, 2), np.expand_dims(input_close_data, 2)
            input_dxy_high_data, input_dxy_low_data, input_dxy_close_data         = np.expand_dims(input_dxy_high_data, 2), np.expand_dims(input_dxy_low_data, 2), np.expand_dims(input_dxy_close_data, 2)
            input_cnh_mid_data, input_ks200_data                                  = np.expand_dims(input_cnh_mid_data, 2) , np.expand_dims(input_ks200_data, 2)
            input_volat_grp                                                       = np.expand_dims(input_volat_grp, 2)
            
            input_data                                                            = np.concatenate([input_open_data, input_high_data, input_low_data, input_close_data, input_dxy_high_data, input_dxy_low_data, input_dxy_close_data, input_cnh_mid_data, input_ks200_data, input_volat_grp], axis=2)
            target_data                                                           = target_data_npy.transpose().reshape(-1) 

        return input_data, target_data  

    def _convert_frame_to_numpy(self, data, seq_len, timestep, use_columns=None, convert_type = None):
        """DataFrame을 3dArray로 변환.

        Parameters
        ----------
        data : DataFrame
            변환하고자 하는 2차원 시계열 DataFrame
        seq_len : int
            하나의 2dArray에 들어갈 데이터 길이
        timestep : int
            2dArray 간의 timestep
        use_columns : None or list
            list : DataFrame의 여러 Column중 변환하고자 하는 컬럼

        Returns
        -------
        3dArray(Value), 2dArray(TimeLocation Array)
            3dArray(Value) : 만일 seq_len = 10, timestep=1 이라면 10일의 2d_array를 1일 간격으로 쌓는다.
            2dArray(TimeLocation Array) : 3dArray의 각 배치의 index위치를 알려줌
        """

        # 오류 검사
        # input data는 DataFrame혹은 Series 이어야 합니다.
        input_data_type = data.__class__
        pandas_available_type = [pd.DataFrame().__class__]
        series_available_type = [pd.Series().__class__]
        available_type = pandas_available_type + series_available_type

        if input_data_type not in available_type:
            raise AttributeError("지원하지 않는 Input Data형식 입니다. 지원 형식 : DataFrame, Series")

        # set_mode variable setting
        # Input Data 자료형에 따른 mode 설정
        if input_data_type in pandas_available_type:
            set_mode = "DataFrame"
        if input_data_type in series_available_type:
            set_mode = "Series"

        # 오류 검사
        # seq_len은 int type이어야 합니다.
        if not isinstance(seq_len, int):
            raise AttributeError("seq_len 변수는 Int이어야 합니다.")

        # 오류 검사
        # timestep은 int type이어야 합니다.
        if not isinstance(timestep, int):
            raise AttributeError("timestep 변수는 Int이어야 합니다.")

        # 오류 검사
        # use_columns를 설정할 경우 dataframe의 column안에 존재해야 함.
        if (use_columns is not None) & (set_mode == "DataFrame"):
            for usable_columns in use_columns:
                if usable_columns not in data.columns:
                    raise AttributeError("{}의 데이터가 존재하지 않습니다.".format(usable_columns))

            dataframe = data.loc[self.srt_mrk_t : self.end_mrk_t - 1][use_columns]

        else:
            dataframe = data.loc[self.srt_mrk_t : self.end_mrk_t - 1]

        # batch size 결정
        if set_mode == "DataFrame":
            row_num = len(dataframe)
            col_num = len(dataframe.columns)
        elif set_mode == "Series":
            row_num = len(dataframe)
            col_num = 1

        
        num_of_batch = int((row_num - seq_len) / timestep + 1)
        if self.data_padding:
            if row_num != 391:
                padding = pd.DataFrame(np.zeros((391 - int(row_num), dataframe.shape[1])) * np.nan, columns = dataframe.columns)
                dataframe = pd.concat([dataframe, padding])
                row_num = 391
            num_of_batch = int(row_num)
            seq_len      = 1

        # array 미리 선정
        reformat_value_array = np.zeros((num_of_batch, seq_len, col_num)) * np.nan
        if self.data_padding:
            reformat_value_array = np.zeros((num_of_batch, num_of_batch, col_num)) * np.nan
        forward_constant = 0   
        idx_delay        = 1 if convert_type == 'target' else 0 
        
        if set_mode == "DataFrame":
            index_size = range(len(dataframe.index))
            indexer = np.array(index_size)
            dataframe = np.array(dataframe, dtype=float)
        elif set_mode == "Series":
            index_size = range(len(dataframe.index))
            indexer = np.array(index_size)
            dataframe = np.array(dataframe, dtype=float).reshape(-1, 1)


        
        # 3차원 Array로 dataframe 변형
        for batch_num in range(num_of_batch):
            if (convert_type == 'target') and (batch_num == num_of_batch - 1):
                break
            
            if self.data_padding:
                padding = np.zeros((num_of_batch - seq_len, col_num))
                value   = dataframe[forward_constant + idx_delay : seq_len + idx_delay]
                reformat_value_array[batch_num, :, :] = np.concatenate([value, padding])
                seq_len += timestep
                continue
            
            
            if batch_num == 0:
                reformat_value_array[batch_num, :, :] = dataframe[
                                                        forward_constant + idx_delay : seq_len + idx_delay
                                                        ]
            else:
                reformat_value_array[batch_num, :, :] = dataframe[
                                                        forward_constant + idx_delay : seq_len + idx_delay + (batch_num * timestep)
                                                        ]
    
            forward_constant += timestep

        return reformat_value_array

    def _class_balancing(self, num_classes):
        
        cls_0_idxs = np.where(self.train_y == 0)[0]
        cls_1_idxs = np.where(self.train_y == 1)[0]
        
        cls_0_cnt  = cls_0_idxs.shape[0]
        shuffled_cls_1_idxs = np.random.permutation(cls_1_idxs)
        dn_smp_cls_1_idxs   = shuffled_cls_1_idxs[ : cls_0_cnt]

        dn_smp_train_set    = np.concatenate([cls_0_idxs, dn_smp_cls_1_idxs])
       
        dn_smp_train_x = self.train_x[dn_smp_train_set]
        dn_smp_train_y = self.train_y[dn_smp_train_set]
        
        print(f"[INFO] cls_0_cnt : {dn_smp_train_y[dn_smp_train_y == 0].shape[0]}, cls_1_cnt : {dn_smp_train_y[dn_smp_train_y == 1].shape[0]}\n")
        
        return dn_smp_train_x, dn_smp_train_y

    def load_data(self):
        
        ### (0) 변동성에 따른 input 선택 
        # train_group_idxs     = np.where((self.train_x_npy[:, -1, -1] == 1) | (self.train_x_npy[:, -1, -1] == 10))[0]
        # val_group_idxs       = np.where((self.val_x_npy  [:, -1, -1] == 1) | (self.val_x_npy  [:, -1, -1] == 10))[0]
        # test_group_idxs      = np.where((self.test_x_npy [:, -1, -1] == 1) | (self.test_x_npy [:, -1, -1] == 10))[0]
        
        # self.train_x_npy, self.train_y_npy  = self.train_x_npy[train_group_idxs] , self.train_y_npy[train_group_idxs]
        # self.val_x_npy  , self.val_y_npy    = self.val_x_npy  [  val_group_idxs] , self.val_y_npy  [  val_group_idxs]
        # self.test_x_npy , self.test_y_npy   = self.test_x_npy [ test_group_idxs] , self.test_y_npy [ test_group_idxs]
        
        ### (1) 데이터 분할 및 Tensor 화
        self.train_x,  self.train_y         = torch.tensor(self.train_x_npy), torch.tensor(self.train_y_npy) 
        self.val_x  ,  self.val_y           = torch.tensor(self.val_x_npy), torch.tensor(self.val_y_npy)  
        self.test_x ,  self.test_y          = torch.tensor(self.test_x_npy), torch.tensor(self.test_y_npy)
        
        ### (2) 클래스 불균형 확인 
        cls_percs    = list()
        for cls_i in range(self.label_class):
            cls_i_cnt = self.train_y[self.train_y == cls_i].shape[0]
            cls_i_per = (cls_i_cnt / self.train_y.shape[0]) * 100
            cls_percs.append(cls_i_per)
                        
        # 5% 이상의 비율 차이가 날 경우, Balancing 함수 실행  
        if max(cls_percs)- min(cls_percs) > 5 :
            self.train_x,  self.train_y     = self._class_balancing(num_classes = num_classes)
        
        ### (3) DataSet Setting
        train_dset, val_dset, test_dset     = Datasets(self.train_x, self.train_y), Datasets(self.val_x, self.val_y), Datasets(self.test_x, self.test_y)

        self.train_dataloader               = DataLoader(train_dset, batch_size = self.trn_batch_size, shuffle = True) 
        self.val_dataloader                 = DataLoader(val_dset  , batch_size = self.val_batch_size)
        self.test_dataloader                = DataLoader(test_dset , batch_size = self.val_batch_size)

        print(f"INFO: [load_data] train shape : {self.train_x.shape} \t {self.train_y.shape}")
        print(f"INFO: [load_data] val shape   : {self.val_x.shape}   \t {self.val_y.shape}")
        print(f"INFO: [load_data] test shape  : {self.test_x.shape}  \t {self.test_y.shape}\n")
        