#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 18:02:24 2021

@author: andrine
"""

from sktime.classification.dictionary_based import ContractableBOSS
import sys
import os
import pandas as pd
import numpy as np
from time import time
from sklearn.preprocessing import MinMaxScaler
import pickle
module_path = os.path.abspath(os.path.join('../..'))

cwd = os.path.abspath(os.path.join(''))

from sktime.utils.data_io import load_from_tsfile_to_dataframe

from sklearn.model_selection import train_test_split

def change_labels(y):
    y_new = y.copy()

    y_new[y_new == 'exp_wheeze'] = 'wheeze'
    y_new[y_new == 'insp_wheeze'] = 'wheeze'
    
    y_new[y_new == 'exp_crackle'] = 'crackle'
    y_new[y_new == 'insp_crackle'] = 'crackle'
    
    return y_new


    
def run_cboss(X_train, X_test, y_train,y_test):
    cboss = ContractableBOSS(
        n_parameter_samples=50, max_ensemble_size=10, random_state=0
    )
    cboss.fit(X_train, y_train)
    

    y_pred = cboss.predict(X_test)
    y_pred_proba = cboss.predict_proba(X_test)
    
    return y_test, y_pred, y_pred_proba
    
    
def save_pred_proba(y_test, pred, proba):
    with open(cwd + '/pickle/true_cBOSS.pkl','wb') as f:
        pickle.dump(y_test, f)
        
        
    with open(cwd + '/pickle/pred_cBOSS.pkl','wb') as f:
        pickle.dump(pred, f)
        
    with open(cwd + '/pickle/prob_cBOSS.pkl','wb') as f:
        pickle.dump(proba, f)     
        
    print('CBOSS predictions are saved')

def main():
    start = time()
    X_train, y_train_ = load_from_tsfile_to_dataframe(module_path + f'/data/ts_files/UiT_5s_TRAIN.ts')
    X_test, y_test_ = load_from_tsfile_to_dataframe(module_path + f'/data/ts_files/UiT_5s_TEST.ts')
    

    y_train = change_labels(y_train_)
    y_test = change_labels(y_test_)
    
    #X_train , X_test = X_train[:100] , X_test[:100]
    #y_train , y_test = y_train[:100] , y_test[:100]
    
    #X_train = np.concatenate([X_train, X_val])
    #y_train = np.concatenate([y_train, y_val])
    
    y_test, y_pred, y_pred_proba = run_cboss(X_train, X_test, y_train,y_test)
    
    save_pred_proba(y_test, y_pred, y_pred_proba)
    
    total_time = time() - start


    f = open(cwd + '/computation_times.txt', 'a')
    f.write('cboss 5s, ' + str(total_time) + '\n')
    print('Finished writing cboss transform')
    
main()