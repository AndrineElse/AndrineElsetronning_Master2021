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


cwd = os.path.abspath(os.path.join(''))

from sktime.utils.data_io import load_from_tsfile_to_dataframe

from sklearn.model_selection import train_test_split


def get_dataset(ts_path):
    return load_from_tsfile_to_dataframe(ts_path)
    
    
def run_cboss(X,y):
    kwargs = dict(test_size=0.2, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, **kwargs)

   
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
    ts_path = cwd + '/data/test.ts'
    
    X, y = get_dataset(ts_path)
    
    y_test, y_pred, y_pred_proba = run_cboss(X,y)
    
    save_pred_proba(y_test, y_pred, y_pred_proba)
    
    total_time = time() - start


    f = open(cwd + '/computation_times.txt', 'a')
    f.write('cboss, ' + str(total_time) + '\n')
    print('Finished writing cboss transform')
    
main()