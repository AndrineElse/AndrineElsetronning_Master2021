'''
Code for transforming the dataset into TRAIN and TEST set 
Using Catch22 for feature extraction
'''
import sys
import os

module_path = os.path.abspath(os.path.join('../..'))
import pandas as pd
import numpy as np
from time import time


cwd = os.path.abspath(os.path.join(''))

from sktime.utils.data_io import load_from_tsfile_to_dataframe
from sktime.transformations.panel.catch22_features import Catch22

from sklearn.model_selection import train_test_split

def get_metadata(ts_path):
    f = open(ts_path, 'r')

    number_of_lines = 15
    metadata = []
    for i in range(number_of_lines):
        line = f.readline()
        metadata.append(line)
        if line[:5] == '@data':
            break
        
    return metadata


def run_catch22(X_train, X_test, X_val, y_train,y_test, y_val):
    c22 = Catch22()
    c22.fit(X_train, y_train)
    
    X_train_t = c22.transform(X_train)
    X_test_t = c22.transform(X_test)
    X_val_t = c22.transform(X_val)
    return X_train_t, X_test_t, X_val_t , y_train, y_test, y_val
    
def save_catch22_transform(X_train, X_test, X_val, y_train,y_test, y_val, path_transformed, path_original):
    new_file_paths = {
        path_transformed + '_transformed_catch22_15s_TRAIN.ts' : (X_train, y_train) ,
        path_transformed + '_transformed_catch22_15s_TEST.ts' : (X_test, y_test),
        path_transformed + '_transformed_catch22_15s_VAL.ts' : (X_val, y_val)
        }
    
    metadata = get_metadata(path_original)
    
    for path, data in new_file_paths.items():
        w = open(path, 'w+')
        for m in metadata:
            w.write(m)
        X, y = data[0], data[1]
        for x , l in zip(X.values, y):
            new_row = str(list(x))[1:-1].replace(' ', '') + ':' + l + '\n'
            w.write(new_row)    
            
    print('Catch22 dataset is saved')

def main():
    start = time()
    X_train, y_train = load_from_tsfile_to_dataframe(module_path + f'/data/ts_files/UiT_15s_TRAIN.ts')
    X_test, y_test = load_from_tsfile_to_dataframe(module_path + f'/data/ts_files/UiT_15s_TEST.ts')
    X_val, y_val = load_from_tsfile_to_dataframe(module_path + f'/data/ts_files/UiT_15s_VAL.ts')
    path_original = module_path + f'/data/ts_files/UiT_5s_TEST.ts'
    path_transformed = module_path + '/src/adil_TSC_test/transformed_datasets/UiT'
    
    
    X_train, X_test, X_val , y_train, y_test, y_val = run_catch22(X_train, X_test, X_val, y_train,y_test, y_val)
    
    save_catch22_transform(X_train, X_test, X_val, y_train,y_test, y_val, path_transformed, path_original)
    
    total_time = time() - start
    f = open(cwd + '/computation_times.txt', 'a')
    f.write('Catch22, ' + str(total_time) + '\n')
    f.close()
    print('Finished writing catch22 transform')
    
main()