'''
Code for transforming the dataset into TRAIN and TEST set 
Using Catch22 for feature extraction
'''
import sys
import os
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


def get_dataset(ts_path):
    return load_from_tsfile_to_dataframe(ts_path)
    
    
def run_catch22(X,y):
    kwargs = dict(test_size=0.2, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, **kwargs)

    c22 = Catch22()
    c22.fit(X_train, y_train)
    
    X_train_t = c22.transform(X_train)
    X_test_t = c22.transform(X_test)
    
    return X_train_t, X_test_t, y_train, y_test
    
def save_catch22_transform(X_train, X_test, y_train, y_test, ts_path):
    new_file_paths = {
        ts_path.split('.')[0] + '_transformed_catch22_TRAIN.ts' : (X_train, y_train) ,
        ts_path.split('.')[0] + '_transformed_catch22_TEST.ts' : (X_test, y_test) }
    
    metadata = get_metadata(ts_path)
    
    for path, data in new_file_paths.items():
        w = open(path, 'w+')
        for m in metadata:
            w.write(m)
        X, y = data[0], data[1]
        for x , l in zip(X.values, y):
            new_row = str(list(x))[1:-1].replace(' ', '') + ':' + l + '\n'
            w.write(new_row)    
            
    print('Catch 22 dataset is saved')

def main():
    start = time()
    ts_path = cwd + '/data/test.ts'
    
    X, y = get_dataset(ts_path)
    
    X_train, X_test, y_train, y_test = run_catch22(X,y)
    
    save_catch22_transform(X_train, X_test, y_train, y_test, ts_path)
    
    total_time = time() - start
    f = open(cwd + '/computation_times.txt', 'a')
    f.write('Catch22, ' + str(total_time) + '\n')
    print('Finished writing catch22 transform')
    
main()