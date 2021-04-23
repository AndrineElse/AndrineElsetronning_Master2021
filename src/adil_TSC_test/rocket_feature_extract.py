'''
Code for transforming the dataset into TRAIN and TEST set 
Using ROCKET for feature extraction
'''
import sys
import os
import pandas as pd
import numpy as np
from time import time
from sklearn.preprocessing import MinMaxScaler

module_path = os.path.abspath(os.path.join('../..'))

cwd = os.path.abspath(os.path.join(''))

from sktime.utils.data_io import load_from_tsfile_to_dataframe
from sktime.transformations.panel.rocket import Rocket

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2

def change_labels(y):
    y_new = y.copy()

    y_new[y_new == 'exp_wheeze'] = 'wheeze'
    y_new[y_new == 'insp_wheeze'] = 'wheeze'
    
    y_new[y_new == 'exp_crackle'] = 'crackle'
    y_new[y_new == 'insp_crackle'] = 'crackle'
    
    return y_new

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

    
    
def run_rocket(X_train, X_test, X_val, y_train,y_test, y_val):
    ROCKET = Rocket(num_kernels=10_000)
    ROCKET.fit(X_train)
    
    X_train_t = ROCKET.transform(X_train)
    X_test_t = ROCKET.transform(X_test)
    X_val_t = ROCKET.transform(X_val)
    
    X_train_t.columns = np.arange(len(X_train_t.columns))
    X_test_t.columns = np.arange(len(X_test_t.columns))
    X_val_t.columns = np.arange(len(X_val_t.columns))
    
    
    scaler = MinMaxScaler()
    scaler.fit(X_train_t)
    X_train_t = pd.DataFrame(scaler.transform(X_train_t))
    X_test_t = pd.DataFrame(scaler.transform(X_test_t))
    X_val_t = pd.DataFrame(scaler.transform(X_val_t))
    
    k = 250
    select = SelectKBest(chi2, k=k)
    _ = select.fit(X_train_t, y_train)
    indices = select.get_support(indices = True)
    
    X_test_t = X_test_t[indices]
    X_val_t = X_val_t[indices]
    X_train_t = X_train_t[indices]
    
    return X_train_t, X_test_t, X_val_t , y_train, y_test, y_val
    
def save_rocket_transform(X_train, X_test, X_val, y_train,y_test, y_val, path_transformed, path_original):
    new_file_paths = {
        path_transformed + '_transformed_rocket_5s_TRAIN.ts' : (X_train, y_train) ,
        path_transformed + '_transformed_rocket_5s_TEST.ts' : (X_test, y_test),
        path_transformed + '_transformed_rocket_5s_VAL.ts' : (X_val, y_val)
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
            
    print('ROCKET dataset is saved')

def main():
    start = time()
    X_train, y_train = load_from_tsfile_to_dataframe(module_path + f'/data/ts_files/UiT_5s_TRAIN.ts')
    X_test, y_test = load_from_tsfile_to_dataframe(module_path + f'/data/ts_files/UiT_5s_TEST.ts')
    X_val, y_val = load_from_tsfile_to_dataframe(module_path + f'/data/ts_files/UiT_5s_VAL.ts')
    path_original = module_path + f'/data/ts_files/UiT_5s_TEST.ts'
    path_transformed = module_path + '/src/adil_TSC_test/transformed_datasets/UiT'
    
    X_train, X_test, X_val, y_train,y_test, y_val = run_rocket(X_train, X_test, X_val, y_train,y_test, y_val)
    
    save_rocket_transform(X_train, X_test, X_val, y_train,y_test, y_val, path_transformed, path_original)
    
    total_time = time() - start
    f = open(cwd + '/computation_times.txt', 'a')
    f.write('ROCKET 5s, ' + str(total_time) + '\n')
    print('Finished writing ROCKET transform')
    f.close()
    
main()