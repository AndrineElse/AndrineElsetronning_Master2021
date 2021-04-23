import sys
import os
import pandas as pd
import numpy as np
from time import time
from sklearn.preprocessing import MinMaxScaler

module_path = os.path.abspath(os.path.join('../..'))

cwd = os.path.abspath(os.path.join(''))

from sktime.utils.data_io import load_from_tsfile_to_dataframe
from sktime_dl.deeplearning import ResNetClassifier
import pickle

    
def change_labels(y):
    y_new = y.copy()

    y_new[y_new == 'exp_wheeze'] = 'wheeze'
    y_new[y_new == 'insp_wheeze'] = 'wheeze'
    
    y_new[y_new == 'exp_crackle'] = 'crackle'
    y_new[y_new == 'insp_crackle'] = 'crackle'
    
    return y_new

def save_pred_proba(y_test, pred):
    with open(cwd + '/pickle/true_ResNet.pkl','wb') as f:
        pickle.dump(y_test, f)
        
        
    with open(cwd + '/pickle/pred_ResNet.pkl','wb') as f:
        pickle.dump(pred, f)
        
    print('ResNet predictions are saved')


def main():
    start = time()
    X_train, y_train_ = load_from_tsfile_to_dataframe(module_path + f'/data/ts_files/UiT_5s_TRAIN.ts')
    X_test, y_test_ = load_from_tsfile_to_dataframe(module_path + f'/data/ts_files/UiT_5s_TEST.ts')
    
    y_train = change_labels(y_train_)
    y_test = change_labels(y_test_)
        
    model = ResNetClassifier(nb_epochs=100, verbose=True)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    save_pred_proba(y_test, y_pred)
    
    #model.save(cwd + '/resNet5s')
   # print('Model saved')
    
    total_time = time() - start
    f = open(cwd + '/computation_times.txt', 'a')
    f.write('ResNet 5s, ' + str(total_time) + '\n')
    print('Finished writing Resnet')
    f.close()
    
main()