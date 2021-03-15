import numpy as np
import pandas as pd
import os
import sys
module_path = os.path.abspath(os.path.join('../..'))
sys.path.insert(1, module_path + '/src')

import utility
import time

from sklearn.linear_model import LogisticRegression

import warnings
from sklearn.exceptions import ConvergenceWarning, DataConversionWarning, FitFailedWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=FitFailedWarning)

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

    
def compare_classifiers(X_train, y_train, X_test, y_test, clf_dict , grid_dict = None, param_dict = None, thresh_fixed= 5, folds_fixed= 10):            
    def get_helping_classifier(X_train_temp, y_train_temp, X_val, y_val, clf_initial):
        '''Get the helping classifier

        returns a classifier, which classifies if something is likely to be classified wrongly

        '''
        new_X = X_val

        clf_initial.fit(X_train_temp,y_train_temp)

        y_pred = clf_initial.predict(X_val)
        new_y = pd.Series(np.hstack([y_pred != y_val]))

        clf = LogisticRegression(random_state=0, solver = 'liblinear',penalty = 'l1', class_weight = 'balanced')
        #clf = SVC(gamma='auto', class_weight = 'balanced')
        if 'True' not in list(new_y.astype(str)):
             return None, None
        clf.fit(new_X,new_y)
        return clf  
    def make_ensemble_classification():
        y_pred = []
        for clf in clf_ensemble: 
            y_pred_temp = clf.predict(X_test).astype(int)
            y_pred.append(y_pred_temp)
        y_pred = np.array(y_pred)
        y_pred_sum = y_pred.sum(axis = 0)
        idx_true = np.where(y_pred_sum > thresh)[0]

        y_pred = np.zeros(y_pred_temp.shape)


        y_pred[idx_true] = 1
        return y_pred
    
    def filter_test_set_ensemble():
        y_pred = make_ensemble_classification()
        keep = np.where(y_pred == 0)[0]
        #print('Indices to delete: ')
        #print(to_del)
        X_test_new = X_test.reset_index(drop = True)
        y_test_new = y_test.reset_index(drop = True)
        X_test_new = X_test_new.iloc[keep]
        y_test_new = y_test_new.iloc[keep]
        return X_test_new, y_test_new, keep
    
    result_dict = {}
    new_test_dict = {}
    # Loop through all the classifiers in the dataset
    
    for name, clf in clf_dict.items():
        if param_dict and (name in param_dict.keys()): 
            thresh = param_dict[name]['threshold']
            folds = param_dict[name]['folds']
        else:
            thresh = thresh_fixed 
            folds = folds_fixed
        #print(f'Setting the initial classifier, for {name}')
        if grid_dict and (name in grid_dict.keys()):

            def get_initial_classifier(clf_initial, grid):
                grid_cv=GridSearchCV(clf_initial,grid,cv=5)
                grid_cv.fit(X_train,y_train)
                return grid_cv.best_estimator_

            #print('Grid search option is activated ')
            clf_init = get_initial_classifier(clf, grid_dict[name]).fit(X_train, y_train)
            clf_dict[name] = clf_init
        else: 
            clf_init = clf.fit(X_train, y_train)

        def get_ensemble_helping_classifier():
            kf = StratifiedKFold(n_splits= folds, random_state=None, shuffle=False)
            #kf.get_n_splits(X_train)
            classifiers = []
            for train_index, val_index in kf.split(X_train, y_train):
                clf_temp = get_helping_classifier(X_train.iloc[train_index], 
                                                  y_train.iloc[train_index],
                                                  X_train.iloc[val_index],
                                                  y_train.iloc[val_index],
                                                  clf_init)
                if clf_temp == None:
                    continue
                classifiers.append(clf_temp)
            return classifiers


        #print(f'Creating an ensemble of helping classifiers, for {name}')
        # Obtaining an ensemble of helping classifiers to assist the initial classifier
        clf_ensemble = get_ensemble_helping_classifier()

        X_test_2, y_test_2, keep = filter_test_set_ensemble()

        def get_acc1_acc2():
            y_pred = clf_init.predict(X_test)
            acc_1 = accuracy_score(y_test, y_pred)

            y_pred_2 = clf_init.predict(X_test_2)
            acc_2 = accuracy_score(y_test_2, y_pred_2)
            return {'original': acc_1, 'filtered': acc_2}


        result_dict[name] = get_acc1_acc2()
        new_test_dict[name] = keep
    return result_dict, new_test_dict

def hyperparam_search(X_train, y_train, X_test, y_test, clf_dict, grid_dict = None, thresh_list = [5], folds_list = [10]):
    '''
    Args:
    thresh_list = List of thresholds to try out, for filtering test set
    folds_list = List of number of folds to have in the bootstrapped ensemble method
    
    Returns:
    A dict of the optimal parameters corresponding to the dict of classifiers (clf_dict)
    key = (optimal_thresh, optimal_folds)
    '''
    max_diff = {key: 0 for key, val in clf_dict.items()}
    return_dict = {}
    for thresh in thresh_list:
        for folds in folds_list:
            if thresh > folds:
                continue
            #print(f'Testing fold = {folds}, and threshold = {thresh}')
            temp_dict, _ = compare_classifiers(X_train, y_train, X_test, y_test, clf_dict, grid_dict, thresh_fixed = thresh, folds_fixed = folds)
            for key, val in temp_dict.items():
                diff = val['filtered'] - val['original']
                if diff > max_diff[key]:
                    max_diff[key] = diff
                    return_dict[key] = {'threshold': thresh, 'folds': folds}
                
    return return_dict
    
def add_noise_dataset(X, ampl = 1, noise_amount = 6 ):
    '''
    noise amount is inverse, meaning higher noise amout, means less noise... 
    '''
    noise_indices = np.random.RandomState(seed=1).permutation(X.index.tolist())[0:len(X)//noise_amount]
    new_X = X.copy()
    for [idx, row] in X.iterrows():
        if idx in noise_indices:
            sig = row.tolist()
            noise = np.random.RandomState(seed=idx%10).normal(0, ampl, len(sig))
            new_X.iloc[idx] = pd.Series(sig + noise)
    return new_X , noise_indices