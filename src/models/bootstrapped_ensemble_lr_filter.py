import numpy as np
import pandas as pd
import os

module_path = os.path.abspath(os.path.join('../..'))

from itertools import product 

from warnings import filterwarnings
from sklearn.exceptions import ConvergenceWarning, DataConversionWarning, FitFailedWarning
filterwarnings(action='ignore', category=ConvergenceWarning)
filterwarnings(action='ignore', category=DataConversionWarning)
filterwarnings(action='ignore', category=FitFailedWarning)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

    
def bootstrapped_ensemble_lr_filter(X_train, y_train, X_test, y_test, clf_dict , grid_dict = None, param_dict = None, thresh_fixed= 5, folds_fixed= 10):
    '''
    Uses a secondary classifier to find data that will perfrom badly with the inital classifier

    Args:
        X_train (pandas.DataFrame): Training set feature samples, with indices [0 , 1, ... , n training samples] and columns [0, 1, ... , n features]
        y_train (pandas.Series): Training set label samples, with indices [0 , 1, ... , n training samples] 
        X_test (pandas.DataFrame): Testing set feature samples, with indices [0 , 1, ... , n testing samples] and columns [0, 1, ... , n features]
        y_test (pandas.Series): Testing set label samples, with indices [0 , 1, ... , n testing samples] 

        clf_dict (Dictionary): Format -->  classifier name/ID (str) : classifier (sklearn type classifier)

        grid_dict (Dictionary, optional): Format --> classifier name/ID (str) : grid of hyperparameter-space to search (Dictionary). Defaults to None. 

        param_dict (Dictonary, optional): Contains the hyperparameters found in a hyperparameter search. Defaults to None.

        thresh_fixed (int, optional): Threshold for how many ensembles needs to classify as likely wrongly classified. Defaults to 5.
        folds_fixed (int, optional): Number of folds to split the training set into, when forming the bootstrapped ensemble voter. Defaults to 10.
    Returns: 
        result_dict [Dictionary]: Contains useful information to perform a hyperparameter search. Format --> 
            classifier name/ID (str) :
                'original' : Accuracy score of the original test set (float)
                'filtered' : Accuracy score of the filtered test set (float)

        indices_dict [Dictionary] : Format --> classifier name/ID (str) : Indices to keep in the testing set (np.array)
    '''
    result_dict = {}
    indices_dict = {}

    def get_helping_classifier(X_train_temp, y_train_temp, X_val, y_val, clf_initial):
        '''
        Get the helping classifier

        Returns:
            clf (sklearn type classifier): Classifier trained to classify if something will likely be classified wrongly by the initial classifier
        '''
        clf_initial.fit(X_train_temp,y_train_temp)

        y_pred = clf_initial.predict(X_val)

        new_X = X_val
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

        return (np.array(y_pred).sum(axis = 0) > thresh).astype(int)
    
    def filter_test_set_ensemble():
        y_pred = make_ensemble_classification()
        keep = np.where(y_pred == 0)[0]

        X_test_new = X_test.reset_index(drop = True)
        y_test_new = y_test.reset_index(drop = True)

        X_test_new = X_test_new.iloc[keep]
        y_test_new = y_test_new.iloc[keep]

        return X_test_new, y_test_new, keep
    

    # Loop through all the classifiers in the dataset
    for name, clf in clf_dict.items():
        # See if the hyperparameters are given
        if param_dict and (name in param_dict.keys()): 
            thresh = param_dict[name]['threshold']
            folds = param_dict[name]['folds']
        else:
            thresh = thresh_fixed 
            folds = folds_fixed

        # Use grid search with 5 fold cross-validation, to find the best hyperparameters for the current classifier
        if grid_dict and (name in grid_dict.keys()):

            def get_initial_classifier(clf_initial, grid):
                grid_cv=GridSearchCV(clf_initial,grid,cv=5)
                grid_cv.fit(X_train,y_train)
                return grid_cv.best_estimator_

            clf_init = get_initial_classifier(clf, grid_dict[name]).fit(X_train, y_train)
            clf_dict[name] = clf_init
        else: 
            clf_init = clf.fit(X_train, y_train)

        def get_ensemble_helping_classifier():
            kf = StratifiedKFold(n_splits= folds, random_state=None, shuffle=False)
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


        clf_ensemble = get_ensemble_helping_classifier()

        X_test_filtered, y_test_filtered, keep = filter_test_set_ensemble()

        def get_accuracy_scores():
            y_pred = clf_init.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            y_pred_filtered = clf_init.predict(X_test_filtered)
            acc_filtered = accuracy_score(y_test_filtered, y_pred_filtered)
            return {'original': acc, 'filtered': acc_filtered}

        result_dict[name] = get_accuracy_scores()
        indices_dict[name] = keep

    return result_dict, indices_dict

def hyperparam_search(X_train, y_train, X_test, y_test, clf_dict, grid_dict = None, thresh_list = [5], folds_list = [10]):
    '''
    
    Performes a search for the hyperparameters which will give the best filtering. 
    The best filtering is defined as the one giving the greatest difference in accuracy between the original test set, and the filtered test set

    Args:
        X_train (pandas.DataFrame): Training set feature samples, with indices [0 , 1, ... , n training samples] and columns [0, 1, ... , n features]
        y_train (pandas.Series): Training set label samples, with indices [0 , 1, ... , n training samples] 
        X_val (pandas.DataFrame): Validation set feature samples, with indices [0 , 1, ... , n validation samples] and columns [0, 1, ... , n features]
        y_val (pandas.Series): Validation set label samples, with indices [0 , 1, ... , n validation samples] 

        clf_dict (Dictionary): Format -->  classifier name/ID (str) : classifier (sklearn type classifier)

        grid_dict (Dictionary, optional): Format --> classifier name/ID (str) : grid of hyperparameter-space to search (Dictionary). Defaults to None. 
        thresh_list (list, optional): List of thresholds to try out for the ensemble voter. Defaults to [5].
        folds_list (list, optional): Number of folds to test out. Defaults to [10].

    Returns:
        param_dict (Dictionary): The optimal found hyperparameters. Format -->
        classifier name/ID (str) : 
            'threshold' : Threshold for ensemble voter (int)
            'folds' : Folds in the ensemble voter (int)
    '''
    max_diff = {key: 0 for key, val in clf_dict.items()}
    return_dict = {}
    param_combos = list(product(thresh_list, folds_list))
    for (thresh, folds) in param_combos:
        if thresh > folds: # Cannot have more votes than voters
            continue

        temp_dict, _ = bootstrapped_ensemble_lr_filter(X_train, y_train, X_test, y_test, clf_dict, grid_dict, thresh_fixed = thresh, folds_fixed = folds)
        for key, val in temp_dict.items():
            diff = val['filtered'] - val['original']
            if diff > max_diff[key]:
                max_diff[key] = diff
                return_dict[key] = {'threshold': thresh, 'folds': folds}
                
    return return_dict
    
