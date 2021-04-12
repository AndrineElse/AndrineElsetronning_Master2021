import numpy as np
import pandas as pd

import helper

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score
from sklearn.cluster import Birch, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from minisom import MiniSom
from sklearn.manifold import TSNE

from sklearn.model_selection import StratifiedKFold
from itertools import product

from warnings import simplefilter, filterwarnings
from sklearn.exceptions import UndefinedMetricWarning
simplefilter('error', UndefinedMetricWarning)
filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

import random
random.seed(10)

class SOM_clustering:
    def __init__(self, n_clusters = 3):
        self.clusters = n_clusters
    
    def fit(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        som_shape = (1, self.clusters)
        som = MiniSom(som_shape[0], som_shape[1], X.shape[1], sigma=.5, learning_rate=.5,
                      neighborhood_function='bubble', random_seed=10)

        som.train_batch(X, 1000, verbose=False)
        self.som = som
        
    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        winner_coordinates = np.array([self.som.winner(x) for x in X]).T
        return winner_coordinates[1]

def get_f1_acc_metric(true, pred): # A performance metric that wants to maximize both accuracy and f1-score at the same time. 
    # NB! Unsure if this metric is sufficient, might just be better to use f1 score 
    try:
        f1 = f1_score(true, pred)
    except UndefinedMetricWarning: 
        return 0
    
    acc = accuracy_score(true, pred)
    return acc
    return np.sqrt(f1**2 + acc**2)
    

def bootstrapped_ensemble_cluster_filter(X_train, y_train, X_test, y_test,
                                     clf_dict, grid_dict = None,
                                     param_dict = None,
                                     clusters_fixed = 20,
                                     thresh_fixed = 2, folds = 10):
    '''Uses BIRCH clustering to find clusters of data which will perform badly on a set classifier

    Args:
        X_train (pandas.DataFrame): Training set feature samples, with indices [0 , 1, ... , n training samples] and columns [0, 1, ... , n features]
        y_train (pandas.Series): Training set label samples, with indices [0 , 1, ... , n training samples] 
        X_test (pandas.DataFrame): Testing set feature samples, with indices [0 , 1, ... , n testing samples] and columns [0, 1, ... , n features]
        y_test (pandas.Series): Testing set label samples, with indices [0 , 1, ... , n testing samples] 

        clf_dict (Dictionary): Format -->  classifier name/ID (str) : classifier (sklearn type classifier)

        grid_dict (Dictionary, optional): Format --> classifier name/ID (str) : grid of hyperparameter-space to search (Dictionary). Defaults to None. 

        param_dict (Dictonary, optional): Contains the hyperparameters found in a hyperparameter search. Defaults to None.

        clusters_fixed (int, optional): Number of clusters to form. Defaults to 20.
        thresh_fixed (int, optional): Threshold for how many ensembles needs to vote for a cluster to keep the cluster. Defaults to 2.
        folds (int, optional): Number of folds to split the training set into, when forming the bootstrapped ensemble cluster voter. Defaults to 8.

    Returns:
        result_dict [Dictionary]: Contains useful information to perform a hyperparameter search. Format --> 
            classifier name/ID (str) :
                'performance' : The performance of the classifier on each of the formed clusters (np.array)
                'keep clusters' : The number ID of the clusters to keep in the test set (np.array)

        indices_dict [Dictionary] : Format --> classifier name/ID (str) : Indices to keep in the testing set (np.array)
    '''

    result_dict = {}
    indices_dict = {}
    
    for name, clf in clf_dict.items():
        # See if hyperparameters are already found 
        if param_dict and (name in param_dict.keys()): 
            clusters = param_dict[name]['custers']
            thresh = param_dict[name]['threshold']
            best_clusters = param_dict[name]['keep clusters']
        else:
            clusters = clusters_fixed
            thresh = thresh_fixed
            best_clusters = None

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

        #model = Birch(n_clusters=clusters)
        #model = MiniBatchKMeans(n_clusters = clusters, random_state = 0)
        #model = GaussianMixture(n_components = clusters, random_state = 0)

        
        model = SOM_clustering(n_clusters = clusters)
        model.fit(X_train)
    
        def get_metric_vector(X_test_temp, y_test_temp, clf_fold, cluster_marked):
            me = []
            for idx in range(clusters): 
                indices = np.where(cluster_marked == idx)[0]
                if len(indices) == 0:
                    #print('No matching indices predicted')
                    me.append(0.5)
                    continue
                X_test_cluster = X_test_temp.iloc[indices]
                y_test_cluster = y_test_temp.iloc[indices]

                y_pred = clf_fold.predict(X_test_cluster)
                me.append(get_f1_acc_metric(y_test_cluster, y_pred))
                
            return np.array(me) 

        def get_test_indices_best_clusters(keep_clusters):
            X_test_marked_clusters = model.predict(X_test)
            
            if len(keep_clusters) > 1:
                keep_indices = np.concatenate([np.where(X_test_marked_clusters == l)[0] for l in keep_clusters])
            else: 
                keep_indices = np.where(X_test_marked_clusters == keep_clusters[0])[0]
            metrics = get_metric_vector(X_test, y_test, clf_init, X_test_marked_clusters)
            return metrics, keep_indices

        
        def get_best_clusters_mask(X_test_temp, y_test_temp, clf_fold):
            cluster_marked = model.predict(X_test_temp) 
            m = get_metric_vector(X_test_temp, y_test_temp, clf_fold, cluster_marked)

            #keep_clusters = m > min(m)
            keep_clusters = m > m[m.argsort()[1]] # Keep all clusters but the two worst performing 
            return keep_clusters

        
        def clustering_based_filter():
            ''' Uses stratified k folds to create a bootstrapped ensemble voter to decide on the best performing clusters 

            Returns:
                cluster_perf (np.array): Performance of each of the clusters 
                keep_indices (np.array): Indices to keep in the testing set, because performance will likely be sufficient
                keep_clusters (np.array): The number ID of the clusters to keep in the test set
            '''
            ensemble_mask = []
            skf = StratifiedKFold(n_splits=folds)
            
            for train_index, test_index in skf.split(X_train, y_train):
                X_train_temp, X_test_temp = X_train.iloc[train_index], X_train.iloc[test_index]
                y_train_temp, y_test_temp = y_train.iloc[train_index], y_train.iloc[test_index]

                clf_fold = clf.fit(X_train_temp, y_train_temp)
                
                temp_mask = get_best_clusters_mask(X_test_temp, y_test_temp, clf_fold)
                ensemble_mask.append(temp_mask)


            best_cluster_mask = np.array(ensemble_mask).sum(axis = 0) > thresh 

            keep_clusters = np.where(best_cluster_mask == True)[0] 

            cluster_perf, keep_indices =  get_test_indices_best_clusters(keep_clusters)

            return cluster_perf, keep_indices, keep_clusters



        if best_clusters is not None:
            metric, indices = get_test_indices_best_clusters(best_clusters)
            indices_dict[name] =  indices
            result_dict[name] = {'performance' : metric, 'keep clusters': best_clusters}
            continue 
        
        metric, indices, best_clusters = clustering_based_filter()
        
        indices_dict[name] =  indices
        
        result_dict[name] = {'performance' : metric, 'keep clusters': best_clusters}
        
    return result_dict, indices_dict


def hyperparam_search(X_train, y_train, X_val, y_val,
                                     clf_dict, grid_dict = None,
                                     clusters_list = [10], thresh_list = [2]):
    '''
    Performes a search for the hyperparameters which will give the best filtering. 
    The best clustering is defined as the one where there is maximum difference in performance between the average cluster and the worst cluster

    Args:
        X_train (pandas.DataFrame): Training set feature samples, with indices [0 , 1, ... , n training samples] and columns [0, 1, ... , n features]
        y_train (pandas.Series): Training set label samples, with indices [0 , 1, ... , n training samples] 
        X_val (pandas.DataFrame): Validation set feature samples, with indices [0 , 1, ... , n validation samples] and columns [0, 1, ... , n features]
        y_val (pandas.Series): Validation set label samples, with indices [0 , 1, ... , n validation samples] 

        clf_dict (Dictionary): Format -->  classifier name/ID (str) : classifier (sklearn type classifier)


        grid_dict (Dictionary, optional): Format --> classifier name/ID (str) : grid of hyperparameter-space to search (Dictionary). Defaults to None. 
        clusters_list (list, optional): Number of clusters to test out. Defaults to [10].
        thresh_list (list, optional): List of thresholds to try out for the ensemble voter. Defaults to [2].

    Returns:
        param_dict (Dictionary): The optimal found hyperparameters. Format -->
        classifier name/ID (str) : 
            'clusters' : Number of clusters (int)
            'threshold' : Threshold for ensemble voter (int)
            'keep clusters' : Cluster IDs that performed well, that should be kept when testing
    '''                            
    max_diff = {key: 0 for key, val in clf_dict.items()}
    param_dict = {}
    
    param_combos = list(product(thresh_list, clusters_list))
    for (t, k) in param_combos:
        temp_dict, _ = bootstrapped_ensemble_cluster_filter(X_train, y_train, X_val, y_val,
                                     clf_dict, grid_dict,
                                     clusters_fixed = k, thresh_fixed = t)

        for key, val in temp_dict.items():
            perf = val['performance']
            diff = np.mean(perf) - np.min(perf[np.nonzero(perf)])
            #diff = np.median(perf) - np.min(perf[np.nonzero(perf)])
            #diff = np.mean(perf) - np.min(perf)

            if diff > max_diff[key]:
                max_diff[key] = diff
                param_dict[key] = {'custers': k, 'threshold': t, 'keep clusters': val['keep clusters']}
                
    return param_dict

def hyperparam_search_noise(X_train, y_train, X_test, y_test,
                                     clf_dict, grid_dict = None,
                                     clusters_list = [10], thresh_list = [2]): 
    '''
    Performes a search for the hyperparameters which will give the best filtering. 
    The best clustering is defined as the one which filters the most samples which contain noise. 
    Simulated noise is added in the hyperparameter search. 

    Args:
        X_train (pandas.DataFrame): Training set feature samples, with indices [0 , 1, ... , n training samples] and columns [0, 1, ... , n features]
        y_train (pandas.Series): Training set label samples, with indices [0 , 1, ... , n training samples] 
        X_val (pandas.DataFrame): Validation set feature samples, with indices [0 , 1, ... , n validation samples] and columns [0, 1, ... , n features]
        y_val (pandas.Series): Validation set label samples, with indices [0 , 1, ... , n validation samples] 

        clf_dict (Dictionary): Format -->  classifier name/ID (str) : classifier (sklearn type classifier)


        grid_dict (Dictionary, optional): Format --> classifier name/ID (str) : grid of hyperparameter-space to search (Dictionary). Defaults to None. 
        clusters_list (list, optional): Number of clusters to test out. Defaults to [10].
        thresh_list (list, optional): List of thresholds to try out for the ensemble voter. Defaults to [2].

    Returns:
        param_dict (Dictionary): The optimal found hyperparameters. Format -->
        classifier name/ID (str) : 
            'clusters' : Number of clusters (int)
            'threshold' : Threshold for ensemble voter (int)
            'keep clusters' : Cluster IDs that performed well, that should be kept when testing
    '''    
    max_diff = {key: 0 for key, val in clf_dict.items()}
    param_dict = {}
    param_combos = list(product(thresh_list, clusters_list))
    for (t, k) in param_combos:
        noise_X, n_idx= helper.add_noise_dataset(X_test, random.randint(10,30), random.randint(3,8))
        temp_dict, indices = bootstrapped_ensemble_cluster_filter(X_train, y_train, noise_X, y_test,
                                     clf_dict, grid_dict,
                                     clusters_fixed = k, thresh_fixed = t)

        def alt_optimal_check():
            amount_of_noise_removed = {}
            for name, vals in indices.items():
                keep = indices[name] 
                count = 0
                for idx in n_idx:
                    if idx not in keep: 
                        count = count + 1
                amount_of_noise_removed[name] = (count/len(n_idx))
            return amount_of_noise_removed

        amount_of_noise_removed = alt_optimal_check()

        for key, val in temp_dict.items():
            if amount_of_noise_removed[key] > max_diff[key]:
                max_diff[key] = amount_of_noise_removed[key]
                param_dict[key] = {'custers': k, 'threshold': t, 'keep clusters': val['keep clusters']}


    return param_dict

def hyperparam_search_accuracy(X_train, y_train, X_test, y_test,
                                     clf_dict, grid_dict = None,
                                     clusters_list = [10], thresh_list = [2]): 
    '''
    Performes a search for the hyperparameters which will give the best filtering. 
    The best clustering is defined as the one which filters the most samples which contain noise. 
    Simulated noise is added in the hyperparameter search. 

    Args:
        X_train (pandas.DataFrame): Training set feature samples, with indices [0 , 1, ... , n training samples] and columns [0, 1, ... , n features]
        y_train (pandas.Series): Training set label samples, with indices [0 , 1, ... , n training samples] 
        X_val (pandas.DataFrame): Validation set feature samples, with indices [0 , 1, ... , n validation samples] and columns [0, 1, ... , n features]
        y_val (pandas.Series): Validation set label samples, with indices [0 , 1, ... , n validation samples] 

        clf_dict (Dictionary): Format -->  classifier name/ID (str) : classifier (sklearn type classifier)


        grid_dict (Dictionary, optional): Format --> classifier name/ID (str) : grid of hyperparameter-space to search (Dictionary). Defaults to None. 
        clusters_list (list, optional): Number of clusters to test out. Defaults to [10].
        thresh_list (list, optional): List of thresholds to try out for the ensemble voter. Defaults to [2].

    Returns:
        param_dict (Dictionary): The optimal found hyperparameters. Format -->
        classifier name/ID (str) : 
            'clusters' : Number of clusters (int)
            'threshold' : Threshold for ensemble voter (int)
            'keep clusters' : Cluster IDs that performed well, that should be kept when testing
    '''    
    max_diff = {key: 0 for key, val in clf_dict.items()}
    param_dict = {}
    param_combos = list(product(thresh_list, clusters_list))
    for (t, k) in param_combos:
        temp_dict, indices = bootstrapped_ensemble_cluster_filter(X_train, y_train, X_test, y_test,
                                     clf_dict, grid_dict,
                                     clusters_fixed = k, thresh_fixed = t)

        for key, val in temp_dict.items():
            clf = clf_dict[key]
            clf.fit(X_train, y_train)
            pred_init = clf.predict(X_test)
            pred_filtered = clf.predict(X_test.iloc[indices[key]])
            acc_init = accuracy_score(y_test, pred_init)
            acc_filtered = accuracy_score(y_test.iloc[indices[key]], pred_filtered)
            diff = acc_filtered - acc_init
            if  diff > max_diff[key]:
                max_diff[key] = diff
                param_dict[key] = {'custers': k, 'threshold': t, 'keep clusters': val['keep clusters']}


    return param_dict


def add_noise_dataset(X, ampl = 10, noise_amount = 4):
    '''
    Noise amount is inverse, meaning higher noise amout, smaller fraction of samples will be effected by noise
    '''
    noise_indices = np.random.RandomState(seed=1).permutation(X.index.tolist())[0:len(X)//noise_amount]
    new_X = X.copy()
    for [idx, row] in new_X.iterrows():
        if idx in noise_indices:
            sig = row
            noise = np.random.RandomState(seed=idx%10).normal(0, ampl, len(sig))
            new_X.iloc[idx] = pd.Series((sig + noise).tolist())
    return new_X , noise_indices