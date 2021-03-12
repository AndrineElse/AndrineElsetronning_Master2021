import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score
from sklearn.cluster import KMeans, Birch

from sklearn.model_selection import StratifiedKFold

import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.simplefilter('error', UndefinedMetricWarning)

import numpy as np
import random

random.seed(10)
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 


def get_f1_acc_metric(true, pred):
    try:
        f1 = f1_score(true, pred)
    except UndefinedMetricWarning: 
        return 0
    acc = accuracy_score(true, pred)
    #return f1
    return np.sqrt(f1**2 + acc**2)
    

def bootstrapped_ensemble_cluster_filter(X_train, y_train, X_test, y_test,
                                     clf_dict, grid_dict = None,
                                     param_dict = None,
                                     clusters_fixed = 20,
                                     thresh_fixed = 2, folds = 8):
    result_dict = {}
    indices_dict = {}
    
    
    # Loop through all the classifiers 
    
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
            
        model = Birch(n_clusters=clusters).fit(X_train)

        def get_metric_vector(X_test_temp, y_test_temp, clf_fold, cluster_marked):
            me = []
            for idx in range(clusters): 
                indices = np.where(cluster_marked == idx)[0]
                if len(indices) == 0:
                    print('No matching indices predicted')
                    me.append(0)
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

            # Get the resulting metrics (sqt(acc ^2 + f1 ^2)) for each of the current clusters
            m = get_metric_vector(X_test_temp, y_test_temp, clf_fold, cluster_marked)

            #mean_metrics = np.mean(m) - np.std(m)
            #mean_metrics = np.median(m)
            #keep_clusters = m >= mean_metrics # Mask of which clusters to keep. If True, then keep the cluster
            keep_clusters = m > min(m)
            return keep_clusters

        
        def clustering_based_filter():
            '''
            Uses stratified k fold to split the dataset into temporary training and validation sets, to find the best performing clusters.
            In the end the ensembles 'votes' for which clusters to keep in the test set. 
            If two or more ensembles has voted for a cluster as the worst performing one, then this cluster is removed from the test set. 
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

            ## Using the best cluster mask on the test set
            keep_clusters = np.where(best_cluster_mask == True)[0] 

            metrics, keep_indices =  get_test_indices_best_clusters(keep_clusters)

            return metrics, keep_indices, keep_clusters

        if best_clusters is not None:
            metric, indices = get_test_indices_best_clusters(best_clusters)
            indices_dict[name] =  indices
            result_dict[name] = {'performance' : metric, 'keep clusters': best_clusters}
            continue 
        
        metric, indices, best_clusters = clustering_based_filter()
        
        indices_dict[name] =  indices
        
        result_dict[name] = {'performance' : metric, 'keep clusters': best_clusters}
        
    return result_dict, indices_dict


def hyperparam_search(X_train, y_train, X_test, y_test,
                                     clf_dict, grid_dict = None,
                                     clusters_list = [10], thresh_list = [2]):
    max_diff = {key: 0 for key, val in clf_dict.items()}
    return_dict = {}
    for t in thresh_list:
        for k in clusters_list:
            #print(f'Testing number of clusters = {k}, and threshold = {t}')
            temp_dict, _ = bootstrapped_ensemble_cluster_filter(X_train, y_train, X_test, y_test,
                                         clf_dict, grid_dict,
                                         clusters_fixed = k, thresh_fixed = t)

            for key, val in temp_dict.items():
                perf = val['performance']
                diff = np.mean(perf) - np.min(perf[np.nonzero(perf)])
                #diff = np.mean(metrics) - min(metrics) # Need to fix return value to make this the filtering criteria

                #diff = val['filtered'] - val['original']
                if diff > max_diff[key]:
                    max_diff[key] = diff
                    return_dict[key] = {'custers': k, 'threshold': t, 'keep clusters': val['keep clusters']}
                
    return return_dict

def hyperparam_search_alt(X_train, y_train, X_test, y_test,
                                     clf_dict, grid_dict = None,
                                     clusters_list = [10], thresh_list = [2]):         
    max_diff = {key: 0 for key, val in clf_dict.items()}
    return_dict = {}
    for t in thresh_list:
        for k in clusters_list:
            #print(f'Testing number of clusters = {k}, and threshold = {t}')
            noise_X, n_idx= add_noise_dataset(X_test, random.randint(1,20), random.randint(3,8))
            results, indices = bootstrapped_ensemble_cluster_filter(X_train, y_train, noise_X, y_test,
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

            for key, val in results.items():
                if amount_of_noise_removed[key] > max_diff[key]:
                    max_diff[key] = amount_of_noise_removed[key]
                    return_dict[key] = {'custers': k, 'threshold': t}
    '''                
    for name, prop in return_dict.items():
        k = prop['custers']
        model = KMeans(n_clusters=k, random_state = 1)
        model.fit(X_train)
        new_y = model.predict(X_train)
        print(f'Optimal clustering for {name}: {k} - clusters \n')
        f = scatterplot_with_colors(X_train.values, new_y)
        plt.show()'''
    return return_dict


def add_noise_dataset(X, ampl = 10, noise_amount = 4):
    '''
    noise amount is inverse, meaning higher noise amout, means less noise... 
    '''
    noise_indices = np.random.RandomState(seed=1).permutation(X.index.tolist())[0:len(X)//noise_amount]
    new_X = X.copy()
    for [idx, row] in new_X.iterrows():
        if idx in noise_indices:
            sig = row
            noise = np.random.RandomState(seed=idx%10).normal(0, ampl, len(sig))
            new_X.iloc[idx] = pd.Series((sig + noise).tolist())
    return new_X , noise_indices

def get_results_dict(X_train, y_train, X_test, y_test, clf_dict, indices_dict):
    results_dict = {}

    for clf_name, clf in clf_dict.items():
        clf.fit(X_train, y_train)
        init = accuracy_score(y_test, clf.predict(X_test))
        filtered = accuracy_score(y_test[indices_dict[clf_name]],
                                  clf.predict(X_test.iloc[indices_dict[clf_name]]))
        results_dict[clf_name] = {'original': init, 'filtered': filtered}
    return results_dict