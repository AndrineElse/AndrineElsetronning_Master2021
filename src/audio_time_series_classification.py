import sys
import os
import pandas as pd
import numpy as np

module_path = os.path.abspath(os.path.join('../..'))

sys.path.insert(1, module_path + '/src')
import utility

from sktime.utils.data_io import load_from_tsfile_to_dataframe
from sktime.utils.data_processing import from_nested_to_2d_array

import librosa
from scipy.stats import skew 

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


class overproduced_audio_time_series_features:
    def __init__(self, ts_file_path, sr, UCR_file = False, n_mfcc = 30,
                 denoise = False, downsample = False, downsample_new_sr = 8000, update = False, name = None,
                 module_path = module_path):
        '''
        Class for producing features for audio data. 

        Args:
            ts_file_path (str): path to where the files are located , without endings (_TRAIN.ts , _TEST.ts ), if UCR file. 
            sr (int): Sampling rate
            UCR_file (bool, optional): The data to be explored is in the UCR archive. Defaults to False.
            n_mfcc (int, optional): Number of MFCC features to include in the transformed feature space. Defaults to 30.
            denoise (bool, optional): Denoise the audio with wavelet denoising before extracting features. Defaults to False.
            downsample (bool, optional): Downsample the audio before extracting features. Defaults to False.
            downsample_new_sr (int, optional): NB! if downsample is true, then specify what one should downsample to. Defaults to 8000.
            update (bool, optional): If true, then the transformed features will be resaved, even if they have been extracted before. Defaults to False.
            name ([type], optional): If not a UCR file, and a spesific name is desired. Defaults to None.
        '''
       
        
        if (UCR_file):
            name_UCR_dataset = ts_file_path.split('/')[-1]

            path1 = ts_file_path + f'/{name_UCR_dataset}_TEST.ts'
            path2 = ts_file_path + f'/{name_UCR_dataset}/{name_UCR_dataset}_TEST.ts'

            if not(os.path.exists(path1)) and not(os.path.exists(path2)):
                # Some of the files in the UCR archive are not converted from .arff format to .ts format, hence this needs to be done
                filepath = ts_file_path + f'/{name_UCR_dataset}'

                filename  = f'{name_UCR_dataset}_TEST.arff'
                utility.convert_arff_to_ts(filepath, filename)
    
                filename  = f'{name_UCR_dataset}_TRAIN.arff'
                utility.convert_arff_to_ts(filepath, filename)

                ts_file_path = filepath + '/' + name_UCR_dataset

            elif not(os.path.exists(path1)) and (os.path.exists(path2)):
                # If the file is already converted, but lies in a different folder
                print('here')
                filepath = ts_file_path + f'/{name_UCR_dataset}'
                ts_file_path = filepath + '/' + name_UCR_dataset
            else:
                ts_file_path = ts_file_path + '/' + name_UCR_dataset
                
            X_train, y_train = load_from_tsfile_to_dataframe(ts_file_path + '_TRAIN.ts')
            X_test, y_test = load_from_tsfile_to_dataframe(ts_file_path + '_TEST.ts')
            name = ts_file_path.split('/')[-1]
        else: 
            X, y = load_from_tsfile_to_dataframe(ts_file_path)

            kwargs = dict(test_size=0.2, random_state=1)
            X_train, X_test, y_train, y_test = train_test_split(X, y, **kwargs)

            if name is None:
                name = ts_file_path.split('.')[0].split('/')[-1]

        self.transform_train_path = module_path + f'/features/extracted_features_ts_files/{name}_preproject_TRAIN.ts'
        self.transform_test_path = module_path + f'/features/extracted_features_ts_files/{name}_preproject_TEST.ts'

        if isinstance(y_test, np.ndarray):
            y_test = pd.DataFrame(y_test)
            y_train = pd.DataFrame(y_train)


        X = pd.concat([X_test.assign(ind="test"), X_train.assign(ind="train")])
        y = pd.concat([y_test.assign(ind="test"), y_train.assign(ind="train")])
        X, y = shuffle(X,y, random_state = 0)
    
        self.X = X
        self.y = y
        
        
        self.sr = sr
        self.n_mfcc = n_mfcc
        
        if os.path.exists(self.transform_train_path) and not(update):
            self.X_transform, self.y_transform = self.extract_existing_transformed()
        else:
            if denoise: 
                self.X['dim_0'] = self.X['dim_0'].apply(lambda x: utility.denoise_audio(x))
            if downsample:
                self.X['dim_0'] = self.X['dim_0'].apply(lambda x: utility.downsample(x, self.sr, downsample_new_sr))

            self.X_transform, self.y_transform = self.transform()
            self.save_transform()

        
    def get_X_y(self, train_test_split = False):
        X = self.X
        y = self.y
        X = X.rename(columns={'dim_0': 'time series'})
        y = y.rename(columns={0: 'label'})
        if train_test_split:
            X_test, X_train = X[X["ind"].eq("test")], X[X["ind"].eq("train")]
            y_test, y_train = y[y["ind"].eq("test")], y[y["ind"].eq("train")]

            X_test, X_train = X_test.drop(columns = 'ind'), X_train.drop(columns = 'ind')
            y_test, y_train = y_test.drop(columns = 'ind'), y_train.drop(columns = 'ind')
            X_train = X_train.reset_index(drop = True)
            X_test = X_test.reset_index(drop = True)
            return X_train, X_test, y_train, y_test
        else:
            X = X.reset_index(drop = True)
            return X.drop(columns = 'ind'), y.drop(columns = 'ind')
    
    def get_X_y_transformed(self, train_test_split = False, normalize = True):
        X = self.X_transform
        y = self.y_transform  

        if train_test_split:
            X_test, X_train = X[X["ind"].eq("test")], X[X["ind"].eq("train")]
            y_test, y_train = y[y["ind"].eq("test")], y[y["ind"].eq("train")]

            X_test, X_train = X_test.drop(columns = 'ind'), X_train.drop(columns = 'ind')
            y_test, y_train = y_test.drop(columns = 'ind'), y_train.drop(columns = 'ind')

            X_train.columns = np.arange(len(X_train.columns))
            X_test.columns = np.arange(len(X_test.columns))

            y_train, y_test = y_train.squeeze(), y_test.squeeze()

            if normalize:
                scaler = MinMaxScaler()
                scaler.fit(X_train)
                X_train = pd.DataFrame(scaler.transform(X_train))
                X_test = pd.DataFrame(scaler.transform(X_test))
            
            X_train = X_train.reset_index(drop = True)
            X_test = X_test.reset_index(drop = True)

            return X_train, X_test, y_train, y_test
       
        X = X.drop(columns = 'ind')
        y = y.drop(columns = 'ind')

        X = X.reset_index(drop = True)
        X.columns = np.arange(len(X.columns))
        y = y.squeeze()

        if normalize:
            scaler = MinMaxScaler()
            scaler.fit(X)
            X = pd.DataFrame(scaler.transform(X))
        return X, y


    def get_features(self, data):
        if isinstance(data, pd.Series):
            data = data.to_numpy()
        sr = self.sr
        n_mfcc = self.n_mfcc
        ft1 = librosa.feature.mfcc(data, sr = sr, n_mfcc=n_mfcc)
        ft2 = librosa.feature.zero_crossing_rate(data)[0]
        ft3 = librosa.feature.spectral_rolloff(data)[0]
        ft4 = librosa.feature.spectral_centroid(data)[0]
        ft5 = librosa.feature.spectral_contrast(data)[0]
        ft6 = librosa.feature.spectral_bandwidth(data)[0]

        ### Get HOS and simple features 
        ft0_trunc = np.hstack((np.mean(data) , np.std(data), skew(data), np.max(data), np.median(data), np.min(data), utility.get_energy(data), utility.get_entropy(data)))

        ### MFCC features
        ft1_trunc = np.hstack((np.mean(ft1, axis=1), np.std(ft1, axis=1), skew(ft1, axis = 1), np.max(ft1, axis = 1), np.median(ft1, axis = 1), np.min(ft1, axis = 1)))

        ### Spectral Features 
        ft2_trunc = np.hstack((np.mean(ft2), np.std(ft2), skew(ft2), np.max(ft2), np.median(ft2), np.min(ft2)))
        ft3_trunc = np.hstack((np.mean(ft3), np.std(ft3), skew(ft3), np.max(ft3), np.median(ft3), np.min(ft3)))
        ft4_trunc = np.hstack((np.mean(ft4), np.std(ft4), skew(ft4), np.max(ft4), np.median(ft4), np.min(ft4)))
        ft5_trunc = np.hstack((np.mean(ft5), np.std(ft5), skew(ft5), np.max(ft5), np.median(ft5), np.min(ft5)))
        ft6_trunc = np.hstack((np.mean(ft6), np.std(ft6), skew(ft6), np.max(ft6), np.median(ft6), np.max(ft6)))
        return pd.Series(np.hstack((ft0_trunc , ft1_trunc, ft2_trunc, ft3_trunc, ft4_trunc, ft5_trunc, ft6_trunc)))
              
    
    def transform(self):
        X_copy = self.X.copy()
        X = X_copy['dim_0'].apply(lambda x: self.get_features(x))
        X['ind'] = X_copy['ind']


        y_copy = self.y.copy()
        y_copy = y_copy.reset_index(drop = True)
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y_copy[0])
        y = pd.DataFrame(y)
        
        y['ind'] = y_copy['ind']

        return X, y

    def save_transform(self):
        X = self.X_transform
        y = self.y_transform
        
        X_test, X_train = X[X["ind"].eq("test")], X[X["ind"].eq("train")]
        y_test, y_train = y[y["ind"].eq("test")], y[y["ind"].eq("train")]
        
        X_test, X_train = X_test.drop(columns = 'ind'), X_train.drop(columns = 'ind')
        y_test, y_train = y_test.drop(columns = 'ind'), y_train.drop(columns = 'ind')
        
        X_train, X_test = X_train.reset_index(drop = True), X_test.reset_index(drop = True)
        y_train, y_test = y_train.reset_index(drop = True), y_test.reset_index(drop = True)


        utility.write_to_ts(self.transform_train_path, X_train, y_train[0].to_numpy().astype(str))
        utility.write_to_ts(self.transform_test_path, X_test, y_test[0].to_numpy().astype(str))
    
    def extract_existing_transformed(self):
        X_train, y_train = load_from_tsfile_to_dataframe(self.transform_train_path)
        X_test, y_test = load_from_tsfile_to_dataframe(self.transform_test_path)

        y_train = pd.DataFrame(y_train)
        y_test = pd.DataFrame(y_test)
        
        X_copy = pd.concat([X_test.assign(ind="test"), X_train.assign(ind="train")])
        y = pd.concat([y_test.assign(ind="test"), y_train.assign(ind="train")])
        
        X = from_nested_to_2d_array(X_copy['dim_0'])
        X['ind'] = X_copy['ind']
        
        return X,y
