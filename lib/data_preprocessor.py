import pickle
import numpy as np
import tensorflow as tf
import pandas as pd
import random

from sklearn.preprocessing import MinMaxScaler


class DataPreprocessor:
    def __init__(self, window_size):
        self._scaler = None
        self._window_size = window_size

    def _load_pickle(self, filepath):
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                return data
        except Exception as e:
            raise Exception('Failed to load data from file {}\n\
                            {}'.format(filepath, e))

    def load_data(self, datapath):
        train_data = self._load_pickle('{}_train.pkl'.format(datapath))
        test_data = self._load_pickle('{}_test.pkl'.format(datapath))
        labels = self._load_pickle('{}_test_label.pkl'.format(datapath))

        # Fill NA with zero
        train_data = np.nan_to_num(train_data.astype('float32'))
        test_data = np.nan_to_num(test_data.astype('float32'))
        return train_data, test_data, labels

    def transform(self, data, build_scaler=False):
        if build_scaler:
            self._scaler = MinMaxScaler()
            scaled_data = self._scaler.fit_transform(data)
        else:
            if self._scaler is None:
                raise ValueError('Scaler has not been initialized yet, \
                                 please initialize it')
            scaled_data = self._scaler.transform(data)
        return scaled_data

    def train_val_split(self, data, validation_split):
        if validation_split <= 0 or validation_split >= 1:
            raise ErrorValue('Validation split is invalid {}\
                    it must be between 0 and 1'.format(validation_split))
        num_val_data = int(len(data) * validation_split)
        train_data = data[:-num_val_data]
        val_data = data[-num_val_data:]
        return train_data, val_data

    def _time_window_sliding(self, data, step=1):
        """Short summary.
        Parameters
        ----------
        data : DataFrame or ndarray or list
            target data
        window_size : int
            a window size for time sliding
        step : int
            step length among time windows
        Returns
        -------
        ndarray
            ex) data : (N, features)
                returns : (N - window_size, window_size, features)
        """
        data_type = type(data)
        if data_type == pd.DataFrame or data_type == list:
            data = np.array(data)
        elif data_type == np.ndarray:
            pass
        else:
            raise TypeError(
                'time_window_sliding only supports array-like of shape (n_samples, n_features), but data type is %s' % (
                    data_type))
        result = []
        for i in range(0, len(data) - self._window_size + 1, step):
            result.append(data[i:i + self._window_size, :])
        return np.array(result)

    def generate_sliding_data(self, data, batch_size=None, shuffle=True):
        sliding_data = self._time_window_sliding(data)
        num_data = len(sliding_data)
        print('num data', num_data)
        if shuffle:
            random.shuffle(sliding_data)
        if batch_size:
            sliding_data = tf.data.Dataset.from_tensor_slices(sliding_data).batch(batch_size)
        return sliding_data, num_data