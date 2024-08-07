import os
import sys
import json
import pickle

HOME = os.path.split(os.path.abspath(__file__))[0]
sys.path.append(HOME + '/lib')

from tensorflow.keras.callbacks import EarlyStopping

from utils import get_best_f1, get_data_dim, get_data_name
from data_preprocessor import DataPreprocessor
from mstvae_model import AverageLossCallback, MSTVAEModel
from optimizer import create_optimizer

from tensorflow.keras.models import model_from_json

def load_train_config(filepath):
    try:
        with open(filepath, 'r') as f:
            config = json.load(f)
            return config
    except Exception as e:
        raise Exception('Failed to load config from file {}\n{}'.format(filepath, e))

def run_experiment():
    # -------------------------------CONFIG------------------------------------
    use_config_from_file = False
    config = {
        'dataset': 'machine-2-1',
        'z_dim': 8,
        'x_dim': 19,
        'logstd_min': -5,
        'logstd_max': 2,
        'output_shapes': [15, 30],
        'l2_reg': 0.0001,
        'window_size': 30,
        'n_samples': 100,
        'learning_rate': 0.001,
        'lr_anneal_factor': 0.5,
        'lr_anneal_epoch_freq': 30,
        'n_mc_chain': 10,
        'mcmc_iter': 10,
        'num_epochs': 1,
        'validation_split': 0.3,
        'train_batch_size': 100,
        'test_batch_size': 50
    }

    # Load train configuration from train_config.json
    if use_config_from_file:
        dataname = get_data_name(config['dataset'])
        cfg_train = load_train_config('train_config.json')
        config.update(cfg_train[dataname])

    #-------------------------------PREPARE DATA-------------------------------
    datapath = 'data/processed/' + config['dataset']
    # Update x_dim in `config`
    config['x_dim'] = get_data_dim(config['dataset'])

    data_preprocessor = DataPreprocessor(config['window_size'])
    train_data, test_data, test_label = data_preprocessor.load_data(datapath)
    scaled_train_data = data_preprocessor.transform(train_data, build_scaler=True)
    scaled_test_data = data_preprocessor.transform(test_data)
    scaled_train_data, scaled_val_data = data_preprocessor.train_val_split(scaled_train_data,
                                                                           validation_split=config['validation_split'])

    sliding_train_data, num_train = data_preprocessor.generate_sliding_data(scaled_train_data,
                                                                 batch_size=config['train_batch_size'],
                                                                 shuffle=True)
    sliding_val_data, num_val = data_preprocessor.generate_sliding_data(scaled_val_data,
                                                               batch_size=config['train_batch_size'],
                                                               shuffle=False)
    sliding_test_data, num_test = data_preprocessor.generate_sliding_data(scaled_test_data,
                                                                batch_size=config['test_batch_size'],
                                                                shuffle=False)

    # -------------------------------MODEL-------------------------------
    detector = MSTVAEModel(config)
    #
    optimizer = create_optimizer(num_X_train=num_train,
                                 batch_size=config['train_batch_size'],
                                 epochs=config['num_epochs'])
    detector.compile(optimizer=optimizer)

    # -------------------------------TRAINING-------------------------------
    # Create callbacks
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=10,
                                   restore_best_weights=True)

    detector.fit(x=sliding_train_data,
                   validation_data=sliding_val_data,
                   epochs=config['num_epochs'],
                   callbacks=[AverageLossCallback(), early_stopping])

    # Save model
    model_json = detector.to_json()
    model_json = json.loads(model_json)
    with open('saved_model/mstvae.json', 'w') as f:
        json.dump(model_json, f, indent=4)
    detector.save_weights('saved_model/mstvae.h5')

    # Load model
    with open('saved_model/mstvae.json', 'r') as f:
        model_json = f.read()
    detector = model_from_json(model_json, custom_objects={'MSTVAEModel': MSTVAEModel})
    detector.load_weights('saved_model/mstvae.h5')


    # # -------------------------------TESTING-------------------------------
    # print('\n\n-------------------EVALUATE THE MODEL-------------------')
    recons_values, anomaly_scores = detector.calculate_anomaly_scores(sliding_test_data,
                                                              get_last_obser=True)

    # Get best f1 score
    test_labels = test_label[config['window_size'] - 1:]
    results = get_best_f1(anomaly_scores, test_labels)

    best_f1 = results[0][0]
    precision = results[0][1]
    recall = results[0][2]
    TP = results[0][3]
    TN = results[0][4]
    FP = results[0][5]
    FN = results[0][6]
    threshold = results[1]

    print('Evaluation results: \nBest F1: {}\nPrecision: {}\nRecall: {}\nTP: {}\nTN: {}\nFP: {}\nFN: {}\n'
          'threshold: {}'.format(best_f1, precision, recall, TP, TN, FP, FN, threshold))


if __name__ == '__main__':
    run_experiment()