from matplotlib import pyplot as plt
import os
import h5py
import pandas as pd
import numpy as np
import healpy as hp
from sklearn.model_selection import (
    cross_validate as _cross_validate, train_test_split
)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error as mse, make_scorer
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models

from pickle import dumps
import dill
import joblib

from pdb import set_trace

def read_data(file_name='3GeV-det1', nmuons=-1, detector_config='1',
              energy_range='3GeV', verbose=False, key_name=''):
    if int(detector_config) == 1:
        ''' colname = ['xpos', 'ypos', 'Alt', 'Az', 'xpos2', 'ypos2',
                     'xpos3', 'ypos3', 'SiPM1', 'SiPM2', 'SiPM3', 'SiPM4',
                     'SiPM5', 'SiPM6', 'SiPM7', 'SiPM8', 'SiPM9', 'SiPM10',
                     'SiPM11', 'SiPM12', 'SiPM13', 'SiPM14', 'SiPM15',
                     'SiPM16', 'SiPM17', 'SiPM18', 'SiPM19', 'SiPM20',
                     'SiPM21', 'SiPM22', 'SiPM23', 'SiPM24']'''

        colname = ['xpos', 'ypos', 'Alt', 'Az', 'SiPM1', 'SiPM2', 'SiPM3',
                   'SiPM4','SiPM5','SiPM6','SiPM7','SiPM8','SiPM9','SiPM10',
                   'SiPM11','SiPM12','SiPM13','SiPM14','SiPM15']

        if energy_range == '3GeV':
            keyname = '3 GeV Muon Info'
        elif energy_range == '100MeV':
            keyname = '100 MeV Muon Info'
        elif energy_range == '10MeV':
            keyname = '10 MeV Muon Info'
        else:
            print("Does this detector have the requested energy range?")
        
        num_responses = 4

    elif int(detector_config) == 2:
        colname = ['xpos', 'ypos', 'Alt', 'Az', 'xpos2', 'ypos2', 'xpos3',
                   'ypos3', 'SiPM1', 'SiPM2', 'SiPM3', 'SiPM4', 'SiPM5',
                   'SiPM6', 'SiPM7', 'SiPM8', 'SiPM9', 'SiPM10', 'SiPM11',
                   'SiPM12', 'SiPM13', 'SiPM14', 'SiPM15', 'SiPM16', 'SiPM17',
                   'SiPM18', 'SiPM19', 'SiPM20', 'SiPM21', 'SiPM22', 'SiPM23',
                   'SiPM24']

        if energy_range == '3GeV':
            keyname = '3 GeV Muon Info'
            # keyname = '10 MeV Muon Info' # Typo in the file naming
            # keyname = '10 GeV Muon Info' # Typo made by gov while creating shuffled h5
        else:
            print("Does this detector have the requested energy range?")

        num_responses = 8

    elif int(detector_config) == 3:
        colname = ['xpos', 'ypos', 'Alt', 'Az', 'xpos2', 'ypos2', 'xpos3',
                   'ypos3', 'SiPM1', 'SiPM2', 'SiPM3', 'SiPM4', 'SiPM5',
                   'SiPM6', 'SiPM7', 'SiPM8', 'SiPM9', 'SiPM10', 'SiPM11',
                   'SiPM12', 'SiPM13', 'SiPM14', 'SiPM15', 'SiPM16', 'SiPM17',
                   'SiPM18', 'SiPM19', 'SiPM20', 'SiPM21', 'SiPM22', 'SiPM23',
                   'SiPM24'] 

        if energy_range == '3GeV':
            keyname = '3 GeV Muon Info'
        else:
            print('Does this detector have the requested energy range?')
        file_name += '-X0comparison'
    
        # TODO
        num_responses = None

    else:
        print('What detector did you pick here? I have been passed: ',
              detector_config)


    if verbose:
        print('Open file', file_name)
        # Xpos Slab1 (mm); Ypos Slab1(mm); Zenith(deg); Azimuth(deg);
        # Photomultiplier 1; PM 2; PM3; ..... PM15.

        with h5py.File(file_name+'.h5', 'r') as f:
            print(f.keys())

    with h5py.File(file_name + '.h5', 'r') as f:
        data = pd.DataFrame(f[keyname][0:nmuons, :], columns=colname)

    # Gov (don't shuffle everytime, just shuffle once and save as shuffled.h5)
    # data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    return data, num_responses

def preprocess_x(X):
    # channels 9, 11, 13, and 15 in det-2 have NaN
    X = X.fillna(0)
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(X), columns=X.columns), scaler

def preprocess(data, unitcircle=True, channels=[
              'SiPM1', 'SiPM2', 'SiPM3', 'SiPM4', 'SiPM5', 'SiPM6',
              'SiPM7', 'SiPM8', 'SiPM9', 'SiPM10', 'SiPM11', 'SiPM12',
              'SiPM13', 'SiPM14', 'SiPM15'
]):
    # X is data features
    X = data[channels].copy()

    X, scaler = preprocess_x(X)

    # y is target
    y = data[['Az','Alt']].copy()

    # for zero to be just greater to limit the model overshooting towards
    # unphysical negative numbers
    y.loc[y.Alt <= 0., 'Alt'] = 1e-32
    y.loc[y.Az <= 0., 'Az'] = 1e-32

    # Azimuth was incorrectly defined, it is the rotation of the model, 
    # not the angle FROM WHICH the muons arrived, meaning determining the
    # azimuth is -1 * azimuth, or equivalenty Az = 360. - Az 
    y.Az = 360. - y.Az
    
    return X, y, scaler

def transform(y, cfg={'coord': 'HEALPix', 'NSIDE': 128, 'nest': True}):
    if cfg['coord'] == 'HEALPix':
        alt = y['Alt'] * np.pi / 180
        az = y['Az'] * np.pi / 180
        y = hp.ang2pix(cfg['NSIDE'], list(alt), list(az), nest=cfg['nest'])
        y = hp.pix2vec(cfg['NSIDE'], y, nest=cfg['nest'])    

    elif cfg['coord'] == 'unitcircle':
        y['Az_cos'], y['Az_sin'] = fromdeg(y.Az)
        # Gov (only one component needed for Alt)
        y['Alt_sin'] = project(y.Alt)

        y.Az_sin.hist()
        y.Alt.hist()

        y.drop('Az', axis=1, inplace=True)
        y.drop('Alt', axis=1, inplace=True)

    else:
        y.Az /= norm_az
        y.Alt /= norm_alt

    return y

def detransform(y, cfg={ 'coord': 'HEALPix', 'NSIDE': 128, 'nest': True }):
    if cfg['coord'] == 'HEALPix':
        y = hp.vec2pix(cfg['NSIDE'], y[0], y[1], y[2], nest=cfg['nest'])    
        y = hp.pix2ang(cfg['NSIDE'], y, nest=cfg['nest'])
        alt = y[0] * 180 / np.pi
        az = y[1] * 180 / np.pi

        return np.stack([az, alt]).T
    else:
        az_pred = todeg(y_pred[:, :2].T)
        alt_pred = unproject(y_pred[:, 2].T)

        az_test = todeg(y_test[:, :2].T)
        alt_test = unproject(y_test[:, 2].T)

        return np.stack([az_pred, alt_pred]).T, np.stack([az_test, alt_test]).T

def fromdeg(d):
    r = d * np.pi / 180.
    return np.array([np.cos(r), np.sin(r)])

def todeg(r):
    if len(r.shape) > 1:
        d = np.arctan2(r[1,:], r[0,:]) * 180. / np.pi
        d[d < 0.] = 360. + d[d < 0.]
    else:
        d = np.arctan2(r[1], r[0]) * 180. / np.pi
        if d < 0.:
            d = 360. + d
    return d

def project(d):
    r = d * np.pi / 180
    return np.sin(r)

def unproject(r):
    return (np.arcsin(r) * 180 / np.pi) % 360

def plot(x, y, path='path'):
    plt.clf()

    lines = plt.plot(x, y)

    if os.path.sep in path:
        directory = os.path.sep.join(path.split('/')[:-1])
        if not os.path.exists(directory):
            os.makedirs(directory)

    lines[0].get_figure().savefig(path)

def wrap_diff(test, preds):
    diff = abs(np.array(test - preds))

    return np.array([-a % 360 if a > 180 else a for a in diff])

def get_rmse(test, preds):
    return np.sqrt(mse(test, preds))

def get_wrapped_rmse(test, preds, diff_fn=wrap_diff):
    assert test.shape == preds.shape, 'Input shapes are not equal!'

    dims = test.shape[1]
    _sum = 0

    for dim in range(dims):
        _sum += np.mean(diff_fn(test.T[dim], preds.T[dim]) ** 2)

    return np.sqrt(_sum / dims)

def get_av_rmse(rmses):
    return np.sqrt(sum([a ** 2 for a in rmses]) / len(rmses))

def loss_fn(y_true, y_pred, **kwargs):
    return get_rmse(y_true, y_pred)

def get_residuals(y_true, y_pred, angular=False, **kwargs):
    assert len(y_true) == len(y_pred)

    num_cols = y_true.shape[1]
    residuals = []
    av_residuals = []    

    for i in range(len(y_true)):
        residual = abs(y_true.iloc[i] - y_pred[i])
        residuals.append(residual)
        av_residuals.append(np.mean(residual))

    return residuals, av_residuals

def cross_validate(model, X, y, n_splits=10,
                   cfg={'coord': 'HEALPix', 'NSIDE': 4, 'nest': True},
                   **kwargs):
    rmses = []
   
    results = _cross_validate(model, X, y,
                              scoring=make_scorer(loss_fn), **kwargs)
    
    return get_av_rmse(results['test_score'])

def fit_predict(model, X, y, **kwargs):
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=42,
                                                        **kwargs)
    
    model.fit(X_train, y_train)
    return X_test, y_test, X_train, y_train, model.predict(X_test), model

def get_sorted_residuals(y_test_, y_pred, num_channels=15, scaler=None):
    set_trace()
    residuals, av_residuals = get_residuals(y_test, y_pred)

    X_test.loc[:, 'av_residual'] = av_residuals
    X_test.loc[:, 'residual_az_1'] = np.array(residuals).T[0]
    X_test.loc[:, 'residual_az_2'] = np.array(residuals).T[1]
    X_test.loc[:, 'residual_alt'] = np.array(residuals).T[2]

    rmse = get_rmse(y_test, y_pred)

    results = X_test.sort_values('av_residual')
    
    if scaler is not None:
        results.reset_index(drop=True, inplace=True)
        raw = pd.DataFrame(scaler.inverse_transform(
                               results.iloc[:, :num_channels]
                           ),
                           columns=X_test.columns)
        results = pd.concat([raw, results.iloc[:, num_channels:]])
    

    return results, rmse

def get_cnn(cfg, input_shape=(3, 3, 3)):
    model = models.Sequential()
    model.add(layers.Conv2D(32, cfg['model']['kernel_size'],
                            activation=cfg['model']['act'],
                            input_shape=input_shape,
                            padding='same',
                            kernel_initializer=cfg['model']['init']))
    model.add(layers.Conv2D(64, cfg['model']['kernel_size'],
                            activation=cfg['model']['act'], padding='same',
                            kernel_initializer=cfg['model']['init']))
    model.add(layers.Conv2D(128, cfg['model']['kernel_size'],
                            activation=cfg['model']['act'], padding='same',
                            kernel_initializer=cfg['model']['init']))
    model.add(layers.Conv2D(256, cfg['model']['kernel_size'],
                            activation=cfg['model']['act'], padding='same',
                            kernel_initializer=cfg['model']['init']))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation=cfg['model']['act']))
    model.add(layers.Dense(16, activation=cfg['model']['act']))
    model.add(layers.Dense(3))

    model.compile(optimizer=cfg['model']['opt'], loss=cfg['model']['loss'])

    return model

def reshape(X, cfg={ 'num_channels': 24 }):
    if cfg['num_channels'] == 24:
        X.insert(4, 0, 0)
        X.insert(13, 1, 0)
        X.insert(22, 2, 0)
    
        cols = [
            'SiPM1', 'SiPM2', 'SiPM3', 'SiPM8', 0,
            'SiPM4', 'SiPM7', 'SiPM6', 'SiPM5',
            'SiPM9', 'SiPM10', 'SiPM11', 'SiPM16', 1,
            'SiPM12', 'SiPM15', 'SiPM14', 'SiPM13',
            'SiPM17', 'SiPM18', 'SiPM19', 'SiPM24', 2,
            'SiPM20', 'SiPM23', 'SiPM22', 'SiPM21'
        ]

        return np.array(X[cols]).reshape((X.shape[0], 3, 3, 3))
    
    if cfg['num_channels'] == 12:
        cols = [
            'SiPM1', 'SiPM3', 'SiPM7', 'SiPM5',
            'SiPM9', 'SiPM11', 'SiPM15', 'SiPM13',
            'SiPM17', 'SiPM19', 'SiPM23', 'SiPM21'
        ]
        
        return np.array(X[cols]).reshape((X.shape[0], 3, 2, 2))

    if cfg['num_channels'] == 8:
        cols = [
            'SiPM1', 'SiPM3', 'SiPM7', 'SiPM5',
            'SiPM17', 'SiPM19', 'SiPM23', 'SiPM21'
        ]

        return np.array(X[cols]).reshape((X.shape[0], 2, 2, 2))

    if cfg['num_channels'] == 15:
        pass #TODO

def cfg_to_str(cfg):
    return str(cfg).replace(' ', '').replace('{', '').replace('\'', '').replace('}', '')

def create_path(path):
    directory = os.path.sep.join(path.split('/')[:-1])
    if not os.path.exists(directory):
        os.makedirs(directory)
    
def write_results(path, data):
    create_path(path)

    with open(path, 'wb') as f:
        f.write(dumps(data))

def dump_model(path, model):
    from keras_pickle_wrapper import KerasPickleWrapper

    create_path(path)

    # models.save_model(model, path)
    with open(path, 'wb') as f:
        f.write(dumps(model))

def dump(path, data):
    create_path(path)

    with open(path, 'wb') as f:
        f.write(dill.dumps(data))

#def write_results(X, rmse

# log transform?
# for col in X:
    # X[col] = X[col].apply(lambda a: log(1e-4) if a == 0 else log(a))
