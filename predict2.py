# chainer 1.24.0
import numpy as np
import argparse
import os
import copy
import matplotlib.pyplot as plt
import alpha_finder
from fractions import Fraction
from scipy.signal import resample_poly
from sklearn import preprocessing
from sklearn.model_selection import RandomizedSearchCV, KFold, TimeSeriesSplit
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from RNN import LSTM
from MLP import MLP
import pickle


def resample(x, sr1=25, sr2=125):
    '''If need to resample x'''
    a, b = Fraction(sr1, sr2)._numerator, Fraction(sr1, sr2)._denominator
    return resample_poly(x, a, b, axis=0).astype(np.float32)


def regress_z(x, z):
	'''If need to regress third variable form x'''
    c = np.linalg.lstsq(np.c_[z, np.ones(z.shape[0])], x)[0]
    r = x - np.c_[z, np.ones(z.shape[0])].dot(c)
    return r.astype(np.float32)


def cv_split(x, t, k=10, shuffle=True):
    # kf = TimeSeriesSplit(n_splits=k) if cv_time_series else KFold(n_splits=k)
    kf = KFold(n_splits=k, shuffle=shuffle)
    Train, Test = [], []
    for i_train, i_test in kf.split(x):
        Train.append([x[i_train], t[i_train]])
        Test.append([x[i_test], t[i_test]])
        # break
    return Train, Test


def standardize(ktrain, ktest, norm_x=True):
    x_scaler = preprocessing.StandardScaler().fit(ktrain[0])
    t_scaler = preprocessing.StandardScaler().fit(ktrain[1])
    if norm_x:
        return [x_scaler.transform(ktrain[0]), t_scaler.transform(ktrain[1])],\
                   [x_scaler.transform(ktest[0]), t_scaler.transform(ktest[1])]
    else:
        return [ktrain[0], t_scaler.transform(ktrain[1])], \
                            [ktest[0], t_scaler.transform(ktest[1])]


def cv_alpha(model, xtrain, ttrain):
    param_dist = {"alpha": np.arange(0, 100000, 100)}
    n_iter_search = 10
    random_search = RandomizedSearchCV(model, param_distributions=param_dist,
                                       n_iter=n_iter_search)
    random_search.fit(xtrain, ttrain)
    return random_search.cv_results_['param_alpha'][random_search.cv_results_['rank_test_score'] == 1].data[0]


def initialize_model(args):
    if args.model == 'ridge':
        return Ridge()
    elif args.model == 'mlp':
        return MLP(n_input=x.shape[-1], n_mid=args.n_mid, lr=args.lr, n_output=t.shape[-1], batch_size=500,
                    shuffle_batches=args.shuffle, drop=args.drop, n_epochs=args.n_epochs, toplot=False)
    elif args.model == 'lstm':
        return LSTM(n_input=x.shape[-1], n_mid=args.n_mid, lr=args.lr, w_decay=0., n_output=t.shape[-1],
                      batch_size=200, n_back=args.n_back, drop=args.drop, n_epochs=args.n_epochs, toplot=False)
    else:
        raise Exception('Unknown model')


##
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--model', type=str, default='ridge')
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--norm_x', type=bool, default=True)
    parser.add_argument('--n_mid', type=int, default=100)
    parser.add_argument('--drop', type=float, default=0.1)
    parser.add_argument('--n_back', type=int, default=5)
    parser.add_argument('--lr', type=int, default=1e-03)
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--out_file', type=str)
    args = parser.parse_args()

    t = np.load(args.output)
    t = resample(t)[:-1, :]
    x = np.load(args.input)
    x = x.reshape((x.shape[0], -1))

    Train, Test = cv_split(x, t, k=10, shuffle=args.shuffle)
    R = []

    for k in range(len(Train)):
        print('Fold ' + str(k) + ':')

        ktrain, ktest = standardize(Train[k], Test[k], norm_x=args.norm_x)
        model = initialize_model(args)
        if args.model == 'ridge': model.alpha = cv_alpha(model, ktrain[0], ktrain[1])
        model.fit(ktrain[0], ktrain[1])
        y_hat = model.predict(ktest[0])

        print('\t\t'+str(mean_squared_error(ktest[1], y_hat)))
        print('\t\t'+str(r2_score(ktest[1], y_hat)))
        print('\t\t'+str(explained_variance_score(ktest[1], y_hat)))
        R.append(np.array([np.corrcoef(ktest[1][:, i], y_hat[:, i])[0, 1] for i in range(y_hat.shape[1])]))
        print('\t\tMax test cor: ' + str(np.max(R[k])))


    r = np.mean(np.array(R), axis=0)

    ##
    np.save('../results/' + args.out_file, r)
    # pickle.dump(model, open('../results/' + args.out_file + '_model.p', 'wb'))
