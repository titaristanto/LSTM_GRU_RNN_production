from __future__ import division
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import time, os, math, warnings
from matplotlib import pyplot

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import LSTM, GRU, SimpleRNN


"""
This file contains various functions related to the data-driven prediction in multiphase 
reservoir flow problems. More specifically, given flow rates and pressure only, without knowing 
the physical parameters of the well and reservoir, we want to see how good machine/deep learning
can be in predicting future events.

The scope of this project includes:
- Performance of 3 deep learning algorithms in solving pressure and flow rate time-series problem. 
- How complexities (noise, heterogeneity, anisotropy, etc.) affect the result of the ML prediction (on-going)
"""

def load_data(filename):
    """
    This function reads csv file from the given url and extracts time-series data of oil & water rate,
    pressure, and water cut.
    :param filename: the url of csv file (stored in GitHub)
    :return: t: vector of time; qo: matrix of oil rate in all wells; qo: matrix of water rate in all wells;
            and p: matrix of bottom hole pressure in all wells
    """
    df = pd.read_csv(filename)

    t = df.loc[:, ['TIME']] # Time in simulation: DAY
    t *= 24 # Converting time from DAY to HOUR
    qo = df.loc[:, ['WOPR:P1', 'WOPR:P2', 'WOPR:P3']]
    qw = df.loc[:, ['WWPR:P1', 'WWPR:P2', 'WWPR:P3']]
    p = df.loc[:, ['WBHP:P1', 'WBHP:P2', 'WBHP:P3']]
    wc = df.loc[:, ['WWCT:P1', 'WWCT:P2', 'WWCT:P3']]
    return t, qo, qw, wc, p

def plot_pressure(t, p_actual, p_pred, title, color):
    """This function plots actual and predicted bottom hole pressure"""
    # Plotting pwf v time
    plt.plot(t, p_actual, 'k-', linewidth=3, label='Actual Pwf')

    if title=='Training Data':
        plt.plot(t[0:int(0.7*p_pred.shape[0])], p_pred[0:int(0.7*p_pred.shape[0])], 'rx', markeredgecolor=color, label=title)
        plt.plot(t[int(0.7*p_pred.shape[0]):], p_pred[int(0.7*p_pred.shape[0]):], 'yx', markeredgecolor='orange', label='Dev Set')
    else:
        plt.plot(t, p_pred, 'gx', markeredgecolor=color, label=title)
    plt.xlabel("Time (hours)")
    plt.ylabel("BH Pressure (psi)", fontsize=9)
    plt.title("BH Pressure Well A", y=1, fontsize=9)
    plt.legend(loc="best", prop=dict(size=8))
    plt.xlim(0, max(t))
    plt.ylim(0, max(max(p_actual.values), max(p_pred)))
    plt.grid(True)

def plot_pred_rate(t, q_actual, q_pred, title, color):
    """This function plots actual and predicted bottom hole pressure"""
    # Plotting pwf v time
    plt.figure()
    plt.plot(t, q_actual, 'k-', linewidth=3, label='Actual oil rate')

    if title=='Training Set':
        plt.plot(t[0:int(0.75 * q_pred.shape[0])], q_pred[0:int(0.75 * q_pred.shape[0])],
                 'r-', markeredgecolor=color, label=title, linewidth=3)
        plt.plot(t[int(0.75 * q_pred.shape[0]):], q_pred[int(0.75 * q_pred.shape[0]):],
                 'y-', markeredgecolor='orange', label='Dev Set', linewidth=3)
    else:
        plt.plot(t, q_pred, 'g-', markeredgecolor=color, label=title, linewidth=3)
    plt.xlabel("Time (hours)")
    plt.ylabel("Flow Rate (STB/d)", fontsize=9)
    plt.title("Flow Rate Well A", y=1, fontsize=9)
    plt.legend(loc="best", prop=dict(size=8))
    plt.xlim(0, max(t))
    plt.ylim(0, max(max(q_actual), max(q_pred)) + 10)
    plt.grid(True)

def plot_rates(t, q, wellname, color, label):
    """This function plots actual flow rates"""
    # Plotting Flow Rate v time
    plt.plot(t, q, color, linewidth=3, label=label)
    plt.xlabel("Time (hours)")
    plt.ylabel("Flow Rate (STB/D)", fontsize=9)
    plt.title('Flow Rate Well ' + wellname, y=0.82, fontsize=9)
    plt.legend(loc="best", prop=dict(size=8))
    plt.xlim(0, max(t))
    plt.ylim(0, max(q) + 10)
    plt.grid(True)

def plot_pressure_rates(x, y, y_pred, labelname):
    """This function plots both pressure and actual flow rates"""
    plt.figure()
    if labelname=='Training Data':
        color='red'
    else:
        color='green'

    plt.subplot(411)
    plot_pressure(x['TIME'], y, y_pred, labelname, color)

    plt.subplot(412)
    plot_rates(x['TIME'], x['WWPR:P1'], wellname='A', color='blue', label='Water Rate')
    plot_rates(x['TIME'], x['WOPR:P1'], wellname='A', color='green', label='Oil Rate')
    plt.ylim(0, max(max(x['WWPR:P1'].values), max(x['WOPR:P1'].values)) + 10)

    plt.subplot(413)
    plot_rates(x['TIME'], x['WOPR:P2'],wellname='B',color='green', label='Oil Rate')
    plot_rates(x['TIME'], x['WWPR:P2'],wellname='B',color='blue', label='Water Rate')
    plt.ylim(0, max(max(x['WWPR:P2'].values), max(x['WOPR:P2'].values)) + 10)

    plt.subplot(414)
    plot_rates(x['TIME'], x['WOPR:P3'], wellname='C', color='green', label='Oil Rate')
    plot_rates(x['TIME'], x['WWPR:P3'], wellname='C', color='blue', label='Water Rate')
    plt.ylim(0, max(max(x['WWPR:P3'].values), max(x['WOPR:P3'].values)) + 10)

    plt.subplots_adjust(top=1.5,bottom=0.2)

def run_model(x_train, y_train, x_dev, y_dev, epochs=500, batch_size=400, method='LSTM'):
    classifier = {'GRU' : GRU,
                  'LSTM' : LSTM,
                  'Simple RNN' : SimpleRNN}

    model = Sequential()

    model.add(classifier[method](input_dim=10, output_dim=16,  return_sequences=True))
    model.add(Dropout(0.5))

    model.add(classifier[method](32, activation='tanh', return_sequences=False))
    model.add(Dropout(0.5))

    model.add(Dense(output_dim=1))
    model.add(Activation("linear"))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae','accuracy'])
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(x_dev, y_dev), verbose=0, shuffle=False)
    return model, history

def create_timeblock(X, Y, look_back=1):
    dataX, dataY = [], []
    for i in range(len(X) - look_back):
        a = X[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(Y[i + look_back, 0])
    print(len(dataY))
    return np.array(dataX), np.array(dataY)

def plot_loss(history, title):
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss', linewidth=3)
    plt.plot(history.history['val_loss'], label='Dev Loss', linewidth=3)
    plt.legend()
    plt.grid()
    plt.xlabel("No of Epochs", fontsize=9)
    plt.ylabel("Loss", fontsize=9)
    plt.title('Loss - ' + title, fontsize=9)
    plt.xlim(0, len(history.history['loss']))
    plt.ylim(0, max(history.history['val_loss']))
    plt.show()

def plot_loss_comparison(histories):
    plt.figure()
    plt.plot(histories['Simple RNN'].history['val_loss'], linewidth=3, label='Simple RNN')
    plt.plot(histories['GRU'].history['val_loss'], linewidth=3, label='GRU')
    plt.plot(histories['LSTM'].history['val_loss'], linewidth=3, label='LSTM')
    plt.legend()
    plt.grid()
    plt.xlabel("No of Epochs", fontsize=9)
    plt.ylabel("Loss", fontsize=9)
    plt.title(' Dev Loss', fontsize=9)
    plt.xlim(0, len(histories['Simple RNN'].history['loss']))
    plt.ylim(0, max(histories['Simple RNN'].history['val_loss']))
    plt.show()

def main():
    ### FLOW RATE PREDICTION ###
    # Load Training and Test Set
    t_train, qo_train, qw_train, wc_train, p_train = load_data('https://raw.githubusercontent.com/titaristanto/data-driven-production-problem/master/far_lowWC_training.csv')
    t_test, qo_test, qw_test, wc_test, p_test = load_data('https://raw.githubusercontent.com/titaristanto/data-driven-production-problem/master/far_lowWC_test.csv')

    X_train_raw, Y_train_raw = pd.concat([p_train['WBHP:P1']], axis=1, join='inner'), qo_train.loc[:, ['WOPR:P1']]
    X_test_raw, Y_test = pd.concat([p_test['WBHP:P1']], axis=1, join='inner'), qo_test.loc[:, ['WOPR:P1']]

    # Data Scaling
    scaler_x = MinMaxScaler()
    X_train_norm = scaler_x.fit_transform(X_train_raw.as_matrix())
    X_test_norm = scaler_x.transform(X_test_raw.as_matrix())
    scaler_y = MinMaxScaler()
    Y_train_norm = scaler_y.fit_transform(Y_train_raw.as_matrix())
    Y_test_norm = scaler_y.transform(Y_test.as_matrix())

    # Create time blocks
    X_train_block, Y_train = create_timeblock(X_train_norm, Y_train_norm, look_back=10)
    X_test_block, y_test = create_timeblock(X_test_norm, Y_test_norm, look_back=10)

    # Reshape input
    X_train = np.reshape(X_train_block, (X_train_block.shape[0], 1, X_train_block.shape[1]))
    x_test = np.reshape(X_test_block, (X_test_block.shape[0], 1, X_test_block.shape[1]))

    # Split train and dev set
    x_train, x_dev, y_train, y_dev = train_test_split(X_train,
                                                      Y_train,
                                                      test_size=0.25,
                                                      shuffle=False)

    # Run model and plot loss curve
    methods = ['Simple RNN', 'GRU', 'LSTM']
    dev_mse = {}
    dev_acc = {}
    train_mse = {}
    train_acc = {}
    test_mse = {}
    test_acc = {}
    histories = {}
    models = {}
    for i, method in enumerate(methods):
        models[method], histories[method] = run_model(x_train, y_train, x_dev, y_dev,
                                                      epochs=1000, batch_size=400, method=method)
        plot_loss(histories[method], title=method)

        # Predict oil rate in the training set
        yhat_train_scaled = models[method].predict(x_train)
        yhat_train = scaler_y.inverse_transform(yhat_train_scaled.reshape(-1, 1))

        # Predict oil rate in the dev set
        yhat_dev_scaled = models[method].predict(x_dev)
        yhat_dev = scaler_y.inverse_transform(yhat_dev_scaled.reshape(-1, 1))

        y_hat_traindev = np.concatenate([yhat_train, yhat_dev])

        # Plot oil rate - Train
        y_train_act = scaler_y.inverse_transform(y_train.reshape(-1, 1))
        y_dev_act = scaler_y.inverse_transform(y_dev.reshape(-1, 1))
        y_traindev_act = np.concatenate([y_train_act, y_dev_act])
        plot_pred_rate(t_train[:len(x_train)+len(x_dev)].as_matrix(),
                       y_traindev_act, y_hat_traindev, title='Training Set', color='red')

        # Predict oil rate in the test set
        yhat_scaled = models[method].predict(x_test)
        yhat_test = scaler_y.inverse_transform(yhat_scaled.reshape(-1, 1))

        # Plot oil rate - Test
        y_test_act = scaler_y.inverse_transform(y_test.reshape(-1, 1))
        plot_pred_rate(t_test[:len(x_test)].as_matrix(),
                       y_test_act, yhat, title='Test Set', color='green')

        train_mse[method], train_acc[method] = mean_squared_error(y_train_act, yhat_train), r2_score(y_train_act, yhat_train)
        dev_mse[method], dev_acc[method] = mean_squared_error(y_dev_act, yhat_dev), r2_score(y_dev_act, yhat_dev)
        test_mse[method], test_acc[method] = mean_squared_error(y_test_act, yhat_test), r2_score(y_test_act, yhat_test)

        print('Method: %s. Training Set Score: %1.4f' % (method, train_acc[method]))
        print('Method: %s. Dev Set Score: %1.4f' % (method, dev_acc[method]))
        print('Method: %s. Test Set Score: %1.4f' % (method, test_acc[method]))


if __name__ == '__main__':
    main()
