import os
import pandas as pd
from itertools import product

import utils
from predictors import (
        Static,
        NbitCounter,
        Bimodal,
        Correlation,
        Gshare,
        Perceptron,
        CNN
        )


trace = utils.read_data('trace.csv')

results = {}
plot = False
normalize = True

#%% Static Predictors
for always_taken in [True, False]:
    predictor = Static(always_taken=always_taken)
    y_pred = predictor.predict(trace['Branch'])
    results[predictor.name] = utils.evaluate(trace['Branch'], y_pred, name=predictor.name,
           plot=plot, normalize=normalize)
    print('\n', results[predictor.name])


#%% n-bit Counter
for n in [1, 2]:
    predictor = NbitCounter(n=n)
    y_pred = predictor.predict(trace['Branch'])
    results[predictor.name] = utils.evaluate(trace['Branch'], y_pred, name=predictor.name,
           plot=plot, normalize=normalize)
    print('\n', results[predictor.name])

#%% Grid search classical algorithms
k_bits, n_bits = [2, 8, 16], [2]

for k, n in product(k_bits, n_bits):
    
    # Bimodal
    predictor = Bimodal(k=k, n=n)
    
    y_pred = predictor.predict(trace['Branch'], trace['PC'])
    results[predictor.name] = utils.evaluate(trace['Branch'], y_pred, name=predictor.name,
           plot=plot, normalize=normalize)
    print('\n', results[predictor.name])
    
    # Correlation Based
    predictor = Correlation(k=k, n=n)
    
    y_pred = predictor.predict(trace['Branch'])
    results[predictor.name] = utils.evaluate(trace['Branch'], y_pred, name=predictor.name,
           plot=plot, normalize=normalize)
    print('\n', results[predictor.name])


    # Gshare
    predictor = Gshare(k=k, n=n)
    
    y_pred = predictor.predict(trace['Branch'], trace['PC'])
    results[predictor.name] = utils.evaluate(trace['Branch'], y_pred, name=predictor.name,
           plot=plot, normalize=normalize)
    print('\n', results[predictor.name])


#%% Best CNN model
predictor = CNN(
    history=21,
    num_hidden_layers=9,
    num_filters=32,
    kernel_size=3,
    activation='relu',
    dilation=1,
    skip=True
    )
predictor.load(os.path.join('best_models',
                            'CNN_history_21_hidden_layers_9_num_filters_32_kernel_size_3_skip_True_dilation_1_activation_relu_epochs_250',
                            'weights_247_0.2341.h5'))
predictor.model.summary()

y_pred, y_true = predictor.predict(trace['Branch'])
results[predictor.name] = utils.evaluate(y_true, y_pred, name='Convolutional Neural Network',
       plot=True, normalize=normalize)
print('\n', results[predictor.name])

#%% Best MLP model
predictor = Perceptron(
        history=21,
        num_hidden_layers=9,
        neurons_per_layer=32,
        activation='relu',
        )
predictor.load(os.path.join('best_models',
                            'MLP_history_21_hidden_layers_9_neurons_32_activation_relu_epochs_250',
                            'weights_250_0.2174.h5'))
predictor.model.summary()

y_pred, y_true = predictor.predict(trace['Branch'])
results[predictor.name] = utils.evaluate(y_true, y_pred, name='Multilayer Perceptron',
       plot=True, normalize=normalize)
print('\n', results[predictor.name])

#%% Output to csv
df = pd.DataFrame.from_dict(results, orient='index')
df.to_csv('algorithms_results.csv')

#%% Top Static
plot = True

predictor = Static(always_taken=False)
y_pred = predictor.predict(trace['Branch'])
results[predictor.name] = utils.evaluate(trace['Branch'], y_pred, name=predictor.name,
       plot=plot, normalize=normalize)
print('\n', results[predictor.name])

#%% Top n-bit counter
predictor = NbitCounter(n=2)
y_pred = predictor.predict(trace['Branch'])
results[predictor.name] = utils.evaluate(trace['Branch'], y_pred, name=predictor.name,
       plot=plot, normalize=normalize)
print('\n', results[predictor.name])


#%% Top Best Bimodal
predictor = Bimodal(k=16, n=2)

y_pred = predictor.predict(trace['Branch'], trace['PC'])
results[predictor.name] = utils.evaluate(trace['Branch'], y_pred, name=predictor.name,
       plot=plot, normalize=normalize)
print('\n', results[predictor.name])

#%% Top Correlation Based
predictor = Correlation(k=2, n=2)

y_pred = predictor.predict(trace['Branch'])
results[predictor.name] = utils.evaluate(trace['Branch'], y_pred, name=predictor.name,
       plot=plot, normalize=normalize)
print('\n', results[predictor.name])


#%% Top Gshare
predictor = Gshare(k=2, n=2)

y_pred = predictor.predict(trace['Branch'], trace['PC'])
results[predictor.name] = utils.evaluate(trace['Branch'], y_pred, name=predictor.name,
       plot=plot, normalize=normalize)
print('\n', results[predictor.name])


#%% Analysis of CNN grid search
paths = [p for p in os.listdir('logs') if 'CNN' in p]

for path in paths:
    weights = [f for f in os.listdir(os.path.join('logs', path)) if f.endswith('.h5')][0]
    print(os.path.join('logs', path, weights))
    
    # Need correct history for input shape
    history = int(path.split('_')[2])
    
    predictor = CNN(
        history=history,
        num_hidden_layers=3,
        num_filters=32,
        kernel_size=3,
        activation='relu',
        dilation=1,
        skip=True
        )
    predictor.load(os.path.join('logs', path, weights))
    
    predictor.name = path
    y_pred, y_true = predictor.predict(trace['Branch'])
    results[predictor.name] = utils.evaluate(y_true, y_pred, name='Convolutional Neural Network',
           plot=False, normalize=normalize)
    
    results[predictor.name]['history'] = int(path.split('_')[2])
    results[predictor.name]['layers'] = int(path.split('_')[5])
    results[predictor.name]['dilation'] = int(path.split('_')[15])
    results[predictor.name]['activation'] = path.split('_')[-1]
    results[predictor.name]['parameters'] = predictor.model.count_params()
    print('\n', results[predictor.name])

#%% Output to csv
df = pd.DataFrame.from_dict(results, orient='index')
df.to_csv('cnn_results.csv')

#%% Best MLP model
paths = [p for p in os.listdir('logs') if 'MLP' in p]

for path in paths:
    weights = [f for f in os.listdir(os.path.join('logs', path)) if f.endswith('.h5')][0]
    print(os.path.join('logs', path, weights))
    
    # Need correct history for input shape
    history = int(path.split('_')[2])
    
    predictor = Perceptron(
            history=history,
            num_hidden_layers=3,
            neurons_per_layer=32,
            activation='relu'
            )
    predictor.load(os.path.join('logs', path, weights))
    predictor.name = path
    y_pred, y_true = predictor.predict(trace['Branch'])
    results[predictor.name] = utils.evaluate(y_true, y_pred, name='Multilayer Perceptron',
           plot=False, normalize=normalize)
    results[predictor.name]['history'] = int(path.split('_')[2])
    results[predictor.name]['layers'] = int(path.split('_')[5])
    results[predictor.name]['activation'] = path.split('_')[-1]
    results[predictor.name]['parameters'] = predictor.model.count_params()
    print('\n', results[predictor.name])
    
#%% Output to csv
df = pd.DataFrame.from_dict(results, orient='index')
df.to_csv('mlp_results.csv')
