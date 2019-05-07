from itertools import product
from predictors import CNN, Perceptron
import utils


trace = utils.read_data('trace.csv')

epochs, batch_size = 25, 256
'''
hist = [5, 9, 21]
num_layers = [3, 9]
activations = ['relu', 'linear']
dilations = [1, 2]

params = list(product(activations, num_layers, dilations, hist))

for activation, num_hidden_layers, dilation, history in params:

    print('\n\nModel: History: {}, Layers: {}, Activation: {}, Dilation: {}'.format(history, num_hidden_layers, activation, dilation))

    predictor = CNN(
        history=history,
        num_hidden_layers=num_hidden_layers,
        num_filters=32,
        kernel_size=3,
        activation=activation,
        dilation=dilation,
        skip=True
        )

    predictor.fit(trace['Branch'], epochs=epochs, batch_size=batch_size)
    del predictor
'''


predictor = CNN(
    history=9,
    num_hidden_layers=3,
    num_filters=32,
    kernel_size=3,
    activation='relu',
    dilation=1,
    skip=True
    )

predictor.fit(trace['Branch'], epochs=epochs, batch_size=batch_size)