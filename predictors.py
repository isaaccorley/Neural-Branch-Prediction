import os
import numpy as np
import keras.backend as K
from copy import deepcopy
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Dense, Conv1D, Add, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint


states = {
            1: {
                'T': {
                    'prediction': 1, # predicts taken
                    'transition': (
                            'N',    # incorrect prediction: transition to NT state
                            'T'      # correction prediction: transition to T state
                            )},
                'N': {
                    'prediction': 0, # predicts not taken
                    'transition': (
                            'T',     # incorrect prediction: transition to T state
                            'N'     # correction prediction: transition to NT state
                            )}
                    },
                    
            2: {
                'ST': {
                    'prediction': 1, # predicts taken
                    'transition': (
                            'WT',    # incorrect prediction: transition to WT state
                            'ST'     # correction prediction: transition to ST state
                            )},
                'WT': {
                    'prediction': 1, # predicts taken
                    'transition': (
                            'WN',   # incorrect prediction: transition to WN state
                            'ST'     # correction prediction: transition to ST state
                            )},
                'SN': {
                    'prediction': 0, # predicts not taken
                    'transition': (
                            'WN',   # incorrect prediction: transition to WN state
                            'SN'    # correction prediction: transition to SN state
                            )},
                'WN': {
                    'prediction': 0, # predicts not taken
                    'transition': (
                            'WT',    # incorrect prediction: transition to WT state
                            'SN'    # correction prediction: transition to SN state
                            )}
                }
        }
                    
                    
class Predictor(object):
    """ Base Predictor Class """

    def predict(self, y_true):
        """ Perform prediction of branch taken/not taken """
        raise NotImplementedError


class NeuralNetwork(Predictor):
    """ Base Neural Network Predictor Class """

    def __build_model(self):
        """ Create neural network static graph """
        raise NotImplementedError
        
        
    def _preprocess(self, y_true):
        """ Preprocess data into rolling window slices """
        X = np.zeros((len(y_true)-self.history, self.history), dtype='uint8')
        y = np.zeros((len(y_true)-self.history), dtype='uint8')
        
        for i, j in enumerate(range(self.history, len(y_true))):
            X[i, :] = y_true[j-self.history:j]
            y[i] = y_true[j]

        if len(self.input_shape) > 1:
            X = np.expand_dims(X, axis=-1)
            
        return train_test_split(X, y, test_size=0.25, shuffle=False, random_state=0)
        
    
    def fit(self, y_true, epochs=50, batch_size=64, tb=False):
        """ Train model on train set, evaluate on test set """

        # Preprocess data
        X_train, X_test, y_train, y_test = self._preprocess(deepcopy(y_true))
        print('X_train shape: {}, X_test shape {}'.format(X_train.shape, X_test.shape))

        # Setup Log directory
        if not os.path.exists(os.path.join('logs', self.name)):
            os.mkdir(os.path.join('logs', self.name))
            
        # Callbacks
        callbacks = []
        callbacks.append(CSVLogger(filename=os.path.join('logs', self.name, 'train_log.csv')))
        
        callbacks.append(ModelCheckpoint(filepath=os.path.join('logs', self.name,
                                    'weights_{epoch:02d}_{val_loss:.4f}.h5'),
                                    save_best_only=True, save_weights_only=False))

        if tb:
            callbacks.append(TensorBoard(log_dir=os.path.join('logs', self.name, 'tb'),
                            histogram_freq=1, write_graph=False, write_images=False))
        
        # Fit model to train set
        _ =  self.model.fit(X_train, y_train, validation_data=(X_test, y_test),
                            epochs=epochs, batch_size=batch_size, callbacks=callbacks)


class Static(Predictor):
    """
    Static Branch Predictor
    Predicts always taken or not taken

    """

    def __init__(self, always_taken):
        self.always_taken = always_taken
        self.name = 'Static Taken' if self.always_taken else 'Static Not Taken'


    def predict(self, y_true):

        taken = 1 if self.always_taken else 0

        y_pred = [taken] * len(y_true)

        return y_pred


class NbitCounter(Predictor):
    """
    N-bit Counter Predictor
    Returns prediction given the current state
    """

    def __init__(self, n):
        
        self.n = n
        self.name = str(self.n) + '-bit Counter'
        
        self.states = states[n].copy()
        self.current_state = 'T' if self.n == 1 else 'ST'
        

    def predict(self, y_true):
        
        # Pre allocate 
        y_pred = [None] * len(y_true)
        
        for i, branch in tqdm(enumerate(y_true)):
            
            # Predict taken/not taken
            y_pred[i] = self.states[self.current_state]['prediction']
            
            # Check if correct/incorrect prediction
            hit = (y_pred[i] == branch)
            
            # Update the state
            self.current_state = self.states[self.current_state]['transition'][hit]
            
        return y_pred


class Bimodal(Predictor):
    """ Bimodal Branch Predictor """

    def __init__(self, m, n):
        
        self.m, self.n = m, n
        self.name = 'Bimodal {}-bit Counter'.format(str(self.n))
        
        # Counters
        self.states = states[n].copy()
        self.init_state = 'T' if self.n == 1 else 'ST'

        # Branch History Table of Counters
        self.bh_table = {i: self.init_state for i in range(self.m)}


    def predict(self, y_true, pc):
        
        # Pre allocate 
        y_pred = [None] * len(y_true)
        
        for i, (pc, branch) in tqdm(enumerate(zip(pc, y_true))):
            
            # Calculate the index of the N-bit counter to use in the table = PC % m
            index = pc % self.m
            
            # Predict taken/not taken
            y_pred[i] = self.states[self.bh_table[index]]['prediction']
            
            # Check if correct/incorrect prediction
            hit = (y_pred[i] == branch)
            
            # Update the state
            self.bh_table[index] = self.states[self.bh_table[index]]['transition'][hit]
        
        return y_pred


class GShare(Predictor):
    """ GShare Branch Predictor """
    
    def __init__(self, history, m, n):
        
        self.history, self.m, self.n = history, m, n
        self.name = 'Bimodal {}-bit Counter'.format(str(self.n))
        
        # Counters
        self.states = states[n].copy()
        self.current_state = 'T' if self.n == 1 else 'ST'

        # Branch History Table of Counters
        self.bh_table = {i: self.init_state for i in range(self.m)}


    def predict(self, y_true, pc):
        
        # Pre allocate 
        y_pred = [None] * len(y_true)
        
        for i, (pc, branch) in tqdm(enumerate(zip(pc, y_true))):
            
            # Calculate the index of the N-bit counter to use in the table = PC % m
            index = pc % self.m
            
            # Predict taken/not taken
            y_pred[i] = self.states[self.bh_table[index]]['prediction']
            
            # Check if correct/incorrect prediction
            hit = (y_pred[i] == branch)
            
            # Update the state
            self.bh_table[index] = self.states[self.bh_table[index]]['transition'][hit]
        
        return y_pred
    
    
    
class Perceptron(NeuralNetwork):
    """
    Multilayer Perceptron (MLP) Branch Predictor
    """

    def __init__(self,
                 history,
                 num_hidden_layers=3,
                 neurons_per_layer=32,
                 activation='relu'
                 ):
        
        self.history = history
        self.num_hidden_layers = num_hidden_layers
        self.neurons_per_layer = neurons_per_layer
        self.activation = activation
        
        self.name = '_'.join(['MLP',
                              'history',
                              str(self.history),
                              'hidden_layers',
                              str(self.num_hidden_layers),
                              'neurons',
                              str(self.neurons_per_layer),
                              'activation',
                              self.activation
                              ])

        self.input_shape = (self.history,)
        self.model = self.__build_model()


    def __build_model(self):
        K.clear_session()

        inputs = Input(shape=self.input_shape)
        x = inputs

        for _ in range(self.num_hidden_layers):
            x = Dense(units=self.neurons_per_layer, activation=self.activation)(x)

        outputs = Dense(1, activation='sigmoid')(x)

        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(lr=1E-3), loss='binary_crossentropy')
        return model
        
    
    def predict(self, y_true):
        pass


class CNN(NeuralNetwork):
    """
    1-D Convolutional Neural Network Branch Predictor
    """
    
    def __init__(self,
                 history,
                 num_hidden_layers=3,
                 num_filters=32,
                 kernel_size=3,
                 skip=False,
                 dilation=1,
                 padding='same',
                 activation='relu',
                 ):
        self.history = history
        self.num_hidden_layers = num_hidden_layers
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.skip = skip
        self.dilation = dilation
        self.strides = 1
        self.padding = 'same'
        
        self.name = '_'.join(['CNN',
                              'history',
                              str(self.history),
                              'hidden_layers',
                              str(self.num_hidden_layers),
                              'num_filters',
                              str(self.num_filters),
                              'kernel_size',
                              str(self.kernel_size),
                              'skip',
                              str(self.skip),
                              'dilation',
                              str(self.dilation),
                              'activation',
                              self.activation
                              ])

        self.input_shape = (self.history, 1)
        self.model = self.__build_model()


    def __build_model(self):
        K.clear_session()

        inputs = Input(shape=self.input_shape)
        x = inputs

        for _ in range(self.num_hidden_layers):
            shortcut = x
            x = Conv1D(filters=self.num_filters,
                       kernel_size=self.kernel_size,
                       padding=self.padding,
                       strides=self.strides,
                       dilation_rate=self.dilation,
                       activation=self.activation
                       )(x)
            
            if self.skip:
                x = Add()([x, shortcut])

        x = Flatten()(x)
        outputs = Dense(1, activation='sigmoid')(x)

        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(), loss='binary_crossentropy')
        return model
        
    
    def predict(self, y_true):
        pass