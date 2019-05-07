import utils
from predictors import (
        Static,
        NbitCounter,
        Bimodal,
        Perceptron,
        CNN
        )

trace = utils.read_data('trace.csv')

results = {}
normalize = True

#%%
predictor = Bimodal(m=64, n=1)

y_pred = predictor.predict(trace['Branch'], trace['PC'])
results[predictor.name] = utils.evaluate(trace['Branch'], y_pred, name=predictor.name, normalize=normalize)
print('\n', results[predictor.name])

#%%
predictor = Bimodal(m=64, n=2)

y_pred = predictor.predict(trace['Branch'], trace['PC'])
results[predictor.name] = utils.evaluate(trace['Branch'], y_pred, name=predictor.name, normalize=normalize)
print('\n', results[predictor.name])

#%%
predictor = Bimodal(m=128, n=1)

y_pred = predictor.predict(trace['Branch'], trace['PC'])
results[predictor.name] = utils.evaluate(trace['Branch'], y_pred, name=predictor.name, normalize=normalize)
print('\n', results[predictor.name])

#%%
predictor = Bimodal(m=128, n=2)

y_pred = predictor.predict(trace['Branch'], trace['PC'])
results[predictor.name] = utils.evaluate(trace['Branch'], y_pred, name=predictor.name, normalize=normalize)
print('\n', results[predictor.name])

#%% Static Taken Predictor
predictor = Static(always_taken=True)

y_pred = predictor.predict(trace['Branch'])
results[predictor.name] = utils.evaluate(trace['Branch'], y_pred, name=predictor.name, normalize=normalize)
print('\n', results[predictor.name])

#%% Static Not Taken Predictor
predictor = Static(always_taken=False)

y_pred = predictor.predict(trace['Branch'])
results[predictor.name] = utils.evaluate(trace['Branch'], y_pred, name=predictor.name, normalize=normalize)
print('\n', results[predictor.name])

#%% 1-bit Counter Predictor
predictor = NbitCounter(n=1)

y_pred = predictor.predict(trace['Branch'])
results[predictor.name] = utils.evaluate(trace['Branch'], y_pred, name=predictor.name, normalize=normalize)
print('\n', results[predictor.name])

#%% 2-bit Counter Predictor
predictor = NbitCounter(n=2)

y_pred = predictor.predict(trace['Branch'])
results[predictor.name] = utils.evaluate(trace['Branch'], y_pred, name=predictor.name, normalize=normalize)
print('\n', results[predictor.name])

#%% MLP Predictor
predictor = Perceptron(
        history=9,
        num_hidden_layers=3,
        neurons_per_layer=32,
        activation='relu',
        )
predictor.fit(trace['Branch'], epochs=50, batch_size=64)

#%%

results[predictor.name] = utils.evaluate(trace['Branch'], y_pred, name=predictor.name, normalize=normalize)
print('\n', results[predictor.name])
