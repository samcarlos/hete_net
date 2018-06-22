import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from keras.layers import Input, Embedding, LSTM, Dense, Dropout
from keras.models import Model
from keras.layers import Input, Lambda
import numpy as np
import pandas as pd
#base network
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from sklearn.grid_search import ParameterGrid
from keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score


def lelu(x):
    return (K.log(K.relu(x)+1))

def base_model(input_shape, num_nodes, dropout, num_layers = 1, activation=lelu):
    input = Input(shape=(input_shape,))
    x = Dense(num_nodes, activation=activation)(input)
    x = Dropout(dropout)(x)

    if(num_layers > 1 ):
      for q in range(num_layers - 1):
        x = Dense(num_nodes, activation=activation)(x)
        x = Dropout(dropout)(x)

    x = Dense(1)(x)
    model = Model(input, x)
    return(model)

#two inputs. this will estimate y ~ f(x,T=t) - t(x, T=0)

def hete_model(input_shape , num_nodes=256, dropout = .5, num_layers = 1, activation = lelu):

    input_observed = Input(shape=(input_shape,), dtype='float32', name='observed')
    input_base = Input(shape=(input_shape,), dtype='float32', name='base')

    base_network = base_model(input_shape, num_nodes, dropout, num_layers, activation)

    output_observed = base_network(input_observed)
    output_base = base_network(input_base)

    diff = keras.layers.Subtract()([output_observed, output_base])
    model = Model(inputs=[input_observed, input_base], outputs=diff)
    model.compile(optimizer='rmsprop', loss='mse')
    return(model , base_network)


def gridsearch_hete_model(X_train, X_train_control, y_train):
    param_grid = dict(num_nodes = [32,128, 256], dropout = [.5], activation = [lelu, 'relu'], num_layers = [1,2])
    grid = ParameterGrid(param_grid)

    from sklearn.model_selection import KFold
    kf = KFold(n_splits=3)
    kf.get_n_splits(y_train)
    results = []

    for params in grid:
        temp_results = []
        for train_index, test_index in kf.split(y_train):
            mod , base = hete_model(X_train.shape[1], params['num_nodes'], params['dropout'],
                params['num_layers'], activation =  params['activation'])
            mod.fit([X_train[train_index], X_train_control[train_index]],
                    y_train[train_index], epochs = 50)

            preds = mod.predict([X_train[test_index], X_train_control[test_index]])

            temp_results.append(mean_squared_error(y_train[test_index], preds))
            print(np.mean(temp_results).mean())

        results.append(np.mean(temp_results).mean())
    optim_grid = [x for x in grid][np.argmax(np.array(results))]
    print(optim_grid)
    print(results[np.argmax(np.array(results))])

    mod , base = hete_model(X_train.shape[1], optim_grid['num_nodes'], optim_grid['dropout'],
                optim_grid['num_layers'], activation =  optim_grid['activation'])
    mod.fit([X_train, X_train_control],
                    y_train, epochs = 50)
    return([mod, base])
