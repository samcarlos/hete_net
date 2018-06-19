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


import keras.backend as K
from keras.layers import multiply

def customLoss(yTrue,yPred):
    output = yTrue[:,:2]* yPred[:,:2]
    output = K.sum(output,axis = -1)*yTrue[:,2]

    return -K.mean(output)


def lelu(x):
    return (K.log(K.relu(x)+1))

def hete_optim_action(input_shape, num_nodes, dropout, num_layers = 1, activation=lelu):
    input = Input(shape=(input_shape,))
    aux_input = Input(shape=(1,))

    x = Dense(num_nodes, activation=activation)(input)
    x = Dropout(dropout)(x)

    if(num_layers > 1 ):
      for q in range(num_layers - 1):
        x = Dense(num_nodes, activation=activation)(x)
        x = Dropout(dropout)(x)

    x = Dense(2, activation = 'softmax')(x)
    x = keras.layers.concatenate([x, aux_input])

    model = Model([input, aux_input], x)
    model.compile(optimizer='rmsprop', loss=customLoss)

    return(model)






def gridsearch_hete_optim_action(X_train, tmt_control,  y_train):

    param_grid = dict(num_nodes = [16, 64, 256], dropout = [.9, .5,.75], activation = [lelu, 'relu'], num_layers = [1,2])

    grid = ParameterGrid(param_grid)


    grid = ParameterGrid(param_grid)

    from sklearn.model_selection import KFold
    kf = KFold(n_splits=3)
    kf.get_n_splits(y_train)

    results = []

    dummy_y_train = np.concatenate([1-tmt_control, tmt_control, y_train], axis = 1)


    for params in grid:
        temp_results = []
        for train_index, test_index in kf.split(y_train):
            mod = hete_optim_action(X_train.shape[1], params['num_nodes'], params['dropout'],
                params['num_layers'], activation =  params['activation'])

            mod.fit([X_train[train_index], X_train[train_index][:,0]],
                    dummy_y_train[train_index], epochs = 100)

            preds = mod.predict([X_train[test_index], X_train[test_index][:,0]])

            optim_value_location = np.argmax(preds[:,:2], axis = 1)

            proposed_model = np.where(optim_value_location.reshape(len(optim_value_location),1) == tmt_control[test_index])[0]

            gains = y_train[test_index][proposed_model].mean()
            print(gains)
            temp_results.append(gains)

        results.append(np.mean(temp_results).mean())

    optim_grid = [x for x in grid][np.argmax(np.array(results))]
    print(optim_grid)
    print(results[np.argmax(np.array(results))])

    mod  = hete_optim_action(X_train.shape[1], optim_grid['num_nodes'], optim_grid['dropout'],
                optim_grid['num_layers'], activation =  optim_grid['activation'])

    mod.fit([X_train, X_train[:,0]],
                    dummy_y_train, epochs = 100)

    return(mod)
