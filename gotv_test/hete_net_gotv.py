import keras
from keras import backend as K

from keras.layers import Input, Embedding, LSTM, Dense, Dropout
from keras.models import Model
from keras.layers import Input, Lambda
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
#import data
data = pd.read_csv('/Users/samweiss/src/hete_net/gotv_train.csv')
test_data = pd.read_csv('/Users/samweiss/src/hete_net/gotv_test.csv')
#format
data = pd.get_dummies(data)
test_data = pd.get_dummies(test_data)

y = data['voted_Yes']
data.drop(['voted_No','voted_Yes','g2004_yes'],axis = 1, inplace = True)
test_data.drop(['voted_No','voted_Yes','g2004_yes'],axis = 1, inplace = True)

scaler = StandardScaler()
scaler.fit(data)
transformed_data = scaler.transform(data)
transformed_test_data = scaler.transform(test_data)

#y = (y - y.mean()) / y.std()






#base network
def lelu(x):
    return (K.log(K.relu(x)+1))

def base_model(num_nodes, dropout, activation=lelu):
    input = Input(shape=(17,))
    x = Dense(num_nodes, activation=activation)(input)
    x = Dropout(dropout)(x)
    x = Dense(1, activation = 'sigmoid')(x)
    model = Model(input, x)
    return(model)

#two inputs. this will estimate y ~ f(x,T=t) - t(x, T=0)

def hete_model(num_nodes=256, dropout = .5, activation = lelu):
    input_observed = Input(shape=(17,), dtype='float32', name='observed')
    input_base = Input(shape=(17,), dtype='float32', name='base')

    base_network = base_model(num_nodes, dropout, activation)

    output_observed = base_network(input_observed)
    output_base = base_network(input_base)


    diff = keras.layers.Subtract()([output_observed, output_base])

    model = Model(inputs=[input_observed, input_base], outputs=diff)
    model.compile(optimizer='rmsprop', loss='mse')
    return(model , base_network)



#setup counterfactual data
transformed_data_control = transformed_data.copy()
transformed_data_control[:,0] = np.unique(transformed_data[:,0])[0]

model, base = hete_model()

model.fit([transformed_data, transformed_data_control], y, epochs = 25)

transformed_test_data_tmt = transformed_test_data.copy()
transformed_test_data_tmt[:,0] = np.unique(transformed_data[:,0])[1]

transformed_test_data_control = transformed_test_data.copy()
transformed_test_data_control[:,0] = np.unique(transformed_data[:,0])[0]

predictions = model.predict([transformed_test_data_tmt,transformed_test_data_control])
pd.DataFrame(predictions).to_csv('/Users/samweiss/src/hete_net/hete_preditions_net_gotv.csv')
