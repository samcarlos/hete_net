import keras
from keras.layers import Input, Embedding, LSTM, Dense, Dropout
from keras.models import Model
from keras.layers import Input, Lambda
import numpy as np
import pandas as pd

#import data
data = pd.read_csv('/Users/samweiss/src/hete_net/hete_data.csv', index_col=0)
y = np.array(data['y'])
x_data =data.drop('y', axis = 1)



#base network
input = Input(shape=(2,))
x = Dense(128, activation='relu')(input)
x = Dense(1)(x)
base_network = Model(input, x)

#two inputs. this will estimate y ~ f(x,T=t) - f(x, T=0)

input_observed = Input(shape=(2,), dtype='float32', name='observed')
input_base = Input(shape=(2,), dtype='float32', name='base')
output_observed = base_network(input_observed)
output_base = base_network(input_base)

diff = keras.layers.Subtract()([output_observed, output_base])
model = Model(inputs=[input_observed, input_base], outputs=diff)
model.compile(optimizer='rmsprop', loss='mse')

#setup counterfactual data
x_data_0 = x_data.copy()
x_data_1 = x_data.copy()

x_data_0['t'] = 0
x_data_1['t'] = 1



model.fit([np.array(x_data), np.array(x_data_0)], y, epochs = 15)
predictions = model.predict([np.array(x_data_1), np.array(x_data_0)])
pd.DataFrame(predictions).to_csv('/Users/samweiss/src/hete_net/hete_preditions_net.csv')
