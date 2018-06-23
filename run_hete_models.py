from load_data import load_data
from hete_net_interactions import gridsearch_hete_model

from sklearn.grid_search import ParameterGrid
from keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
import numpy as np

def run_scenario_hete_model(scenario_number):
  t_train, t_test, t_x_train, t_x_test, X_train, X_test, y_train,y_test = load_data(scenario_number)

  X_train = np.concatenate([X_train, t_train], axis = 1)
  X_test = np.concatenate([X_test, t_test], axis = 1)

  X_train_control = X_train.copy()
  X_train_control[:,0] = np.unique(X_train[:,0])[0]

  model, base = gridsearch_hete_model(X_train, X_train_control, y_train)


  X_test_tmt = X_test.copy()
  X_test_tmt[:,0] = np.unique(X_test[:,0])[1]

  X_test_control = X_test.copy()
  X_test_control[:,0] = np.unique(X_test[:,0])[0]

  predictions = model.predict([X_test_tmt,X_test_control])
  pd.DataFrame(predictions).to_csv('/Users/sweiss/src/hete_net/hete_dgp/predicted_data_hete_net/hete_preds_'+scenario_number+'.csv')

[run_scenario_hete_model(x) for x in range(8)]
