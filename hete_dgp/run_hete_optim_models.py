from load_data import load_data
from hete_optim import gridsearch_hete_optim_action, hete_optim_action
import pandas as pd
import numpy as np



def run_scenario_hete_optim_model(scenario_number):
  t_train, t_test, t_x_train, t_x_test, X_train, X_test, y_train,y_test = load_data(scenario_number)


  model = gridsearch_hete_optim_action(X_train, t_train, y_train)


  predictions = model.predict([X_test,X_test[:,0]])
  pd.DataFrame(predictions).to_csv('/Users/sweiss/src/hete_net/hete_dgp/predicted_data_hete_net/hete_preds_optim_'+str(scenario_number)+'.csv')

[run_scenario_hete_optim_model(x) for x in range(8)]
