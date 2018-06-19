import numpy as np
import pandas as pd

def load_data(scenario, directory = '/Users/sweiss/src/hete_net/hete_dgp/created_data/'):
  t = np.array(pd.read_csv(directory+'scenario_0_t.csv'))
  t_x = np.array(pd.read_csv(directory+'scenario_0_t_x.csv'))
  x = np.array(pd.read_csv(directory+'scenario_0_x.csv'))
  y = np.array(pd.read_csv(directory+'scenario_0_y.csv'))

  t_train = t[:15000]
  t_x_train = t_x[:15000]
  x_train = x[:15000]
  y_train = y[:15000]

  t_test = t[15000:]
  t_x_test = t_x[15000:]
  x_test = x[15000:]
  y_test = y[15000:]

  return([t_train, t_test, t_x_train, t_x_test, x_train, x_test, y_train,y_test])
