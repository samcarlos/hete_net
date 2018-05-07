import numpy as np
import pandas as pd

x = np.random.normal(0, 1, 10000)
t = np.random.binomial(1,.5,10000)
y = x + x*t + np.random.normal(0, 1, 10000)
test_data = pd.DataFrame([x,t,y]).T
test_data.columns = ['x','t','y']

test_data.to_csv('/Users/samweiss/src/hete_net/hete_data.csv')
