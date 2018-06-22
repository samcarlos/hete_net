import numpy as np
import pandas as pd
import os

##https://arxiv.org/pdf/1707.00102.pdf
def f_1(x):
    return(np.zeros(x.shape[0]))

def f_2(x):
    y = 5*(x[:,0]>1) - 5
    return(y)

def f_3(x):
    y = 2*x[:,2] - 4
    return(y)

def f_4(x):
    # dumb way to do this.. im sure theres a better way
    y = x[:,3]*x[:,5]*x[:,7] + \
        2*x[:,3]*x[:,5]*(1-x[:,7]) + \
        3*x[:,3]*(1-x[:,5])*x[:,7] + \
        4*x[:,3]*(1-x[:,5])*(1-x[:,7])+\
        5*(1-x[:,3])*x[:,5]*x[:,7] + \
        6*(1-x[:,3])*x[:,5]*(1-x[:,7]) + \
        7*(1-x[:,3])*(1-x[:,5])*x[:,7] + \
        8*(1-x[:,3])*(1-x[:,5])*(1-x[:,7])
    return(y)

def f_5(x):
    y = x[:,0] + x[:,2] + x[:,4] + x[:,6] + x[:,7] +x[:,8] - 2
    return(y)

def f_6(x):
    y = 4*(x[:,0]>1)*(x[:,2]>1) + 4*(x[:,4]>1)*(x[:,6]>1) + 2*x[:,7]*x[:,8]
    return(y)

def f_7(x):
    y = .5* (x[:,0]**2 + x[:,1] + x[:,2]**2+ x[:,3] +x[:,4]**2 + x[:,5] + x[:,6]**2 + x[:,7] + x[:,8]**2 - 11)
    return(y)

def f_8(x):
    y = (.5**.5) * ( f_4(x) + f_5(x) )
    return(y)

def create_dataset(n_obs, p_over_2, main_effects, interactions):
    x = np.concatenate([np.concatenate([np.random.normal(0,1,n_obs).reshape(n_obs,1), np.random.binomial(1, .5 ,n_obs).reshape(n_obs,1)],axis = 1) for x in range(p_over_2)], axis = 1)
    t = np.random.binomial(1, .5 ,n_obs)
    u_x = main_effects(x)
    t_x = interactions(x)
    y_mean = u_x + (t - .5) * t_x
    y = np.random.normal( y_mean ,1,n_obs)
    y = 1/(1+np.exp(-y))
    y = np.random.binomial(1, y, n_obs)
    return([y, t, x, t_x])


scenarios = [[f_8,f_1], [f_5, f_6], [f_4, f_3], [f_7,f_4], [f_3,f_5], [f_1, f_6], [f_2,f_7], [f_6,f_8]]

data_sets = [create_dataset(18000, 200, sen[0], sen[1]) for sen in scenarios]

def save_scenario(data, j):
  list_of_names = ['y', 't', 'x', 't_x']
  [pd.DataFrame(x).to_csv(os.getcwd()+'/created_data/scenario_'+str(j)+'_'+ list_of_names[i]+'.csv', index = False) for i,x  in enumerate(data) ]

[save_scenario(dat,j) for j, dat in enumerate(data_sets)]
