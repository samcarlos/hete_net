scenario = 1
directory = '/Users/sweiss/src/hete_net/hete_dgp/created_data/'


for(scenario in 0:10){
t = read.csv(paste0(directory,'scenario_',scenario,'_t.csv'))
t_x = read.csv(paste0(directory,'scenario_',scenario,'_t_x.csv'))
#x = read.csv(paste0(directory,'scenario_',scenario,'_x.csv'))
y = read.csv(paste0(directory,'scenario_',scenario,'_y.csv'))

t_train = t[(1:15000),]
t_x_train = t_x[(1:15000),]
#x_train = x[(1:15000),]
y_train = y[(1:15000),]

t_test = t[-c(1:15000),]
t_x_test = t_x[-c(1:15000),]
#x_test = x[-c(1:15000),]
y_test = y[-c(1:15000),]
hete_net = read.csv(paste0('/Users/sweiss/src/hete_net/hete_dgp/predicted_data_hete_net/hete_preds_',scenario,'.csv'))
hete_optim = read.csv(paste0('/Users/sweiss/src/hete_net/hete_dgp/predicted_data_hete_net/hete_preds_optim_',scenario,'.csv'))
hete_R = read.csv(paste0('/Users/sweiss/src/hete_net/hete_dgp/predicted_data_hete_net/hete_R_preds_','scenario_',scenario,'_t.csv'))

combined_data = cbind(t_x_test,hete_net = hete_net[,2], hete_optim = hete_optim[,2], hete_r = hete_R[,2])
pairs(combined_data)
}
