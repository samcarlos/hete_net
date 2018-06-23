library(hete)
scenario = 1
directory = '/Users/sweiss/src/hete_net/hete_dgp/created_data/'
run_model = function(directory, scenario){
  t = read.csv(paste0(directory,'scenario_',scenario,'_t.csv'))
  t_x = read.csv(paste0(directory,'scenario_',scenario,'_t_x.csv'))
  x = read.csv(paste0(directory,'scenario_',scenario,'_x.csv'))
  y = read.csv(paste0(directory,'scenario_',scenario,'_y.csv'))
  
  t_train = t[(1:15000),]
  t_x_train = t_x[(1:15000),]
  x_train = x[(1:15000),]
  y_train = y[(1:15000),]
  
  t_test = t[-c(1:15000),]
  t_x_test = t_x[-c(1:15000),]
  x_test = x[-c(1:15000),]
  y_test = y[-c(1:15000),]
  
  new_data = data.frame(t= as.factor(t_train), x_train)
  y_train = as.factor(y_train)
  model = hete_single(y_train~.| t, data = new_data,
                      est = random_forest)
  
  data = cbind(t = t_test, x_test)
  data[,1] = as.factor(data[,1])
  
  data = data.frame(t= factor(t_test, levels = levels(new_data[, 1])), x_test)
  data_1 = new_data
  data_1[1:3000,] = data
  predictions = predict(model, newdata=data_1)
  predictions = predictions[1:3000]
  write.csv(predictions, file = paste0('/Users/sweiss/src/hete_net/hete_dgp/predicted_data_hete_net/hete_R_preds_','scenario_',scenario,'_t.csv'))

  
  }

lapply(0:8,function(scenario) run_model('/Users/sweiss/src/hete_net/hete_dgp/created_data/',scenario))
