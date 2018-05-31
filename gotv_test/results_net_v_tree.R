library(hete)

df_test = read.csv( '/Users/samweiss/src/hete_net/gotv_test.csv')

hete_net = read.csv('/Users/samweiss/src/hete_net/hete_preditions_net_gotv.csv')[,2]
hete_tree = read.csv('/Users/samweiss/src/hete_net/hete_preditions_tree_gotv.csv')[,1]


oracle =  .1*df_test[,'age'] +
  .1*df_test['hh_size']
oracle = oracle[,1]

reg_oracle_net = lm(oracle ~ hete_net)
reg_oracle_tree = lm(oracle ~ hete_tree)

summary(reg_oracle_net)
#r^2 of 0.3654
summary(reg_oracle_tree)
#r^2 fo 0.1883

uplift_tree= uplift(df_test[,'voted'], df_test[,'treatment'], hete_tree, bins = 10)
#q of 0.02028622
uplift_net= uplift(df_test[,'voted'], df_test[,'treatment'], hete_net, bins = 10)
#q of 0.007719851
