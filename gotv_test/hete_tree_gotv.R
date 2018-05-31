library(hete)
library(tidyverse)

data = read.csv('/Users/samweiss/src/hete_net/gotv_train.csv')
test_data = read.csv('/Users/samweiss/src/hete_net/gotv_test.csv')

m <- hete_single(voted ~ . | treatment, data = data, est = random_forest)
p <- predict(m, test_data)

write.csv(p, file = '/Users/samweiss/src/hete_net/hete_preditions_tree_gotv.csv')
