library(hete)

data = read.csv("/Users/samweiss/src/hete_net/hete_data.csv")
data = data[,-1]

m <- hete_single(y ~ . | t, data = data, est = random_forest)
p <- predict(m, data)

write.csv(p, file = '/Users/samweiss/src/hete_net/hete_preditions_tree.csv')
