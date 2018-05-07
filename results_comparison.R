data = read.csv("/Users/samweiss/src/hete_net/hete_data.csv")
data = data[,-1]

hete_tree = read.csv('/Users/samweiss/src/hete_net/hete_preditions_tree.csv')[,2]
hete_net = read.csv('/Users/samweiss/src/hete_net/hete_preditions_net.csv')[,2]

#x is the hete effect 
x = data[,'x']


pairs(cbind(x, hete_tree, hete_net))


lm_tree = lm(x ~ hete_tree)
lm_net = lm(x ~ hete_net)


print(summary(lm_tree))
print(summary(lm_net))
