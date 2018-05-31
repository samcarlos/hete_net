library(hete)
library(tidyverse)
data(gotv)

df <- gotv %>%
  filter(treatment %in% c("Control", "Neighbors")) %>%
  mutate(treatment = ifelse(treatment == "Control", 0, 1)) %>%
  as.data.frame()

set.seed(25)

#use explanatory variable to create known interaction function
voted =  .1*df[,'treatment']*df[,'age'] +
  .1*df[,'treatment']*df['hh_size'] + df[,'hh_size'] + (df[,'g2002']=="no")*1

voted = as.numeric(voted[,1])
voted = scale(voted)
voted = 1/(1+exp(-voted))
voted_1 = rbinom(length(voted),1,
                 voted)
voted_2 = rep('Yes', length(voted_1))
voted_2[which(voted_1==0)] = 'No'

df[,'voted'] = as.factor(voted_2)

sample_train = sample(nrow(df), nrow(df)*2/3)
df_train = df[sample_train, ]
df_test = df[-sample_train, ]

m <- hete_single(voted ~ . | treatment, data = df_train, est = random_forest)
p <- predict(m, df_test)
write.csv(df_train, '/Users/samweiss/src/hete_net/gotv_train.csv', row.names = FALSE)
write.csv(df_test, '/Users/samweiss/src/hete_net/gotv_test.csv', row.names = FALSE)
write.csv(p, '/Users/samweiss/src/hete_net/hete_preditions_tree_gotv.csv', row.names = FALSE)
