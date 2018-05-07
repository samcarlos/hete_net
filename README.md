# hete_net

sets up a simple example for a hete treatment effect: the data generating process is:
  x~normal(0,1)
  t~binomial(.5)
  y~normal(x+x*t, 1)
The treatment effect is x.


I set up a net to predict the response as a function of the counterfactual. That is y ~ f(x, T=t) - f(x, T=0). This appears to work...

The results show that r^2 between hete effect of net vs x is .99 compared with hete tree model of .9



to run:

python create_data.py
python3 hete_net.py
R CMD BATCH hete_tree.R
R CMD BATCH results_comparison.R
