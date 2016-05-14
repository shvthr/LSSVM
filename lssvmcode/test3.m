load ('iris.mat');
t=1;
degree=1;
[alpha,b] = trainlssvm({X,Y,'c',gam,[],'lin_kernel'});
plotlssvm({X,Y,'c',gam,[],'lin_kernel'},{alpha,b});
[alpha,b] = trainlssvm({X,Y,'c',gam,[t;degree],'poly_kernel'});