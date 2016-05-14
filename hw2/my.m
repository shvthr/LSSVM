X=(-10:0.1:10)';
Y = cos(X) + cos(2*X) + 0.1.*rand(length(X),1);

out=[15 17 19];
Y(out)=0.7+0.3*rand(size(out));
out=[41 44 46];
Y(out)=1.5+0.2*rand(size(out));

gam=100;sig2=0.1;
[alpha,b]=trainlssvm({X,Y,'f',gam,sig2,'RBF_kernel'});
plotlssvm({X,Y,'f',gam,sig2,'RBF_kernel'},{alpha,b});
%%
model=initlssvm(X,Y,'f',[],[],'RBF_kernel');
costFun='rcrossvalidatelssvm';
wFun='whuber';
model=tunelssvm(model,'simplex',costFun,{10,'mae'},wFun);
model=robustlssvm(model);
plotlssvm(model);