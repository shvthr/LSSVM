%addpath('E:\Academic\SVN\thesis\code\lssvmlab')

load santafe;
order=10;
X=windowize(Z,1:(order+1));
Y=X(:,end);
X=X(:,1:order);
gam=10; sig2=10;
[gam,sig2]=tunelssvm({X,Y,'f',[],[],'RBF_kernel'},'simplex','crossvalidatelssvm',{10,'mse'});
[alpha,b]=trainlssvm({X,Y,'f',gam,sig2});
Xnew=Z((end-order+1):end)';
Z(end+1)=simlssvm({X,Y,'f',gam,sig2},{alpha,b},Xnew);
Xnew=Z((end-order+1):end)';
Z(end+1)=simlssvm({X,Y,'f',gam,sig2},{alpha,b},Xnew);
horizon=length(Z)-order;
Zpt=predict({X,Y,'f',gam,sig2},Z(1:order),horizon);
rmse = sqrt(mse(Zpt-Z(order+1:end)));
disp(' ')
close all;
disp(['Root mean squared error : ', num2str(rmse)])
%figure, hold on, plot(Zpt(1:end),'r'), plot(Z,'b');
plot([Z(order+1:end) Zpt]);