addpath(genpath('E:\Academic\SVN\thesis\code\lssvmlab'));

X=(-10:0.1:10)';
Y = cos(X) + cos(2*X) + 0.1.*rand(length(X),1);

Xtrain = X(1:2:length(X));
Ytrain = Y(1:2:length(Y));
Xtest = X(2:2:length(X));
Ytest = Y(2:2:length(Y));

%%
sigs = [0.1 10 0.1 10 0.1 10]; gammas=[1 1 10 10 70 70];
figure;
for i=1:length(gammas)
    gam = gammas(i);
    sig2 = sigs(i);
    mdl_in = {Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel'};
    [alpha,b] = trainlssvm(mdl_in);
 
    subplot(4, 2, i);
    plotlssvm(mdl_in, {alpha,b});
  
    YtestEst = simlssvm(mdl_in, {alpha,b},Xtest);
    plot(Xtest,Ytest,'b.');
    hold on;
    plot(Xtest,YtestEst,'g+');
    %legend('Ytest','YtestEst');
    title(['sig2=' num2str(sig2) ',gam=' num2str(gam)]);
    hold off;
end


%%
cost_crossval = crossvalidate({Xtrain,Ytrain,'f',gam,sig2},10);
cost_loo = leaveoneout({Xtrain,Ytrain,'f',gam,sig2});

optFun = 'gridsearch';
globalOptFun = 'csa';
mdl_in = {Xtrain,Ytrain,'f',[],[],'RBF_kernel',globalOptFun};
[gam,sig2,cost] = tunelssvm(mdl_in, optFun,'crossvalidatelssvm',{10,'mse'})

%  mdl_in = {Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel'};
 [alpha,b] = trainlssvm({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel'});

 plotlssvm({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel'}, {alpha,b});

 YtestEst = simlssvm({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel'}, {alpha,b},Xtest);
 plot(Xtest,Ytest,'r.');
 hold on;
 plot(Xtest,YtestEst,'g+');
 title(['sig2=' num2str(sig2) ',gam=' num2str(gam)]);
legend('Ytest','YtestEst');

%%
optFun = 'gridsearch';
globalOptFun = 'csa';
mdl_in = {Xtrain,Ytrain,'f',[],[],'RBF_kernel',globalOptFun};
tic
for i=1:20
    [gam_csa_grid(i),sig2_csa_grid(i),cost_csa_grid(i)] = tunelssvm(mdl_in, optFun,'crossvalidatelssvm',{10,'mse'});
end
t1=toc;
t1=t1/20;

[c,idx]=min(cost_csa_grid); a=gam_csa_grid(idx);
fprintf('min=%0.5f \nmean=%0.5f \nvar=%0.5f \n', c, mean(cost_csa_grid), var(cost_csa_grid))
b=sig2_csa_grid(idx);
fprintf('t=%0.5f s \ngam=%0.5f \nsig2=%0.5f \n', mean(t1), a, b)

%%
optFun = 'simplex';
globalOptFun = 'csa';
mdl_in = {Xtrain,Ytrain,'f',[],[],'RBF_kernel',globalOptFun};
tic
for i=1:20
    [gam_csa_simplex(i),sig2_csa_simplex(i),cost_csa_simplex(i)] = tunelssvm(mdl_in, optFun,'crossvalidatelssvm',{10,'mse'});
end
t1=toc;
t1=t1/20;

[c,idx]=min(cost_csa_simplex); a=gam_csa_simplex(idx); b=sig2_csa_simplex(idx);
fprintf('min=%0.5f \nmean=%0.5f \nvar=%0.5f \n', c, mean(cost_csa_simplex), var(cost_csa_simplex))
fprintf('t=%0.5f s \ngam=%0.5f \nsig2=%0.5f \n', mean(t1), a, b)
%%
optFun = 'gridsearch';
globalOptFun = 'ds';
mdl_in = {Xtrain,Ytrain,'f',[],[],'RBF_kernel',globalOptFun};
tic
for i=1:20
    [gam_ds_grid(i),sig2_ds_grid(i),cost_ds_grid(i)] = tunelssvm(mdl_in, optFun,'crossvalidatelssvm',{10,'mse'});
end
t1=toc;
t1=t1/20;

[c,idx]=min(cost_ds_grid); a=gam_ds_grid(idx); b=sig2_ds_grid(idx);
fprintf('min=%0.5f \nmean=%0.5f \nvar=%0.5f \n', c, mean(cost_ds_grid), var(cost_ds_grid))
fprintf('t=%0.5f s \ngam=%0.5f \nsig2=%0.5f \n', mean(t1), a, b)

%%
optFun = 'simplex';
globalOptFun = 'ds';
mdl_in = {Xtrain,Ytrain,'f',[],[],'RBF_kernel',globalOptFun};
tic
for i=1:20
    [gam_ds_simplex(i),sig2_ds_simplex(i),cost_ds_simplex(i)] = tunelssvm(mdl_in, optFun,'crossvalidatelssvm',{10,'mse'});
end
t1=toc;
t1=t1/20;

[c,idx]=min(cost_ds_simplex); a=gam_ds_simplex(idx); b=sig2_ds_simplex(idx);
fprintf('min=%0.5f \nmean=%0.5f \nvar=%0.5f \n', c, mean(cost_ds_simplex), var(cost_ds_simplex))
fprintf('t=%0.5f s \ngam=%0.5f \nsig2=%0.5f \n', mean(t1), a, b)

%%
sig2 = 0.5; gam = 10;
criterion_L1 = bay_lssvm({Xtrain,Ytrain,'f',gam,sig2},1)
criterion_L2 = bay_lssvm({Xtrain,Ytrain,'f',gam,sig2},2)
criterion_L3 = bay_lssvm({Xtrain,Ytrain,'f',gam,sig2},3)


%%
gam=100; sig2=0.05;
[~,alpha,b] = bay_optimize({Xtrain,Ytrain,'f',gam,sig2}, 1);
[~,gam] = bay_optimize({Xtrain,Ytrain,'f',gam,sig2},2);
[~,sig2] = bay_optimize({Xtrain,Ytrain,'f',gam,sig2},3);
sig2e = bay_errorbar({Xtrain,Ytrain,'f',gam,sig2},'figure');

%%
% load iris;
gam=5; sig2=0.75;
cnt=1;
for gam=[1 10 50 100]
    for sig2=[0.1 1 5 10]
        subplot(4,4,cnt);
        bay_modoutClass({X,Y,'c',gam,sig2},'figure');
        cnt=cnt+1;
    end
end

%%
X = 10.*rand(100,3)-3;
Y = cos(X(:,1)) + cos(2*(X(:,1))) +0.3.*randn(100,1);
[selected, ranking, costs2] = bay_lssvmARD({X,Y,'class', 100, 0.1});

%%
X = (-10:0.2:10)';
Y = cos(X) + cos(2*X) +0.1.*rand(size(X));
out = [15 17 19];
Y(out) = 0.7+0.3*rand(size(out));
out = [41 44 46];
Y(out) = 1.5+0.2*rand(size(out));

mdl_in = {X, Y,'f', 100, 0.1,'RBF_kernel'};
[alpha,b] = trainlssvm(mdl_in);
plotlssvm(mdl_in, {alpha,b});

%%
model = initlssvm(X,Y,'f',[],[],'RBF_kernel');
costFun = 'rcrossvalidatelssvm';
wfuns = {'whuber', 'whampel', 'wlogistic', 'wmyriad'};
for i=1:4
    wFun = wfuns{i};
    model = tunelssvm(model,'simplex',costFun,{10,'mae'},wFun);
    model = robustlssvm(model);
    subplot(2,2,i);
    plotlssvm(model);
end

%% hw

addpath(genpath('E:\Education\KULEUVEN\2-Semester2\SVM\SVM course scripts\LSSVMlab'));
cnt = 1;
%orders = 50; 
 orders=[2 8 10 15 25 35 55 75 85 95 100];
my_mse=[];
for order=orders
    %order = 10;
    X = windowize(Z,1:(order+1));
    Y = X(:,end);
    X = X(:,1:order);
    %..................
    %gam = 1; sig2 = 0.01;
    
    [alpha,b] = trainlssvm({X,Y,'f',gam,sig2});
    Xnew = Z((end-order+1):end)';
    Z(end+1) = simlssvm({X,Y,'f',gam,sig2},{alpha,b},Xnew);
    %...............
    optFun = 'gridsearch'; globalOptFun = 'ds';
    mdl_in = {X,Y,'f',[],[],'RBF_kernel', globalOptFun};
    [gam,sig2,cost] = tunelssvm(mdl_in, optFun,'crossvalidatelssvm',{10,'mse'});
    
    %gam = 296430; sig2 = 127.7589;
    Zwork = Ztest;
    horizon = length(Zwork)-order;
    Zpt = predict({X,Y,'f',gam,sig2,}, Zwork(1:order), horizon);
    my_mse(cnt) = (mse(Zwork(order+1:end) - Zpt));
    disp(['Root mean squared error : ', num2str(rmse)])
    plot([Zwork(order+1:end) Zpt]);
    cnt = cnt+1;
end
plot(orders, my_mse)

