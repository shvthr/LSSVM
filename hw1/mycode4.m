addpath(genpath('E:\Education\KULEUVEN\2-Semester2\SVM\SVM-rep\lssvmcode\SVM course scripts\LSSVMlab'));

close all
gam = 0.1;
sig2 = 20;
type = 'classification';
krnl_type = 'lin_kernel'; %'lin_kernel' , 'RBF_kernel' , 'poly_kernel'
load('diabetes.mat');
X = trainset; Y=labels_train; Xt=testset; Yt=labels_test;

idx = randperm(size(X,1));
nall = size(X,1); ntrn = 80;
Xtrain=X(idx(1:ntrn), :);
Ytrain=Y(idx(1:ntrn));
Xval = X(idx(ntrn+1:nall), :);
Yval = Y(idx(ntrn+1:nall));

mdl_in = {X, Y, type, [], [], 'RBF_kernel', 'ds'}; %csa ds
gamma=[]; cost=[]; sigs2=[];
for i=1:30
    [gamma(i,1), sigs2(i,1), cost(i,1)] = tunelssvm(mdl_in, 'gridsearch', 'crossvalidatelssvm', {10, 'misclass'});
    %[gamma(i,1), sigs2, cost(i,1)] = tunelssvm(mdl_in, 'gridsearch', 'crossvalidatelssvm', {10, 'misclass'});
    %[gamma(i,1), sig2(i), cost(i,1)] = tunelssvm(mdl_in, 'gridsearch', 'leaveoneoutlssvm', {'misclass'});
end
[~,indx] = min(cost);
cost(indx)
gam = gamma(indx)
sig2 = sigs2(indx)

mdl_in = {X, Y, type, gam, sig2, 'RBF_kernel'}; %'preprocess'
[alpha,b] = trainlssvm(mdl_in);
%plotlssvm(mdl_in, {alpha,b});

[Yc, Ylatent] = simlssvm(mdl_in, {alpha,b}, Xt);
accuracy = sum(Yc==Yt)/length(Yc) * 100;
fprintf('acc=%.2f \n',accuracy);
%roc(Ylatent, Yt)

% for sig2=[0.1 1 10]
%     for gamma=[1 10 100]
%         %mdl_in = {Xtrain, Ytrain, type, gamma, sig2, 'RBF_kernel'};
%         %[alpha,b] = trainlssvm(mdl_in);
%         %Yc = simlssvm(mdl_in, {alpha,b}, Xval);
%         %accuracy = sum(Yc==Yval)/length(Yval) * 100;
%         
%         mdl_in = {X, Y, type, gamma, sig2, 'RBF_kernel'};        
%         cost = crossvalidate(mdl_in, 10, 'misclass');
%         %cost = leaveoneout(mdl_in, 'misclass');
%         fprintf('acc=%.2f ', cost);
%     end
%      fprintf('\n');
% end


%mdl_in = {X, Y, type, [], [], 'RBF_kernel', 'csa'};   
%[gamma, sig2, cost] = tunelssvm(mdl_in, 'simplex', 'crossvalidatelssvm', {10, 'misclass'});

% mdl_in = {Xtrain, Ytrain, type, 10, 2, 'RBF_kernel','preprocess'};
% [alpha,b] = trainlssvm(mdl_in);
% [Yc, Ylatent] = simlssvm(mdl_in, {alpha,b}, Xval);
% roc(Ylatent, Yval)