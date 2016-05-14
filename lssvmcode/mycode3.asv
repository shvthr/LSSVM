close all
gam = 0.4;
% sig2 = 530; 
type = 'classification';
krnl_type = 'lin_kernel'; %'lin_kernel' , 'RBF_kernel' , 'poly_kernel'
load('ripley.mat');

   
%clasification
mdl_in = {X, Y, type, gam, sig2, 'RBF_kernel'}; %'preprocess'
[alpha,b] = trainlssvm(mdl_in);
plotlssvm(mdl_in, {alpha,b});

[Yc, Ylatent] = simlssvm(mdl_in, {alpha,b}, Xt);
accuracy = sum(Yc==Yt)/length(Yc) * 100;
fprintf('acc=%.2f \n',accuracy);
roc(Ylatent, Yt)

mdl_in = {X, Y, type, 0.0376, [], 'lin_kernel'}; %'preprocess'
[alpha,b] = trainlssvm(mdl_in);
[Yc, Ylatent2] = simlssvm(mdl_in, {alpha,b}, Xt);
roc(Ylatent2, Yt)
plotlssvm(mdl_in, {alpha,b});
%%
for deg=1:5
    mdl_in = {X, Y, type, gam, [1 deg], 'poly_kernel'}; %'preprocess'
    [alpha,b] = trainlssvm(mdl_in);
    
    Yc = simlssvm(mdl_in, {alpha,b}, Xt);
    acc = sum(Yc==Yt)/length(Yc) * 100;
    fprintf('acc=%.2f \n',acc);
end


%%
cnt=1;
sigs=[0.01 0.1 0.5 1 2 5 10];
for sig=sigs
    mdl_in = {X, Y, type, gam, sig, 'RBF_kernel'}; %'preprocess'
    [alpha,b] = trainlssvm(mdl_in);
    
    Yc = simlssvm(mdl_in, {alpha,b}, Xt);
    acc = sum(Yc==Yt)/length(Yc) * 100;
    
    subplot(3,3,cnt)
    plotlssvm(mdl_in, {alpha,b});
    title(['sig2=' num2str(sig) ',acc=' num2str(acc)]);
    cnt=cnt+1;
    fprintf('acc=%.2f \n',acc);
end


%%
cnt=1;
gammas=[0.01 0.1 1 5 10 50 100];
for gamma=gammas
    mdl_in = {X, Y, type, gamma, 0.5, 'RBF_kernel'}; %'preprocess'
    [alpha,b] = trainlssvm(mdl_in);
    
    Yc = simlssvm(mdl_in, {alpha,b}, Xt);
    acc = sum(Yc==Yt)/length(Yc) * 100;
    
    subplot(3,3,cnt)
    plotlssvm(mdl_in, {alpha,b});
    title(['gamma=' num2str(gamma) ',acc=' num2str(acc)]);
    cnt=cnt+1;
    fprintf('acc=%.2f \n',acc);
end