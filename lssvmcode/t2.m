close all
gam = 100;
sig2 = 530; 
type = 'classification';
krnl_type = 'lin_kernel'; %'lin_kernel' , 'RBF_kernel' , 'poly_kernel'
load('iris.mat');
cnt=1;
sigs=[0.001 0.01 0.1 1 2 3 7 15];
for sig=sigs
    mdl_in = {X, Y, type, gam, sig, 'RBF_kernel'}; %'preprocess'
    [alpha,b] = trainlssvm(mdl_in);
    
    Yc = simlssvm(mdl_in, {alpha,b}, Xt);
    acc = sum(Yc==Yt)/length(Yc) * 100;
   
    subplot(2,4,cnt)
    %figure(cnt)
    plotlssvm(mdl_in, {alpha,b});
    title(['sigma2=' num2str(sig) ',accuracy=' num2str(acc)]);
    cnt=cnt+1;
    fprintf('acc=%.2f \n',acc);
end