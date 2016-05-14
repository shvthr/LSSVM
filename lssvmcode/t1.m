close all
gam = 100;
sig2 = 530; 
type = 'classification';
krnl_type = 'lin_kernel'; %'lin_kernel' , 'RBF_kernel' , 'poly_kernel'
load('iris.mat');

cnt=1;
gammas=[0.1 1 3 5 10 25 50 100];
figure(1)
for gamma=gammas
    mdl_in = {X, Y, type, gamma, 0.5, 'RBF_kernel'}; %'preprocess'
    [alpha,b] = trainlssvm(mdl_in);
    
    Yc = simlssvm(mdl_in, {alpha,b}, Xt);
    acc = sum(Yc==Yt)/length(Yc) * 100;
%     figure(cnt)

    subplot(2,4,cnt)
    plotlssvm(mdl_in, {alpha,b});
    title(['gamma=' num2str(gamma) ',accuracy=' num2str(acc)]);
    cnt=cnt+1;
    fprintf('acc=%.2f \n',acc);
%     hold off
end