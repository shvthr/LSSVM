
cnt=1;
for cn=401%[2 4 16 64 256]
    
    nc;
    [lam,U] = kpca([samplesyin;samplesyang],'RBF_kernel',sig2,[],approx,nc);

    xd = denoise_kpca([samplesyin;samplesyang],'RBF_kernel',sig2,[],approx,nc);
    %h2=figure;
    subplot(2,3,cnt)
    hold on
    plot(samplesyin(:,1),samplesyin(:,2),'o');
    plot(samplesyang(:,1),samplesyang(:,2),'o');
    plot(xd(:,1),xd(:,2),'r+');
    hold off
    title(['n=' num2str(nc)]);
    cnt=cnt+1;
end


%%
nc = 4;
cnt=1;
for sig2=[0.01 0.1 0.5 1 10]
    
    sig2;
    [lam,U] = kpca([samplesyin;samplesyang],'RBF_kernel',sig2,[],approx,nc);

    xd = denoise_kpca([samplesyin;samplesyang],'RBF_kernel',sig2,[],approx,nc);
    %h2=figure;
    subplot(2,3,cnt)
    hold on
    plot(samplesyin(:,1),samplesyin(:,2),'o');
    plot(samplesyang(:,1),samplesyang(:,2),'o');
    plot(xd(:,1),xd(:,2),'r+');
    hold off
    title(['sig2=' num2str(sig2)]);
    cnt=cnt+1;
end

%%

nc = 4;
cnt=1;
for degree=1 %[1 2 3 4 5]
    
    degree
    [lam,U] = kpca([samplesyin;samplesyang],'poly_kernel',[1,degree],[],approx,nc);

    xd = denoise_kpca([samplesyin;samplesyang],'poly_kernel',[1,degree],[],approx,nc);
    %h2=figure;
    subplot(2,3,cnt)
    hold on
    plot(samplesyin(:,1),samplesyin(:,2),'o');
    plot(samplesyang(:,1),samplesyang(:,2),'o');
    plot(xd(:,1),xd(:,2),'r+');
    hold off
    title(['degree=' num2str(degree)]);
    cnt=cnt+1;
end

%%

sig2=0.05;              % set the kernel parameters
cnt=1;
for sig2=[0.001 0.005 0.01 0.2]
    K=kernel_matrix(X,'RBF_kernel',sig2);
    D=diag(sum(K));
    [U,lambda]=eigs(inv(D)*K,3);                              
    clust=sign(U(:,2)); 

    figure(1)
    subplot(2,2,cnt);
    scatter3(X(:,1),X(:,2),X(:,3),30,clust);
    title(['sig2=' num2str(sig2)]);
    
    figure(2)
    subplot(2,2,cnt);
    proj=K*U(:,2:3);
    scatter(proj(:,1),proj(:,2),15,clust);    
    title(['sig2=' num2str(sig2)]);
    cnt=cnt+1;
end

%%
randn('state',100)
X = 3.*randn(100,2);
X = round(X*100)/100;
cnt=1;
for sig2=[0.1 1 10 50]
    [subset r2] = kentrop(X, sig2);

    figure(1)
    subplot(2,2,cnt)
    plot(X(:,1),X(:,2),'b*'); hold on;
    plot(subset(:,1),subset(:,2),'ro','linewidth',6); hold off; 
    title(['sig2=' num2str(sig2)]);
    
    features = AFEm(subset,'RBF_kernel',sig2,X);    
    
    figure(2)
    subplot(2,2,cnt)
    plot3(features(:,1), features(:,2), features(:,3),'k*'); hold on;
    plot3(features(r2,1),features(r2,2),features(r2,3),'ro','linewidth',6); hold off;
    title(['sig2=' num2str(sig2)]);
    
    cnt=cnt+1;
end

%%
sigs = log([1.1:1:6.1]);
i=1;
for sigmafactor=sigs
    digitsdn(sigmafactor,i);
    i=i+1;
end

%%
sigs = [1 10 30 60 100];
i=1;
for sig2=sigs
    rr(i,:) = digitsdn2(sig2);
    i=i+1;
    
end

%%
fslssvm_script

