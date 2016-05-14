clc;
clear;

nb = input('\n How many datapoints? [400] '); if isempty(nb) nb=400; end;
sig = input('\n Dataset dispersion ? [0.3] '); if isempty(sig) sig=0.3; end;

nb=nb/2;


% construct data
leng = 1;
for t=1:nb, 
  yin(t,:) = [2.*sin(t/nb*pi*leng) 2.*cos(.61*t/nb*pi*leng) (t/nb*sig)]; 
  yang(t,:) = [-2.*sin(t/nb*pi*leng) .45-2.*cos(.61*t/nb*pi*leng) (t/nb*sig)]; 
  samplesyin(t,:)  = [yin(t,1)+yin(t,3).*randn   yin(t,2)+yin(t,3).*randn];
  samplesyang(t,:) = [yang(t,1)+yang(t,3).*randn   yang(t,2)+yang(t,3).*randn];
end


%% plot dataset
h=figure; hold on
plot(samplesyin(:,1),samplesyin(:,2),'o');
plot(samplesyang(:,1),samplesyang(:,2),'o');
xlabel('X_1');
ylabel('X_2');
title('Structured dataset');
disp('Press any key to continue');
pause;


% get user-defined parameters
nc = input('\n Number of components to be extracted? [6] '); if isempty(nc) nc=6; end;
sig2 = input('\n RBF kernel parameter sig2? [0.4] '); if isempty(sig2) sig2=0.4; end;
approx = input('\n Approximation technique ? 1 for ''Lanczos'', 2 for ''Nystrom'' [1] '); if isempty(approx) approx=1; end;

if approx ==1
    approx='eigs';
else
    approx='eign';
end


% calculate the eigenvectors in the feature space (principal components)

[lam,U] = kpca([samplesyin;samplesyang],'RBF_kernel',sig2,[],approx,nc);


% calculate the projections on the principal components
% Xax = -3:.1:3; Yax = -3.2:.1:3.2;
% [A,B] = meshgrid(Xax,Yax);
% grid = [reshape(A,prod(size(A)),1) reshape(B,1,prod(size(B)))'];
% k = kernel_matrix([samplesyin;samplesyang],'RBF_kernel',sig2,grid)';
% projections = k*U;

% plot the projections on the first component

% plot(samplesyin(:,1),samplesyin(:,2),'o');hold on;
% plot(samplesyang(:,1),samplesyang(:,2),'o');
% contour(Xax,Yax,reshape(projections(:,1),length(Yax),length(Xax)));
% title('Kernel PCA - Projections of the input space on the first principal component');
% figure(h);
% disp('Press any key to continue');
% pause;

% Denoise the data by minimizing the reconstruction error

xd = denoise_kpca([samplesyin;samplesyang],'RBF_kernel',sig2,[],approx,nc);
%h2=figure;
plot(samplesyin(:,1),samplesyin(:,2),'o');
plot(samplesyang(:,1),samplesyang(:,2),'o');
plot(xd(:,1),xd(:,2),'r+');
title('Kernel PCA - Denoised datapoints in red');

disp('Press any key to continue');
pause;

% Projections on the first component using linear PCA

dat=[samplesyin;samplesyang];
dat(:,1)=dat(:,1)-mean(dat(:,1));
dat(:,2)=dat(:,2)-mean(dat(:,2));


[lam_lin,U_lin] = pca(dat);


%proj_lin=grid*U_lin;

figure;

plot(samplesyin(:,1),samplesyin(:,2),'o');hold on;
plot(samplesyang(:,1),samplesyang(:,2),'o');
%contour(Xax,Yax,reshape(proj_lin(:,1),length(Yax),length(Xax)));

xdl=dat*U_lin(:,1)*U_lin(:,1)';
plot(xdl(:,1),xdl(:,2),'r+');

title('Linear PCA - Denoised data points using the first principal component');

