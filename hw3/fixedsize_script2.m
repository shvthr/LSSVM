clear
close all


%
% make a subset
%
X = 3.*randn(25,2);
ssize = 3;
sig2 = 1;
indices_subset = 1:ssize;

  
%
% make a figure
%
subplot(1,2,1);
plot(X(:,1),             X(:,2),'b*'); hold on;
plot(X(indices_subset,1),X(indices_subset,2),'ro','linewidth',6); hold off; 
title('original space')

%
% transform the data in feature space
%
features = AFEm(X(indices_subset,:),'RBF_kernel',sig2,X);
subplot(1,2,2);
plot3(features(:,1),             features(:,2),             features(:,3),'k*'); hold on;
plot3(features(indices_subset,1),features(indices_subset,2),features(indices_subset,3),'ro','linewidth',6); hold off;
title('feature space')


