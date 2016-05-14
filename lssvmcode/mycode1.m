clc
randn('state',100)
X1 = +1 + randn(50,2);
randn('state',200);    
X2 = -1 + randn(51,2);
X = [X1; X2];  
Y1 = ones(size(X1,1),1); 
Y2 = -1*ones(size(X2,1),1);
Y = [Y1; Y2];

%test data
x = -5.0:0.25:5.0;
y = -5.0:0.25:5.0;
[xt, yt] = meshgrid(x,y);
grid = [xt(:) yt(:)];

%bayes classifier
%  pc1 = normpdf2(grid, [-1; -1], [1 0;0 1]);
%  pc2 = normpdf2(grid, [+1; +1], [1 0;0 1]);
%  class = pc1>=pc2;

%estimate bayes classfier
mean1 = mean(X1); mean2 = mean(X2);
cov1 = cov(X1); cov2 = cov(X2);
pc1 = normpdf2(grid, mean2, cov2);
pc2 = normpdf2(grid, mean1, cov1);
class = pc1>=pc2;

%visualization
grid = reshape(class,length(x),length(y));
contourf(x,y,grid,2);hold on;    
hold on
plot(X1(:,1),X1(:,2),'ro'); 
plot(X2(:,1),X2(:,2),'bo'); 
title('Plot of Samples');     
%legend([pos neg],'positive data','negative data');
hold off