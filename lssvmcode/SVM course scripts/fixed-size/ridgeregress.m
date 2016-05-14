function [w,b,Yt] = ridgeregress(X,Y,gam,svX,Xt)
% Linear ridge regression
% 
% >> [w, b]     = ridgeregress(X, Y, gam)
% >> [w, b, Yt] = ridgeregress(X, Y, gam, Xt)
% 
% Ordinary Least squares with a regularization parameter (gam).
% 
% Full syntax
% 
% >> [w, b, Yt] = ridgeregress(X, Y, gam, Xt)
% 
% Outputs    
%   w     : d x 1 vector with the regression coefficients
%   b     : bias term
%   Yt(*) : Nt x 1 vector with predicted outputs of test data
% Inputs    
%   X     : N x d matrix with the inputs of the training data
%   Y     : N x 1 vector with the outputs of the training data
%   gam   : Regularization parameter
%   Xt(*) : Nt x d matrix with the inputs of the test data
% 
% See also:
% bay_rr,bay_lssvm



% Copyright (c) 2002,  KULeuven-ESAT-SCD, License & help @ http://www.esat.kuleuven.ac.be/sista/lssvmlab

  
if size(X,1)~=size(Y,1),
  error('X and Y need to have the same number of data points');
end
if size(Y,2)~=1,
  error('Only handling one-dimensional output');
end
if nargin==4 && size(Xt,2)~=size(X,2),
  error('Training input and test inputs need to have the same dimension');
end
  
[nD,nx] = size(X);
typesize = size(svX,1);
if nx>nD, warning('dim datapoints larger than number of datapoints...');end

if (nD<50000)
    H = [X ones(nD,1)]'*[X ones(nD,1)] + inv(gam).*eye(nx+1);
    sol = [X ones(nD,1)]'*Y;
    sol = pinv(H)*sol;
    clear X;
    w = sol(1:end-1);
    b = sol(end);
elseif (nD>50000)
    %Construct blocks 
    datapoints=nD;
    blocks=ceil(datapoints/50000);
    K= zeros(typesize,typesize);
    OneK = zeros(1,typesize);
    OneY = 0;
    KY = zeros(typesize,1);
    One = 0;
    for k=1:blocks
        %fprintf('Performing Block Iteration for block = %d in total block = %d\n',k,blocks);
        if (k==blocks)
            blockindex = (k-1)*50000+1;
            Xtrain = X(blockindex:datapoints,:);
            Ytrain = Y(blockindex:datapoints,:);
        else
            blockindex1 = (k-1)*50000+1;
            blockindex2 = (k)*50000;
            Xtrain = X(blockindex1:blockindex2,:);
            Ytrain = Y(blockindex1:blockindex2,:);
        end;
        %Xfeatures = kernel_matrix(Xtrain,kernel_type,sig,svX);
        Xfeatures = Xtrain;
        onevector = ones(1,size(Xtrain,1));
        K = K + Xfeatures'*Xfeatures; 
        OneK = OneK + onevector*Xfeatures;
        OneY = OneY + onevector*Ytrain;
        One = One + onevector*onevector';
        KY = KY + Xfeatures'*Ytrain;
        clear Xfeatures; 
        clear Ytrain; clear Xtrain;
    end;
    Xe = [K OneK';OneK One] + inv(gam).*eye(typesize+1);
    Ye = [KY;OneY];
    sol = pinv(Xe)*Ye;
    clear Xe; 
    w = sol(1:end-1);
    b = sol(end);
end;
%Xe = [X ones(nD,1)];
%H = Xe'*Xe + inv(gam).*eye(nx+1);

%sol = Xe'*Y;
%sol = pinv(H)*sol;
%w = sol(1:end-1);
%b = sol(end);


if nargin<4, return; end
Yt = Xt*w+b;
  
  
   
