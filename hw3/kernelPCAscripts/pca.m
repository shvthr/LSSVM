function [eigvals,eigvec] = pca(X, N)
%function [eigvals,eigvec] = pca(X, N)
%Principal Components Analysis
%  computes the eigenvectors and eigenvalues of the covariance matrix of the dataset X. 
%
%Inputs
%  X: data matrix
%  N: if specified, only the first N PCs will be returned
%
%Outputs
%  eigvec : eigenvectors, representing the principal components.
%  eigvals: eigenvalues, giving the variance of X along the corresponding PCs.
%

if nargin == 1
   N = size(X, 2);
end

%
% Find the sorted eigenvalues of the data covariance matrix
%

if nargout<2
   temp_eigvals = eig(cov(X));

   % To make sure eigenvalues returned in descending order
   eigvals = sort(-temp_eigvals); eigvals = -eigvals(1:N);
   return
end

% Use eig function unless fraction of eigenvalues required is tiny
if (N/size(X, 2)) > 0.04
   [temp_eigvec, temp_eigvals] = eig(cov(X));
else
   options.disp = 0;
   [temp_eigvec, temp_eigvals] = eigs(cov(X), N, 'LM', options);
end
temp_eigvals = diag(temp_eigvals);

% To make sure eigenvalues returned in descending order, but just
[eigvals perm] = sort(-temp_eigvals);
eigvals = -eigvals(1:N);
eigvec = temp_eigvec(:, perm(1:N));
