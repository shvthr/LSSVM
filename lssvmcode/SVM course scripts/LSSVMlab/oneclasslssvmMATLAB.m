function [model,H] = oneclasslssvmMATLAB(model) 
% Only for intern LS-SVMlab use;
%
% MATLAB implementation of the OneClass LS-SVM algorithm. This is slower
% than the C-mex implementation, but it is more reliable and flexible;
%
%
% This implementation is quite straightforward, based on MATLAB's
% backslash matrix division (or PCG if available) and total kernel
% matrix construction. It has some extensions towards advanced
% techniques, especially applicable on small datasets (weighed
% LS-SVM, gamma-per-datapoint)

% Copyright (c) 2011,  KULeuven-ESAT-SCD, License & help @ http://www.esat.kuleuven.be/sista/lssvmlab


assert(model.y_dim == 1, 'cannot have y>1 dimension for OneClass LS-SVM');

% remove outliers
selector = find(model.ytrain(:,1) == 1 & ~isnan(model.ytrain(:,1)));
nb_data=numel(selector);

% computation omega and H
H = kernel_matrix(model.xtrain(:, 1:model.x_dim), ...
    model.kernel_type, model.kernel_pars);


% initiate alpha and b
model.b = zeros(1,model.y_dim);
model.alpha = zeros(model.nb_data,model.y_dim);
   
invgam = model.gam(1,1).^-1; 
for t=1:nb_data, H(t,t) = H(t,t)+invgam; end

R = chol(H(selector,selector));
nu = R\(R'\ones(nb_data,1));

model.b(1) = -1/sum(nu);
model.alpha(selector,1) = -nu(:,1)*model.b(1);


