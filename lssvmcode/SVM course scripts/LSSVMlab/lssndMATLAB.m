function [model,H] = lssndMATLAB(model) 
% Only for intern LS-SVMlab use;
%
% MATLAB implementation of the LS-SND algorithm. 
%
% This implementation is quite straightforward, based on MATLAB's
% backslash matrix division (or PCG if available) and total kernel
% matrix construction. It has some extensions towards advanced
% techniques, especially applicable on small datasets (weighed
% LS-SND, gamma-per-datapoint)

% Copyright (c) 2011,  KULeuven-ESAT-SCD, License & help @ http://www.esat.kuleuven.be/sista/lssvmlab

% computation omega and H
omega = kernel_matrix(model.xtrain(model.selector, 1:model.x_dim), ...
    model.kernel_type, model.kernel_pars);

model.selector=~isnan(model.ytrain(:,1));
m = model.nb_data;
n = model.y_dim-1;

% assert(model.nu > n-1, 'nu <= n-1');

mu1 = (model.nu + n - 1)^-1 * (model.nu - 1)^-1;
mu2 = model.nu + n - 2;

% initiate alpha and b
model.b = zeros(1,n);
model.alpha = zeros(m,n);

H = omega;
alpha_ones = zeros(n,m*n); 
omega_ext = zeros(m*n,m*n);
y_ones = zeros(n,m*n);

for i=1:n
    alpha_ones(i,(i-1)*m+1:i*m) = ones(1,m);
    y_ones(i,(i-1)*m+1:i*m) = model.ytrain(model.selector,i)';
    for j=1:n
        Ys = model.ytrain(model.selector,i)*model.ytrain(model.selector,j)';
        if (i == j)
            omega_ext((i-1)*m+1:i*m,(j-1)*m+1:j*m) = -mu1 * mu2 * (H .* Ys) - (model.gam(1,1)^-1)*eye(m);
        else
            omega_ext((i-1)*m+1:i*m,(j-1)*m+1:j*m) = mu1 * (H .* Ys);
        end
    end
end

%finding the solution
solution = [zeros(n,n) y_ones; alpha_ones' omega_ext]\[-ones(n,1); zeros(m*n,1)];

%marking the solution
model.status = 'trained';

%unvectorization of the solution
for i=1:n
    model.alpha(:,i) = mu1 .* build_solution(model.ytrain,solution,mu2,n,m,i);
end

model.b = [-solution(1:n,:)' 0];
model.alpha(:,model.y_dim) = zeros(m,1);

end

function solution = build_solution(ytrain,s,mu,n,m,i)
   solution = zeros(m,1);
   for j=1:n
       alphas = s(n+(i-1)*m+1:n+i*m,:) .* ytrain(:,j);
       if (i == j)
           solution = solution + mu * alphas;
       else
           solution = solution - alphas;
       end
   end
end




