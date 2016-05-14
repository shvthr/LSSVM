function cost = modcrossvalidatelssvm(model,L,estfct,fs,svX,omega,omegaN,combinefct)
% Estimate the model performance of a model with l-fold crossvalidation
%
%%%%%%%%%%%%%%%%%%%%%
% INTERNAL FUNCTION %
%%%%%%%%%%%%%%%%%%%%%
% Estimate the model performance of a model with fast l-fold crossvalidation.
% Implementation based on "De Brabanter et al., Computationsl Statistics & Data Analysis, 2010"

% Copyright (c) 2010,  KULeuven-ESAT-SCD, License & help @% http://www.esat.kuleuven.ac.be/sista/lssvmlab
eval('L;','L=min(round(sqrt(size(model.xfull,1))),10);')
eval('estfct;','estfct=''mse'';');
eval('combinefct;','combinefct=''mean'';');

gams = model.gam; try sig2s = model.kernel_pars; catch, sig2s = [] ; end

%initialize: no incremental  memory allocation
costs = zeros(L,length(gams));

% check whether there are more than one gamma or sigma
for j =1:numel(gams)
    if strcmp(model.kernel_type,'RBF_kernel') || strcmp(model.kernel_type,'RBF4_kernel')
        model.gam = gams(j);
        model.kernel_pars = sig2s(j);
    elseif strcmp(model.kernel_type,'lin_kernel')
        model.gam = gams(j);
    elseif strcmp(model.kernel_type,'poly_kernel')
        model.gam = gams(j);
        model.kernel_pars = [sig2s(1,j);sig2s(2,j)];
    %else
    %    model = changelssvm(changelssvm(model,'gam',gams(j)),'kernel_pars',[sig2s(1,j);sig2s(2,j);sig2s(3,j)]);
    end;
    
    X = model.xtrain;
    Y = model.ytrain;
    nb_data = length(Y);
    sig=model.kernel_pars;
    gam=model.gam;
    if (strcmp(model.kernel_type,'RBF_kernel'))
        xomega = exp(-omega./(2*sig));
        Xfeatures=exp(-omegaN./(2*sig));
    elseif (strcmp(model.kernel_type,'lin_kernel'))
        xomega=omega;
        Xfeatures=omegaN;
    elseif (strcmp(model.kernel_type,'poly_kernel'))
        xomega=omega;
        Xfeatures=(omegaN + sig(1)).^sig(2);
    end;
    block_size = floor(nb_data/L);
    %start loop over l validations
    %par
    for l = 1:L
        %divide data in validations set and training set
        if (l==L)
            train = 1:block_size*(l-1); % not used
            validation = block_size*(l-1)+1:nb_data;
        else
            train = [1:block_size*(l-1) block_size*l+1:nb_data];
            validation = block_size*(l-1)+1:block_size*l;
        end;
        testY = Y(validation,:);
        trainY = Y(train,:);
        trainfeatures = Xfeatures(train,:);
        testfeatures = X(validation,:);
        [alpha,b,testYh]=modridgeregress(trainfeatures,trainY,gam,model.kernel_type,sig,svX,xomega,testfeatures);
        if ~(model.type(1)=='c')
            costs(l,j) = feval(estfct,testYh - testY);
        else
            costs(l,j) = feval(estfct,testY,sign(testYh));
        end;
    end;
end
cost = feval(combinefct, costs);