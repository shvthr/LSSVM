function cost = crossvalidatefslssvm(model,L,estfct,fs,svX,omega,omegaN,combinefct)
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
    
    nb_data = length(model.ytrain);
    sig=model.kernel_pars;
    gam=model.gam;
    block_size = floor(nb_data/L);
    subsetsize = size(svX,1);
    if (nb_data<50000)
    %Perform the Nystrom Approximation
        if (strcmp(model.kernel_type,'RBF_kernel'))
            Xfeatures = modAFEm(model.kernel_type,sig,exp(-omega./(2*sig)),exp(-omegaN./sig));
        elseif (strcmp(model.kernel_type,'lin_kernel'))
            Xfeatures = modAFEm(model.kernel_type,sig,omega,omegaN);
        elseif (strcmp(model.kernel_type,'poly_kernel'))
            Xfeatures = modAFEm(model.kernel_type,sig,(omega+sig(1)).^sig(2),(omegaN + sig(1)).^sig(2));
        end;
    end;
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
        trainX = model.xtrain(train,:);
        testY = model.ytrain(validation,:);
        trainY = model.ytrain(train,:);
        testX = model.xtrain(validation,:);
        datapoints=size(trainY,1);
        if (datapoints<50000)
            trainfeatures = Xfeatures(train,:);
            testfeatures = Xfeatures(validation,:);
            [~,~,testYh]=ridgeregress(trainfeatures,trainY,gam,svX,testfeatures);
            if ~(model.type(1)=='c')
                costs(l,j) = feval(estfct,testYh - testY);
            else
                costs(l,j) = feval(estfct,testY,sign(testYh));
            end;
        elseif (datapoints>=50000)
            blocks=ceil(datapoints/50000);
            if (strcmp(model.kernel_type,'RBF_kernel'))
                omeganew = exp(-omega./(2*sig));
                [eigvec,eigvals] = eig(omeganew+2*eye(size(omeganew,1))); % + jitter factor
                eigvals = diag(eigvals); 
                clear omeganew;
                eigvals = (eigvals-2)/subsetsize;
                peff = eigvals>eps;
                eigvals = eigvals(peff);
                eigvec = eigvec(:,peff); clear peff; 
            elseif (strcmp(model.kernel_type,'lin_kernel') || strcmp(model.kernel_type,'poly_kernel'))
                if (strcmp(model.kernel_type,'lin_kernel'))
                    omeganew = omega;
                elseif(strcmp(model.kernel_type,'poly_kernel'))
                    omeganew = (omega + sig(1)).^sig(2);
                end;
                [eigvec,eigvals] = eig(omeganew+2*eye(size(omeganew,1))); % + jitter factor
                eigvals = diag(eigvals); 
                clear omeganew;
                eigvals = (eigvals-2)/subsetsize;
                peff = eigvals>eps;
                eigvals = eigvals(peff);
                eigvec = eigvec(:,peff); clear peff; 
            end;
            newsize = size(eigvec,2);
            K=zeros(newsize,newsize);
            OneK = zeros(1,newsize);
            OneY = 0;
            KY = zeros(newsize,1);
	        One = 0;
            for k=1:blocks
                %fprintf('Performing Block Iteration for block = %d in total block = %d\n',k,blocks);
                if (k==blocks)
                    blockindex = (k-1)*50000+1;
                    Xtrain = trainX(blockindex:datapoints,:);
                    Ytrain = trainY(blockindex:datapoints,:);
                else
                    blockindex1 = (k-1)*50000+1;
                    blockindex2 = (k)*50000;
                    Xtrain = trainX(blockindex1:blockindex2,:);
                    Ytrain = trainY(blockindex1:blockindex2,:);
                end;
                XXh1 = sum(Xtrain.^2,2)*ones(1,size(svX,1));
                XXh2 = sum(svX.^2,2)*ones(1,size(Xtrain,1));
                omegaN = XXh1+XXh2' - 2*Xtrain*svX';
                XXh1 = sum(testX.^2,2)*ones(1,size(svX,1));
                XXh2 = sum(svX.^2,2)*ones(1,size(testX,1));
                testomegaN = XXh1+XXh2' - 2*testX*svX';
                len1 = size(omegaN,1);
                len2 = size(testomegaN,1);
                clear XXh1;
                clear XXh2;
                if (strcmp(model.kernel_type,'RBF_kernel'))
                    omegaN = exp(-omegaN./sig);
                    testomegaN = exp(-testomegaN./sig);
                    Xfeatures = omegaN*eigvec; clear omegaN
                    Xfeatures = repmat((1 ./ sqrt(eigvals))',len1,1).*Xfeatures;
                    testfeatures = testomegaN*eigvec; clear testomegaN
                    testfeatures = repmat((1 ./ sqrt(eigvals))',len2,1).*testfeatures;
                elseif (strcmp(model.kernel_type,'lin_kernel'))
                    Xfeatures = omegaN*eigvec; clear omegaN
                    Xfeatures = repmat((1 ./ sqrt(eigvals))',len1,1).*Xfeatures;
                    testfeatures = testomegaN*eigvec; clear testomegaN
                    testfeatures = repmat((1 ./ sqrt(eigvals))',len2,1).*testfeatures;
                elseif (strcmp(model.kernel_type,'poly_kernel'))
                    omegaN=(omegaN + sig(1)).^(sig(2));
                    testomegaN = (testomegaN + sig(1)).^(sig(2));
                    Xfeatures = omegaN*eigvec; clear omegaN
                    Xfeatures = repmat((1 ./ sqrt(eigvals))',len1,1).*Xfeatures;
                    testfeatures = testomegaN*eigvec; clear testomegaN
                    testfeatures = repmat((1 ./ sqrt(eigvals))',len2,1).*testfeatures;
                end;
                onevector = ones(1,size(Xtrain,1));
                K = K + Xfeatures'*Xfeatures; 
                OneK = OneK + onevector*Xfeatures;
                OneY = OneY + onevector*Ytrain;
                One = One + onevector*onevector';
                KY = KY + Xfeatures'*Ytrain;
                clear Xfeatures; 
                clear Ytrain; clear Xtrain;
            end;
            Xe = [K OneK';OneK One] + inv(gam).*eye(newsize+1);
            Ye = [KY;OneY];
            sol = pinv(Xe)*Ye;
            w = sol(1:end-1,:);
            b = sol(end,:);
            testYh = testfeatures*w+b;
            if ~(model.type(1)=='c')
                costs(l,j) = feval(estfct,testYh - testY);
            else
                costs(l,j) = feval(estfct,testY,sign(testYh));
            end;
            clear trainX;
            clear trainY;
        end;
    end;
end;
cost = feval(combinefct, costs);
