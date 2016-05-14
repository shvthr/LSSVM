function cost = crossvalidatelssnd(model,L,estfct,combinefct)

X_all = model.xtrain;
Y_all = model.ytrain;
nb_data = size(X_all,1);

% LS-SVMlab
eval('model = initlssvm(model{:});',' ');
model.status = 'changed';

eval('L;','L=min(round(sqrt(size(model.xfull,1))),10);');
eval('estfct;','estfct=''mse'';');
eval('combinefct;','combinefct=''mean'';');

gams = model.gamcsa; nus = model.nucsa; try sig2s = model.kernel_parscsa; catch, sig2s = [];end

%initialize: no incremental  memory allocation
costs = zeros(L,length(gams));
Indices = crossvalind('Kfold', nb_data, L);
codetype = model.codetype;
code = model.code;

% check whether there are more than one gamma or sigma
for j =1:numel(gams)
    for k=1:L
        Y{k} = Y_all(Indices ~= k,:);
        Y_test{k} = Y_all(Indices == k,:);
        X{k} = X_all(Indices ~= k,:);
        X_test{k} = X_all(Indices == k,:);
    end
    for k=1:L
        model = initlssvm(X{k},Y{k},model.type,[],[],model.kernel_type);
        if strcmp(model.kernel_type,'RBF_kernel') || strcmp(model.kernel_type,'RBF4_kernel')
            model = changelssvm(changelssvm(changelssvm(model,'nu',nus(j)),'gam',gams(j)),'kernel_pars',sig2s(j));
        elseif strcmp(model.kernel_type,'lin_kernel')
            model = changelssvm(changelssvm(model,'nu',nus(j)),'gam',gams(j));
        elseif strcmp(model.kernel_type,'poly_kernel')
            model = changelssvm(changelssvm(changelssvm(model,'nu',nus(j)),'gam',gams(j)),'kernel_pars',[sig2s(1,j);sig2s(2,j)]);
        else
            model = changelssvm(changelssvm(changelssvm(model,'nu',nus(j)),'gam',gams(j)),'kernel_pars',[sig2s(1,j);sig2s(2,j);sig2s(3,j)]);
        end
        
        model = lssndMATLAB(model);   
        model.codetype = codetype;
        model.code = code;
        
        yh = simlssvm(model,X_test{k});
        [~, yt] = max(Y_test{k},[],2);
        
        if (sum(yt == model.y_dim) == 0) 
            costs(k,j) = feval(estfct,yt,yh);
        else
            costs(k,j) = 0.25*feval(estfct,yt(yt==model.y_dim),yh(yt==model.y_dim));
            costs(k,j) = costs(k,j) + 0.75*feval(estfct,yt(yt~=model.y_dim),yh(yt~=model.y_dim));
        end

    end
end
cost = feval(combinefct, costs);