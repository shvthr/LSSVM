function [model,cost,O3] = tunelssvm(model, varargin)
% Tune the hyperparameters of the model with respect to the given performance measure
%
% 1. Using the functional interface:
%
% >> [gam, sig2, cost] = tunelssvm({X,Y,type,[],[],kernel,preprocess}, optfun, costfun, costargs)
%
%      Outputs
%        gam     : Optimal regularization parameter
%        sig2    : Optimal kernel parameter(s)
%        cost(*) : Estimated cost of the optimal hyperparameters
%      Inputs
%        X       : N x d matrix with the inputs of the training data
%        Y       : N x 1 vector with the outputs of the training data
%        type    : 'function estimation' ('f') or 'classifier' ('c')
%        kernel(*) : Kernel type (by default 'RBF_kernel')
%        preprocess(*) : 'preprocess'(*) or 'original'
%        optfun : Optimization function: 'simplex' or 'gridsearch'
%        costfun : Function estimating the cost-criterion: 'crossvalidatelssvm', 'leaveoneoutlssvm', 'gcrossvalidatelssvm'
%        costargs(*) : Cell with extra cost function arguments
%
% 2. Using the object oriented interface:
%
% >> model = tunelssvm(model, optfun, costfun, costargs)
%
%      Outputs
%        model            : Object oriented representation of the LS-SVM model with optimal hyperparameters
%      Inputs
%        model            : Object oriented representation of the LS-SVM model with initial hyperparameters
%        optfun(*)        : Optimization function (by default 'gridsearch')
%        costfun : Function estimating the cost-criterion: 'crossvalidatelssvm', 'leaveoneoutlssvm', 'gcrossvalidatelssvm'
%        optfun(*)        : Cell with extra cost function arguments
%
%  See also:
%    trainlssvm, crossvalidate, gridsearch, linesearch, simplex, csa

% Copyright (c) 2011,  KULeuven-ESAT-SCD, License & help @ http://www.esat.kuleuven.be/sista/lssvmlab

if iscell(model),
    model = initlssvm(model{:});
    func=1;
else
    func=0;
end

%
% defaults
%
if length(varargin)>=1, optfun = varargin{1}; else optfun='gridsearch';end
if length(varargin)>=2, costfun = varargin{2}; else costfun ='crossvalidatelssvm'; end
if length(varargin)>=3, costargs = varargin{3}; else costargs ={}; end

if strcmp(costfun,'crossvalidatelssvm') || strcmp(costfun,'crossvalidateoneclasslssvm')...
        || strcmp(costfun,'rcrossvalidatelssvm') || strcmp(costfun,'crossvalidatesparselssvm')
    if size(costargs,2)==1, error('Specify the number of folds for CV'); end
    [Y,omega] = helpkernel(model.xtrain,model.ytrain,model.kernel_type,costargs{2},0);
    costargs = {Y,costargs{1},omega,costargs{2}};
end

if strcmp(costfun,'crossvalidatelssnd') 
    if size(costargs,2)==1, error('Specify the number of folds for CV'); end
    costargs = {costargs{1},costargs{2}};
end

if strcmp(costfun,'crossvalidate2lp1')
    fprintf('\n')
    disp('-->> Cross-Validation for Correlated Errors: Determine optimal ''l'' for leave (2l+1) out CV')
    % if user specifies 'l'
    if numel(costargs)==1, luser = NaN; else luser = costargs{2};end
    [l,index] = cvl(model.xtrain,model.ytrain,luser); % First determine the 'l' for the CV
    fprintf(['\n -->> Optimal l = ' num2str(l)]);
    fprintf('\n')
    [Y,omega] = helpkernel(model.xtrain,model.ytrain,model.kernel_type,[],1);
    costargs = {Y,index,omega,costargs{1}};
end

if strcmp(costfun,'gcrossvalidatelssvm') || strcmp(costfun,'leaveoneoutlssvm')
    [Y,omega] = helpkernel(model.xtrain,model.ytrain,model.kernel_type,[],0);
    costargs = {Y,omega,costargs{1}};
end

if strcmp(costfun,'rcrossvalidatelssvm')
    eval('model.weights = varargin{4};','model.weights = ''wmyriad''; ')
end

if strcmp(costfun,'crossvalidatelssvm_SIM')
    [Y,omega] = helpkernel(model.xtrain,model.ytrain,model.kernel_type,[],1);
    costargs = {model.xtrain,Y,costargs{1},omega,costargs{2}};
end

% change the coding type for multiclass and set default 'OneVsOne' if no
% coding type specified
%if length(varargin)>=5 && ~isempty(varargin{5})
if model.type(1) =='c' && ~(sum(unique(model.ytrain))==1 || sum(unique(model.ytrain))==0)
    eval('coding = varargin{4};','coding = ''code_OneVsOne''; ')
    varargin{5}= coding;
    model = changelssvm(model,'codetype',coding);
    [yc,cb,oldcb] = code(model.ytrain,coding);
    y_dimold = model.y_dim;
    model.ytrain = yc; model.y_dim = size(yc,2);
    varargin{end} = []; clear yc
end

%
% multiple outputs
if (model.y_dim>1)% & (size(model.kernel_pars,1)==model.y_dim |size(model.gam,1)==model.y_dim |prod(size(model.kernel_type,1))==model.y_dim))
    disp('-->> tune individual outputs');
    if model.type(1) == 'c'
        fprintf('\n')
        disp(['-->> Encoding scheme: ',coding]);
    end
    costs = zeros(model.y_dim,1); gamt = zeros(1,model.y_dim);
    for d=1:model.y_dim,
        sel = ~isnan(model.ytrain(:,d));
        fprintf(['\n\n -> dim ' num2str(d) '/' num2str(model.y_dim) ':\n']);
        try kernel = model.kernel_type{d}; catch, kernel=model.kernel_type;end
        [g,s,c] = tunelssvm({model.xtrain(sel,:),model.ytrain(sel,d),model.type,[],[],kernel,model.global_opt,'original'},varargin{:});
        gamt(:,d) = g;
        try kernel_part(:,d) = s; catch, kernel_part = [];end
        costs(d) = c;
    end
    model.gam = gamt;
    model.kernel_pars = kernel_part;
    if func,
        O3 = costs;
        cost = model.kernel_pars;
        model = model.gam;
    end
    % decode to the original model.yfull
    if model.code(1) == 'c', % changed
        model.ytrain = code(model.ytrain, oldcb, [], cb, 'codedist_hamming');
        model.y_dim = y_dimold;
    end
    return
end

% change type of coding for LSSND model
if model.type(1) =='s',
    coding = 'code_OneVsAll';
    model = changelssvm(model,'codetype',coding);
    [yc,cb,ocb] = code(model.ytrain,coding);
    model.ytrain = yc; model.y_dim = size(yc,2);
    model.codebook1 = ocb;
    model.codebook2 = cb;
    model.code = 'changed';
    clear yc
end

if strcmp(model.global_opt, 'ds')
    method = 'Directional Search (DFO)   ';
else
    method = 'Coupled Simulated Annealing';
end

%-------------------------------------------------------------------------%
if strcmp(model.kernel_type,'lin_kernel'),
    if model.type(1) =='s',
        [par,fval] = csa(rand(2,5),@(x)simanncostfun1(x,model,costfun,costargs));
        model.nu = exp(par(2)); 
    elseif strcmp(model.global_opt,'ds'),
        [par,fval] = dsp(rand(1,5),@(x)simanncostfun1(x,model,costfun,costargs));
    elseif ~strcmp(model.weights,'wmyriad') && ~strcmp(model.weights,'whuber')
        [par,fval] = csa(rand(1,5),@(x)simanncostfun1(x,model,costfun,costargs));
    else
        [par,fval] = csa(rand(2,5),@(x)simanncostfun1(x,model,costfun,costargs));
        model.delta = exp(par(2));
    end
    model = changelssvm(changelssvm(model,'gam',exp(par(1))),'kernel_pars',[]); clear par
    fprintf('\n')
    
    disp([' 1. ' method ' results:  [gam]         ' num2str(model.gam)]);
    if model.type(1) =='s', 
        disp(['                                          [nu]          ' num2str(model.nu)]); 
    end
    disp(['                                          F(X)=         ' num2str(fval)]);
    disp(' ')
    
elseif strcmp(model.kernel_type,'RBF_kernel') || strcmp(model.kernel_type,'sinc_kernel') || strcmp(model.kernel_type,'RBF4_kernel')
    if model.type(1) =='s',
        [par,fval] = csa(rand(3,5),@(x)simanncostfun2(x,model,costfun,costargs));
        model.nu = exp(par(3));
    elseif strcmp(model.global_opt,'ds'),
        [par,fval] = dsp(rand(2,5),@(x)simanncostfun2(x,model,costfun,costargs));
    elseif ~strcmp(model.weights,'wmyriad') && ~strcmp(model.weights,'whuber')
        [par,fval] = csa(rand(2,5),@(x)simanncostfun2(x,model,costfun,costargs));
    else
        [par,fval] = csa(rand(3,5),@(x)simanncostfun2(x,model,costfun,costargs));
        model.delta = exp(par(3));
    end
    model = changelssvm(changelssvm(model,'gam',exp(par(1))),'kernel_pars',exp(par(2)));
    
    fprintf('\n')
    disp([' 1. ' method ' results:  [gam]         ' num2str(model.gam)]);
    disp(['                                          [sig2]        ' num2str(model.kernel_pars)]);
    if model.type(1) =='s', 
        disp(['                                          [nu]          ' num2str(model.nu)]); 
    end
    disp(['                                          F(X)=         ' num2str(fval)]);
    disp(' ')
    
elseif strcmp(model.kernel_type,'poly_kernel'),
    warning off
    if model.type(1) =='s',
        [par,fval] = simann(@(x)simanncostfun3(x,model,costfun,costargs),[0.5;0.5;1],[-5;0.1;1],[10;3;1.9459],1,0.9,5,20,2);
        model.nu = exp(par(4)); 
    elseif ~strcmp(model.weights,'wmyriad') && ~strcmp(model.weights,'whuber')
        [par,fval] = simann(@(x)simanncostfun3(x,model,costfun,costargs),[0.5;0.5;1],[-5;0.1;1],[10;3;1.9459],1,0.9,5,20,2);
    else
        [par,fval] = simann(@(x)simanncostfun3(x,model,costfun,costargs),[0.5;0.5;1;0.2],[-5;0.1;1;eps],[10;3;1.9459;1.5],1,0.9,5,20,2);
        model.delta = exp(par(4));
    end
    warning on
    %[par,fval] = csa(rand(3,5),@(x)simanncostfun3(x,model,costfun,costargs));
    model = changelssvm(changelssvm(model,'gam',exp(par(1))),'kernel_pars',[exp(par(2));round(exp(par(3)))]);
    
    fprintf('\n\n')
    disp([' 1. Simulated Annealing results:          [gam]         ' num2str(model.gam)]);
    disp(['                                          [t]           ' num2str(model.kernel_pars(1))]);
    disp(['                                          [degree]      ' num2str(round(model.kernel_pars(2)))]);
    if model.type(1) =='s', 
        disp(['                                          [nu]          ' num2str(model.nu)]); 
    end
    disp(['                                          F(X)=         ' num2str(fval)]);
    disp(' ')
elseif strcmp(model.kernel_type,'wav_kernel'),
    if ~strcmp(model.weights,'wmyriad') && ~strcmp(model.weights,'whuber')
        [par,fval] = csa(rand(4,5),@(x)simanncostfun4(x,model,costfun,costargs));
    else
        [par,fval] = csa(rand(5,5),@(x)simanncostfun4(x,model,costfun,costargs));
        model.delta = exp(par(5));
    end
    model = changelssvm(changelssvm(model,'gam',exp(par(1))),'kernel_pars',[exp(par(2));exp(par(3));exp(par(4))]);
    fprintf('\n')
    disp([' 1. Coupled Simulated Annealing results:  [gam]         ' num2str(model.gam)]);
    disp(['                                          [sig2]        ' num2str(model.kernel_pars')]);
    disp(['                                          F(X)=         ' num2str(fval)]);
    disp(' ')
    model.implementation = 'matlab';
end

%-------------------------------------------------------------------------%

if length(model.gam)>1,
    error('Only one gamma per output allowed');
end

if model.type(1) == 's' && ~strcmp(optfun,'simplex'),
    error('LSSND model allows only simplex optimization');
end


if fval ~= 0
    %
    % lineare kernel
    %
    if strcmp(model.kernel_type,'lin_kernel'),
        
        if ~strcmp(optfun,'simplex'),optfun = 'linesearch';end
        disp(' TUNELSSVM: chosen specifications:');
        disp([' 2. optimization routine:           ' optfun]);
        disp(['    cost function:                  ' costfun]);
        disp(['    kernel function                 ' model.kernel_type]);
        if strcmp(costfun,'rcrossvalidatelssvm')
            if strcmp(model.weights,'wmyriad') && strcmp(model.weights,'whuber')
                fprintf('\n    weight function:                %s, delta = %2.4f',model.weights,model.delta)
            else
                fprintf('\n    weight function:                %s', model.weights)
            end
        end
        disp(' ');
        eval('startvalues = log(startvalues);','startvalues = [];');
        % construct grid based on CSA start values
        startvalues = log(model.gam)+[-5;10];
        
        if ~strcmp(optfun,'simplex')
            et = cputime;
            c = costofmodel1(startvalues(1),model,costfun,costargs);
            et = cputime-et;
            fprintf('\n')
            disp([' 3. starting values:                   ' num2str(exp(startvalues(1,:)))]);
            disp(['    cost of starting values:           ' num2str(c)]);
            disp(['    time needed for 1 evaluation (sec):' num2str(et)]);
            disp(['    limits of the grid:   [gam]         ' num2str(exp(startvalues(:,1))')]);
            disp(' ');
            disp('OPTIMIZATION IN LOG SCALE...');
            optfun = 'linesearch';
            [gs, cost] = feval(optfun, @costofmodel1,startvalues,{model, costfun,costargs});
        else
            c = fval;
            fprintf('\n')
            disp([' 3. starting value:                   ' num2str(model.gam)]);
            if model.type(1) =='s',
                [gs,cost] = simplex(@(x)simplexcostfun1(x,model,costfun,costargs),[log(model.gam),log(model.nu)],model.kernel_type);
                fprintf('Simplex results: \n')
                fprintf('X=%f,   nu=%.4f, F(X)=%e \n\n',exp(gs(1)),exp(gs(2)),cost)
            elseif ~strcmp(model.weights,'wmyriad') && ~strcmp(model.weights,'whuber')
                [gs,cost] = simplex(@(x)simplexcostfun1(x,model,costfun,costargs),log(model.gam),model.kernel_type);
                fprintf('Simplex results: \n')
                fprintf('X=%f  , F(X)=%e \n\n',exp(gs(1)),cost)
            else
                [gs,cost] = rsimplex(@(x)simplexcostfun1(x,model,costfun,costargs),[log(model.gam) log(model.delta)],model.kernel_type);
                fprintf('Simplex results: \n')
                fprintf('X=%f,  delta=%.4f, F(X)=%e \n\n',exp(gs(1)),exp(gs(2)),cost)
            end
        end
        gamma = exp(gs(1));
        eval('model.delta = exp(gs(2));','')
        eval('model.nu = exp(gs(2));','')
        
        %
        % RBF kernel
        %
    elseif strcmp(model.kernel_type,'RBF_kernel') || strcmp(model.kernel_type,'sinc_kernel') || strcmp(model.kernel_type,'RBF4_kernel'),        
        
        disp(' TUNELSSVM: chosen specifications:');
        disp([' 2. optimization routine:           ' optfun]);
        disp(['    cost function:                  ' costfun]);
        disp(['    kernel function                 ' model.kernel_type]);
        if strcmp(costfun,'rcrossvalidatelssvm')
            if strcmp(model.weights,'wmyriad') && strcmp(model.weights,'whuber')
                fprintf('\n    weight function:                %s, delta = %2.4f',model.weights,model.delta)
            else
                fprintf('\n    weight function:                %s', model.weights)
            end
        end
        disp(' ');
        eval('startvalues = log(startvalues);','startvalues = [];');
        % construct grid based on CSA start values
        startvalues = [log(model.gam)+[-3;5] log(model.kernel_pars)+[-2.5;2.5]];
        
        if ~strcmp(optfun,'simplex')
            %tic;
            et = cputime;
            c = costofmodel2(startvalues(1,:),model,costfun,costargs);
            %et = toc;
            et = cputime-et;
            fprintf('\n')
            disp([' 3. starting values:                   ' num2str(exp(startvalues(1,:)))]);
            disp(['    cost of starting values:           ' num2str(c)]);
            disp(['    time needed for 1 evaluation (sec):' num2str(et)]);
            disp(['    limits of the grid:   [gam]         ' num2str(exp(startvalues(:,1))')]);
            disp(['                          [sig2]        ' num2str(exp(startvalues(:,2))')]);
            disp(' ');
            disp('OPTIMIZATION IN LOG SCALE...');
            [gs, cost] = feval(optfun,@costofmodel2,startvalues,{model, costfun,costargs});
        else
            c = fval;
            fprintf('\n')
            disp([' 3. starting values:                   ' num2str([model.gam model.kernel_pars])]);
            if model.type(1) =='s',
                [gs,cost] = simplex(@(x)simplexcostfun2(x,model,costfun,costargs),[log(model.gam) log(model.kernel_pars) log(model.nu)],model.kernel_type);
                fprintf('Simplex results: \n')
                fprintf('X=%f   %f, nu=%.4f, F(X)=%e \n\n',exp(gs(1)),exp(gs(2)),exp(gs(3)),cost)
            elseif ~strcmp(model.weights,'wmyriad') && ~strcmp(model.weights,'whuber')
                [gs,cost] = simplex(@(x)simplexcostfun2(x,model,costfun,costargs),[log(model.gam) log(model.kernel_pars)],model.kernel_type);
                fprintf('Simplex results: \n')
                fprintf('X=%f   %f, F(X)=%e \n\n',exp(gs(1)),exp(gs(2)),cost)
            else
                [gs,cost] = rsimplex(@(x)simplexcostfun2(x,model,costfun,costargs),[log(model.gam) log(model.kernel_pars) log(model.delta)],model.kernel_type);
                fprintf('Simplex results: \n')
                fprintf('X=%f   %f, delta=%.4f, F(X)=%e \n\n',exp(gs(1)),exp(gs(2)),exp(gs(3)),cost);
            end
        end
        
        gamma = exp(gs(1));
        kernel_pars = exp(gs(2));
        eval('delta=exp(gs(3));','')
        eval('nu=exp(gs(3));','')
        
        
        %
        % polynoom kernel
        %
    elseif strcmp(model.kernel_type,'poly_kernel'),
        
        dg = model.kernel_pars(2);
        disp(' TUNELSSVM: chosen specifications:');
        disp([' 2. optimization routine:           ' optfun]);
        disp(['    cost function:                  ' costfun]);
        disp(['    kernel function                 ' model.kernel_type]);
        if strcmp(costfun,'rcrossvalidatelssvm')
            if strcmp(model.weights,'wmyriad') && strcmp(model.weights,'whuber')
                fprintf('\n    weight function:                %s, delta = %2.4f',model.weights,model.delta)
            else
                fprintf('\n    weight function:                %s', model.weights)
            end
        end
        disp(' ');
        eval('startvalues = log(startvalues);','startvalues = [];');
        % construct grid based on CSA start values
        startvalues = [log(model.gam)+[-3;5] log(model.kernel_pars(1))+[-2.5;2.5]];
        
        if ~strcmp(optfun,'simplex')
            et = cputime;
            warning off
            c = costofmodel3(startvalues(1,:),dg,model,costfun,costargs);
            warning on
            et = cputime-et;
            fprintf('\n')
            disp([' 3. starting values:                   ' num2str([exp(startvalues(1,:)) dg])]);
            disp(['    cost of starting values:           ' num2str(c)]);
            disp(['    time needed for 1 evaluation (sec):' num2str(et)]);
            disp(['    limits of the grid:   [gam]         ' num2str(exp(startvalues(:,1))')]);
            disp(['                          [t]           ' num2str(exp(startvalues(:,2))')]);
            disp(['                          [degree]      ' num2str(dg)]);
            disp('OPTIMIZATION IN LOG SCALE...');
            warning off
            [gs, cost] = feval(optfun,@costofmodel3,startvalues,{dg,model, costfun,costargs});
            warning on
            gamma = exp(gs(1));
            kernel_pars = [exp(gs(2:end));dg];
        else
            c = fval;
            fprintf('\n')
            disp([' 3. starting values:                   ' num2str([model.gam model.kernel_pars'])]);
            warning off
            if model.type(1) =='s',
                [gs,cost] = simplex(@(x)simplexcostfun3(x,model,costfun,costargs),[log(model.gam) log(model.kernel_pars(1)) log(model.nu)],model.kernel_type);
                fprintf('Simplex results: \n')
                fprintf('X=%f   %f    %d, nu=%.4f, F(X)=%e \n\n',exp(gs(1)),exp(gs(2)),model.kernel_pars(2),exp(gs(3)),cost)
            elseif ~strcmp(model.weights,'wmyriad') && ~strcmp(model.weights,'whuber')
                [gs,cost] = simplex(@(x)simplexcostfun3(x,model,costfun,costargs),[log(model.gam) log(model.kernel_pars(1))],model.kernel_type);
                fprintf('Simplex results: \n')
                fprintf('X=%f   %f    %d, F(X)=%e \n\n',exp(gs(1)),exp(gs(2)),model.kernel_pars(2),cost)
            else
                [gs,cost] = rsimplex(@(x)simplexcostfun3(x,model,costfun,costargs),[log(model.gam) log(model.kernel_pars(1)) log(model.delta)],model.kernel_type);
                fprintf('Simplex results: \n')
                fprintf('X=%f   %f    %d, delta=%.4f, F(X)=%e \n\n',exp(gs(1)),exp(gs(2)),model.kernel_pars(2),exp(gs(3)),cost)
            end
            warning on
            
            gamma = exp(gs(1));
            kernel_pars = [exp(gs(2)) model.kernel_pars(2)];
            eval('delta=exp(gs(3));','')
            eval('nu=exp(gs(3));','')
        end
    else
        warning('MATLAB:ambiguousSyntax','Tuning for other kernels is not actively supported,  see ''gridsearch'' and ''linesearch''.')
    end
    
    if cost <= fval % gridsearch/simplex found lower value
        model.gam = gamma; eval('model.kernel_pars = kernel_pars;','model.kernel_pars = [];')
        eval('model.delta = delta;','')
        eval('model.nu = nu;','')
        model.costCV = cost; 
    else
        fprintf('simplex/gridsearch result is not better than CSA... returning to the previous values\n')
    end
    
else %fval = 0 --> CSA already found lowest possible value
    %disp(['Obtained hyper-parameters: [gamma sig2]: ' num2str([model.gam model.kernel_pars])]);
    cost = fval;
    model.costCV = cost; 
end

% display final information
if strcmp(model.kernel_type,'lin_kernel')
    disp(['Obtained hyper-parameters: [gamma]: ' num2str(model.gam)]);
elseif strcmp(model.kernel_type,'RBF_kernel')
    disp(['Obtained hyper-parameters: [gamma sig2]: ' num2str([model.gam model.kernel_pars])]);
elseif strcmp(model.kernel_type,'poly_kernel')
    disp(['Obtained hyper-parameters: [gamma t degree]: ' num2str([model.gam model.kernel_pars])]);
elseif strcmp(model.kernel_type,'wav_kernel')
    disp(['Obtained hyper-parameters: [gamma sig2]: ' num2str([model.gam model.kernel_pars'])]);
end

if func,
    O3 = cost;
    eval('cost = [model.kernel_pars;degree];','cost = model.kernel_pars;');
    model = model.gam;
elseif nargout == 3
    O3 = cost; eval('cost = [model.kernel_pars;degree];','cost = model.kernel_pars;');
    model = model.gam;
elseif nargout == 2
    eval('cost = [model.kernel_pars;degree];','cost = model.kernel_pars;');
    model = model.gam;
else
    model = changelssvm(changelssvm(model,'gam',model.gam),'kernel_pars',model.kernel_pars);
end

function [Y,omega] = helpkernel(X,Y,kernel,L,flag)
n = size(X,1);
if flag==0 % otherwise no permutation for correlated errors
    if L==n, p = 1:n; else p = randperm(n); end
    X = X(p,:);
    Y = Y(p,:);
    clear i p
end
% calculate help kernel matrix of the support vectors en training data
if strcmp(kernel,'RBF_kernel') || strcmp(kernel,'RBF4_kernel')
    omega = sum(X.^2,2)*ones(1,n);
    omega = omega+omega'-2*(X*X');
elseif strcmp(kernel,'sinc_kernel')
    omega = sum(X,2)*ones(1,n);
    omega = omega-omega';
elseif strcmp(kernel,'lin_kernel') || strcmp(kernel,'poly_kernel')
    omega = X*X';
elseif strcmp(kernel,'wav_kernel')
    omega = cell(1,2);
    omega{1} = sum(X.^2,2)*ones(1,n);
    omega{1} = omega{1}+omega{1}'-2*(X*X');
    
    omega{2} = (sum(X,2)*ones(1,n))-(sum(X,2)*ones(1,n))';
    
else
    error('kernel not supported')
end

function cost =  costofmodel1(gs, model,costfun,costargs)
gam = exp(min(max(gs(1),-50),50));
model = changelssvm(model,'gamcsa',gam);
cost = feval(costfun,model,costargs{:});

function cost = simanncostfun1(x0,model,costfun,costargs)
model = changelssvm(changelssvm(model,'gamcsa',exp(x0(1,:))),'kernel_parscsa',[]);
eval('model.deltacsa = exp(x0(2,:));','')
eval('model.nucsa = exp(x0(2,:));','')
cost = feval(costfun,model,costargs{:});

function cost = simplexcostfun1(x0,model,costfun,costargs)
model = changelssvm(changelssvm(model,'gamcsa',exp(x0(1))),'kernel_parscsa',[]);
eval('model.deltacsa = exp(x0(2,:));','')
eval('model.nucsa = exp(x0(2,:));','')
cost = feval(costfun,model,costargs{:});

function cost =  costofmodel2(gs, model,costfun,costargs)
gam = exp(min(max(gs(1),-50),50));
sig2 = zeros(length(gs)-1,1);
for i=1:length(gs)-1, sig2(i,1) = exp(min(max(gs(1+i),-50),50)); end
model = changelssvm(changelssvm(model,'gamcsa',gam),'kernel_parscsa',sig2);
cost = feval(costfun,model,costargs{:});

function cost = simanncostfun2(x0,model,costfun,costargs)
model = changelssvm(changelssvm(model,'gamcsa',exp(x0(1,:))),'kernel_parscsa',exp(x0(2,:)));
eval('model.deltacsa = exp(x0(3,:));','')
eval('model.nucsa = exp(x0(3,:));','')
cost = feval(costfun,model,costargs{:});

function cost = simplexcostfun2(x0,model,costfun,costargs)
model = changelssvm(changelssvm(model,'gamcsa',exp(x0(1))),'kernel_parscsa',exp(x0(2)));
eval('model.deltacsa = exp(x0(3));','')
eval('model.nucsa = exp(x0(3));','')
cost = feval(costfun,model,costargs{:});

function cost =  costofmodel3(gs,d, model,costfun,costargs)
gam = exp(min(max(gs(1),-50),50));
sig2 = exp(min(max(gs(2),-50),50));
model = changelssvm(changelssvm(model,'gamcsa',gam),'kernel_parscsa',[sig2;d]);
cost = feval(costfun,model,costargs{:});

function cost = simanncostfun3(x0,model,costfun,costargs)
model = changelssvm(changelssvm(model,'gamcsa',exp(x0(1,:))),'kernel_parscsa',[exp(x0(2,:));round(exp(x0(3,:)))]);
eval('model.deltacsa = exp(x0(4,:));','')
eval('model.nucsa = exp(x0(4,:));','')
cost = feval(costfun,model,costargs{:});

function cost = simplexcostfun3(x0,model,costfun,costargs)
model = changelssvm(changelssvm(model,'gamcsa',exp(x0(1))),'kernel_parscsa',[exp(x0(2));model.kernel_pars(2)]);
eval('model.deltacsa = exp(x0(3));','')
eval('model.nucsa = exp(x0(3));','')
cost = feval(costfun,model,costargs{:});
