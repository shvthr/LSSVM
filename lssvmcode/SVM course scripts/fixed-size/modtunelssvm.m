function [gam,sig] = modtunelssvm(nmodel,svX,L,estfunc,optfun,loss_type)

addpath('../LSSVMlab');

%Obtain the model parameters
model.xtrain = nmodel{1,1};
model.ytrain = nmodel{1,2};
model.type = nmodel{1,3};
model.gam = nmodel{1,4};
model.kernel_pars = nmodel{1,5};
model.kernel_type = nmodel{1,6};
model.global_opt = nmodel{1,7};

[omega,omegaN] = helpkernel(model.xtrain,model.kernel_type,svX);
if (nargin>5)
    model.weights = loss_type;
    costfun = 'modcrossvalidatelssvm';
    costargs = {L,estfunc,'fs',svX,omega,omegaN};
else
    model.weights = ''; %Start with an empty string 
    costfun = 'modcrossvalidatelssvm';
    costargs = {L,estfunc,'fs',svX,omega,omegaN};
end;

if strcmp(model.global_opt, 'ds')
    method = 'Directional Search (DFO)   ';
else
    method = 'Coupled Simulated Annealing';
end
%Perform the csa to get an initial starting point
if strcmp(model.kernel_type,'lin_kernel'),
    if strcmp(model.global_opt,'ds'),
        [par,fval] = dsp(rand(1,5),@(x)simanncostfun1(x,model,'modcrossvalidatelssvm',costargs));
    elseif ~strcmp(model.weights,'wmyriad') && ~strcmp(model.weights,'whuber')
        [par,fval] = csa(rand(1,5),@(x)simanncostfun1(x,model,'modcrossvalidatelssvm',costargs));
    else
        [par,fval] = csa(rand(2,5),@(x)simanncostfun1(x,model,'modcrossvalidatelssvm',costargs));
        model.delta = exp(par(2));
    end
    model.gam = exp(par(1));
    model.kernel_pars = [];
    fprintf('\n')
    disp([' 1. ' method ' results:  [gam]         ' num2str(model.gam)]);
    disp(['                                          F(X)=         ' num2str(fval)]);
    disp(' ');
 
elseif strcmp(model.kernel_type,'RBF_kernel')
    if strcmp(model.global_opt,'ds'),
        [par,fval] = dsp(rand(2,5),@(x)simanncostfun2(x,model,'modcrossvalidatelssvm',costargs));
    elseif ~strcmp(model.weights,'wmyriad') && ~strcmp(model.weights,'whuber')
        [par,fval] = csa(rand(2,5),@(x)simanncostfun2(x,model,'modcrossvalidatelssvm',costargs));
    else
        [par,fval] = csa(rand(3,5),@(x)simanncostfun2(x,model,'modcrossvalidatelssvm',costargs));
        model.delta = exp(par(3));
    end;
    model.gam = exp(par(1));
    model.kernel_pars = exp(par(2));
    
    fprintf('\n')   
    disp([' 1. ' method ' results:  [gam]         ' num2str(model.gam)]);
    disp(['                                          [sig2]        ' num2str(model.kernel_pars)]);
    disp(['                                          F(X)=         ' num2str(fval)]);
    disp(' ');

elseif strcmp(model.kernel_type,'poly_kernel'),
    warning off
    if ~strcmp(model.weights,'wmyriad') && ~strcmp(model.weights,'whuber')
        [par,fval] = simann(@(x)simanncostfun3(x,model,'modcrossvalidatelssvm',costargs),[0.5;0.5;1],[-5;0.1;1],[10;3;1.9459],1,0.9,5,20,2);
    else
        [par,fval] = simann(@(x)simanncostfun3(x,model,'modcrossvalidatelssvm',costargs),[0.5;0.5;1;0.2],[-5;0.1;1;eps],[10;3;1.9459;1.5],1,0.9,5,20,2);
        model.delta = exp(par(4));
    end
    warning on
    model.gam  = exp(par(1));
    model.kernel_pars = [exp(par(2));round(exp(par(3)))];
    
    fprintf('\n\n')
    disp([' 1. Simulated Annealing results:          [gam]         ' num2str(model.gam)]);
    disp(['                                          [t]           ' num2str(model.kernel_pars(1))]);
    disp(['                                          [degree]      ' num2str(round(model.kernel_pars(2)))]);
    disp(['                                          F(X)=         ' num2str(fval)]);
    disp(' ')
end;

if (strcmp(optfun,''))
    gam=model.gam;
    sig=model.kerenl_pars;
    return;
end;
%----------------------------------------------------------------------------
%Simplex Starts from here

if (strcmp(model.kernel_type,'lin_kernel'))
    if ~strcmp(optfun,'simplex'),optfun = 'linesearch';end
    disp(' TUNELSSVM: chosen specifications:');
    disp([' 2. optimization routine:           ' optfun]);
    disp(['    cost function:                  ' costfun]);
    disp(['    kernel function                 ' model.kernel_type]);
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
        if ~strcmp(model.weights,'wmyriad') && ~strcmp(model.weights,'whuber')
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
    try delta = exp(gs(2)); catch ME,'';ME.stack;end
        
elseif (strcmp(model.kernel_type,'RBF_kernel'))
    if ~strcmp(optfun,'simplex'),optfun = 'linesearch';end
    disp(' TUNELSSVM: chosen specifications:');
    disp([' 2. optimization routine:           ' optfun]);
    disp(['    cost function:                  ' costfun]);
    disp(['    kernel function                 ' model.kernel_type]);

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
    
        fprintf('\n')
        disp([' 3. starting values:                   ' num2str([model.gam model.kernel_pars])]);
        if ~strcmp(model.weights,'wmyriad') && ~strcmp(model.weights,'whuber')
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

elseif strcmp(model.kernel_type,'poly_kernel') 
    if ~strcmp(optfun,'simplex'),optfun = 'linesearch';end
    dg = model.kernel_pars(2);
    disp(' TUNELSSVM: chosen specifications:');
    disp([' 2. optimization routine:           ' optfun]);
    disp(['    cost function:                  ' costfun]);
    disp(['    kernel function                 ' model.kernel_type]);
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
        model.gam = exp(gs(1));
        model.kernel_pars = [exp(gs(2:end));dg];
    else
        c = fval;
        fprintf('\n')
        disp([' 3. starting values:                   ' num2str([model.gam model.kernel_pars'])]);
        warning off
        if ~strcmp(model.weights,'wmyriad') && ~strcmp(model.weights,'whuber')
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
        try delta = exp(gs(3)); catch ME,'';ME.stack;end
    end
else
    warning('MATLAB:ambiguousSyntax','Tuning for other kernels is not actively supported,  see ''gridsearch'' and ''linesearch''.')
end

if cost <= fval % gridsearch found lower value
    model.gam = gamma; 
    eval('model.kernel_pars = kernel_pars;','model.kernel_pars = [];');
    eval('model.delta = delta;','');
    model.costCV = cost;
end;
gam = model.gam;
sig = model.kernel_pars;
if strcmp(model.kernel_type,'lin_kernel')
    disp(['Obtained hyper-parameters: [gamma]: ' num2str(model.gam)]);
elseif strcmp(model.kernel_type,'RBF_kernel')
    disp(['Obtained hyper-parameters: [gamma sig2]: ' num2str([model.gam model.kernel_pars])]);
elseif strcmp(model.kernel_type,'poly_kernel')
    disp(['Obtained hyper-parameters: [gamma t degree]: ' num2str([model.gam model.kernel_pars])]);    
end;


function [omega,omegaN] = helpkernel(X,kernel,svX)
if (strcmp(kernel,'RBF_kernel'))
    XXh = sum(svX.^2,2)*ones(1,size(svX,1));
    omega = XXh+XXh'-2*(svX*svX'); 
    XXh1 = sum(X.^2,2)*ones(1,size(svX,1));
    XXh2 = sum(svX.^2,2)*ones(1,size(X,1));
    omegaN = XXh1+XXh2' - 2*X*svX';
elseif strcmp(kernel,'lin_kernel') || strcmp(kernel,'poly_kernel')
    omega = svX*svX';
    omegaN = X*svX';
end;

function cost =  costofmodel1(gs,model,costfun,costargs)
gam = exp(min(max(gs(1),-50),50));
model.gam = gam;
cost = feval(costfun,model,costargs{:});

function cost = simanncostfun1(x0,model,costfun,costargs)
model.gam = exp(x0(1,:));
model.kernel_pars = [];
eval('model.delta = exp(x0(2,:));','');
cost = feval(costfun,model,costargs{:});

function cost = simplexcostfun1(x0,model,costfun,costargs)
model.gam = exp(x0(1));
model.kernel_pars = [];
try model.delta = exp(x0(2)); catch ME, '';ME.stack;end
cost = feval(costfun,model,costargs{:});

function cost = simanncostfun2(x0,model,costfun,costargs)
model.gam = exp(x0(1,:));
model.kernel_pars = exp(x0(2,:));
eval('model.delta = exp(x0(3,:));','')
cost = feval(costfun,model,costargs{:});

function cost = simplexcostfun2(x0,model,costfun,costargs)
model.gam = exp(x0(1));
model.kernel_pars = exp(x0(2));
eval('model.delta = exp(x0(3));','')
cost = feval(costfun,model,costargs{:});

function cost =  costofmodel2(gs, model,costfun,costargs)
gam = exp(min(max(gs(1),-50),50));
sig2 = zeros(length(gs)-1,1);
for i=1:length(gs)-1, sig2(i,1) = exp(min(max(gs(1+i),-50),50)); end
model.gam = gam;
model.kernel_pars = sig2;
cost = feval(costfun,model,costargs{:});

function cost =  costofmodel3(gs,d, model,costfun,costargs)
gam = exp(min(max(gs(1),-50),50));
sig2 = exp(min(max(gs(2),-50),50));
%sig2 = zeros(length(gs)-1,2);
model.gam = gam;
%for i=1:length(gs)-1, sig2(i,1) = exp(min(max(gs(i+1),-50),50)) ; sig2(i,2) = d; end
model.kernel_pars = [sig2;d];
cost = feval(costfun,model,costargs{:});

function cost = simanncostfun3(x0,model,costfun,costargs)
model.gam = exp(x0(1,:));
model.kernel_pars = [exp(x0(2,:));round(exp(x0(3,:)))];
eval('model.delta = exp(x0(4,:));','')
cost = feval(costfun,model,costargs{:});

function cost = simplexcostfun3(x0,model,costfun,costargs)
model.gam = exp(x0(1));
model.kernel_pars = [exp(x0(2));model.kernel_pars(2)];
eval('model.delta = exp(x0(3));','')
cost = feval(costfun,model,costargs{:});