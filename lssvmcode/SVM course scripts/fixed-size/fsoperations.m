function [errmatrix,svmatrix,timematrix] = fsoperations(X,Y,function_type,kernel_type,global_opt,user_process,windowrange,renyie,subset,svX,svY,testX,testY)

%Input:
%       X = training dataset
%       Y = test dataset
%       k = constant factor used to determine number of representative
%       points by heursitic k*sqrt(N) where N = dataset size
%       function_type = 'c' or 'f' for classification & regression
%       respectively
%       kernel_type = currently supports 'RBF_kernel' [Extension to
%       'lin_kernel' to be done]
%       user_process = process user wishes to perfrom options
%       
%       ['L0_norm','SV_L0_norm','ALL_l0_norm','WINDOW' & by default '']
%       L0_norm -> Performs an L0_norm over the set of support vectors
%       while training on the entire dataset (does not scale up)
%       SV_L0_norm -> Performs an L0_norm over the Nystrom approximated
%       support vectors and incorporates information about the entire
%       dataset in the kernel matrix while just using the support vectors for training 
%       ALL_L0_norm -> Performs an L0_norm over the Nystrom approximated
%       support vectors and trains over the entire dataset and scales up due to presence of sparsity
%       WINDOW -> Performs a FS-LSSVM on the representative points and then
%       selects points which are correctly classified closest and farthest
%       from decision boundary to capture the treu boundary of the class
%       '' -> Performs the default FS-LSSVM using Nystrom Approximations
%
%      window_size = optional parametes used for process = 'WINDOW' for
%      determining the size of support vectors to select
%      renyie =  time required for renyi entropy calculation
%      subset = subset selected by Renyi entropy
%      svX,svY = initial prototype vectors 
        
%Output:
%       errmatrix = Contains error for 10 randomizations
%       svmatrix = Contains support vectors for 10 randomizations
%       timematrix = Contains the time complexity for 10 randomizations

addpath('../LSSVMlab');

%Default allocations
if (isempty(kernel_type))
    kernel_type = 'RBF_kernel';
end;
if (isempty(windowrange))
    windowrange = [15,20,25];
end;

if (function_type=='c')
    %Initialization of other variable
    N=length(X);
    
    %10 fold cross validation
    folds=10;
    folds1=3;
    block_size = floor(N/folds1);
    process_type=user_process;
    %Perform Classification
    [errmatrix,svmatrix,timematrix]=classification(X,Y,N,renyie,subset,svX,svY,folds,block_size,kernel_type,global_opt,process_type,windowrange,testX,testY);    
        
end; 

% Regression Analysis
if (function_type=='f')
    %Initialization of other variable
    N=length(X);
    
    %Perform 10 fold cross-validation
    folds=10;
    folds1=3;
    block_size = floor(N/folds1);
    process_type=user_process;
    [errmatrix,svmatrix,timematrix]=regression(X,Y,N,renyie,subset,svX,svY,folds,block_size,kernel_type,global_opt,process_type,windowrange,testX,testY);
end;

function [ematrix,smatrix,tmatrix] = classification(X,Y,N,renyie,subset,svX,svY,folds,block_size,kernel_type,global_opt,process_type,windowrange,testX,testY)

if ((~isempty(process_type(strcmp(process_type(:),'WINDOW')))&&isempty(process_type(strcmp(process_type(:),'LSSVMwin')))&&isempty(process_type(strcmp(process_type(:),'LSSVMwinL'))))...
        ||(isempty(process_type(strcmp(process_type(:),'WINDOW')))&&isempty(process_type(strcmp(process_type(:),'LSSVMwinL')))&&~isempty(process_type(strcmp(process_type(:),'LSSVMwin'))))...
        ||(isempty(process_type(strcmp(process_type(:),'WINDOW')))&&isempty(process_type(strcmp(process_type(:),'LSSVMwin')))&&~isempty(process_type(strcmp(process_type(:),'LSSVMwinL')))))
    ematrix=zeros(length(process_type)+length(windowrange)-1,folds);
    smatrix=zeros(length(process_type)+length(windowrange)-1,folds);
    tmatrix=zeros(length(process_type)+length(windowrange)-1,folds);
    avgerr=0.0*ones(length(process_type)+length(windowrange)-1,1);
elseif (~isempty(process_type(strcmp(process_type(:),'WINDOW')))&&~isempty(process_type(strcmp(process_type(:),'LSSVMwin')))&&isempty(process_type(strcmp(process_type(:),'LSSVMwinL'))) ...
        ||(~isempty(process_type(strcmp(process_type(:),'WINDOW')))&&isempty(process_type(strcmp(process_type(:),'LSSVMwin')))&&~isempty(process_type(strcmp(process_type(:),'LSSVMwinL'))))...
        ||(isempty(process_type(strcmp(process_type(:),'WINDOW')))&&~isempty(process_type(strcmp(process_type(:),'LSSVMwinL')))&&~isempty(process_type(strcmp(process_type(:),'LSSVMwin')))))
    ematrix=zeros(length(process_type)+2*length(windowrange)-2,folds);
    smatrix=zeros(length(process_type)+2*length(windowrange)-2,folds);
    tmatrix=zeros(length(process_type)+2*length(windowrange)-2,folds);
    avgerr=0.0*ones(length(process_type)+2*length(windowrange)-2,1);
    %glbavgerr=inf*ones(length(process_type)+2*length(windowsize)-2,1);
elseif (~isempty(process_type(strcmp(process_type(:),'WINDOW')))&&~isempty(process_type(strcmp(process_type(:),'LSSVMwin')))&&~isempty(process_type(strcmp(process_type(:),'LSSVMwinL'))))
    ematrix=zeros(length(process_type)+3*length(windowrange)-3,folds);
    smatrix=zeros(length(process_type)+3*length(windowrange)-3,folds);
    tmatrix=zeros(length(process_type)+3*length(windowrange)-3,folds);
    avgerr=0.0*ones(length(process_type)+2*length(windowrange)-3,1);
elseif (isempty(process_type(strcmp(process_type(:),'WINDOW')))&&isempty(process_type(strcmp(process_type(:),'LSSVMwin')))&&isempty(process_type(strcmp(process_type(:),'LSSVMwinL'))))
    ematrix = zeros(length(process_type),folds);
    smatrix = zeros(length(process_type),folds);
    tmatrix = zeros(length(process_type),folds);
    avgerr = 0.0*ones(length(process_type),1);
    %glbavgerr=inf*ones(length(process_type),1);
end;
folds1=1;
%par
for j=1:10
    avgerr=0.0*avgerr;
    if (isempty(testX) && isempty(testY))
    %Once obtained the best gamma and sigma perform the 10-fold cross
    %validation to verify the results
    %For very large datasets make folds1 = 1
        for l=1:folds1   %3-fold cross-validation
            if (folds1~=1)
                if l==folds1,
                    train = 1:block_size*(l-1); % not used
                    validation = block_size*(l-1)+1:N;
                else
                    train = [1:block_size*(l-1) block_size*l+1:N]; % not used
                    validation = block_size*(l-1)+1:block_size*l;
                end;
            else
                train = 1:N-block_size;
                validation = N-block_size+1:N;
            end;
            %Tuning the parameters of the problem using coupled simulating analysis
            if (((isempty(process_type(strcmp(process_type(:),'L0_norm')))) && (isempty(process_type(strcmp(process_type(:),'Approx_LSSVM')))) && (isempty(process_type(strcmp(process_type(:),'LSSVMwin')))) && (isempty(process_type(strcmp(process_type(:),'FullL0_norm'))))...
                    && (isempty(process_type(strcmp(process_type(:),'LSSVMwinL')))))||(~isempty(process_type(strcmp(process_type(:),'FS-LSSVM')))))
                t=cputime;
                [gam,sig]=tunefslssvm({X(train,:),Y(train,:),'c',[],[],kernel_type,global_opt},svX,folds,'misclass','simplex');
                t1=cputime-t;
            end;
            if (~isempty(process_type(strcmp(process_type(:),'L0_norm')))||~isempty(process_type(strcmp(process_type(:),'LSSVMwin')))...
                    ||~isempty(process_type(strcmp(process_type(:),'FullL0_norm'))) ||~isempty(process_type(strcmp(process_type(:),'LSSVMwinL')))||~isempty(process_type(strcmp(process_type(:),'Approx_LSSVM'))))
                tl=cputime;
                if(~isempty(process_type(strcmp(process_type(:),'L0_norm')))||~isempty(process_type(strcmp(process_type(:),'LSSVMwin')))||~isempty(process_type(strcmp(process_type(:),'LSSVMwinL')))||~isempty(process_type(strcmp(process_type(:),'Approx_LSSVM'))))
                    [gam1,sig1] = tunelssvm({svX,svY,'c',[],[],kernel_type,global_opt},'simplex','crossvalidatelssvm',{folds,'misclass'});
                    tl1=cputime-tl;
                end;
                tll=cputime;
                if (~isempty(process_type(strcmp(process_type(:),'FullL0_norm'))))
                    [gam2,sig2]=modtunelssvm({X(train,:),Y(train,:),'c',[],[],kernel_type,global_opt},svX,folds,'misclass','simplex');
                    tl2=cputime-tll;
                end
            end;
            %%modsparseoperations(X,Y,train,validation,svX,svY,subset,sigma_optimal,gam_optimal,min_pts,
            %%cutoff,function_type,process_type,window_size,plot_handler)
            for k=1:length(process_type)
                %If no 'WINDOW' based approach
                if (isempty(process_type(strcmp(process_type(:),'WINDOW')))&&isempty(process_type(strcmp(process_type(:),'LSSVMwin')))&&isempty(process_type(strcmp(process_type(:),'LSSVMwinL'))))
                    if (~strcmp(process_type(k),'L0_norm') && ~strcmp(process_type(k),'FullL0_norm') && ~strcmp(process_type(k),'Approx_LSSVM'))
                        [tim,err,newsvX,newsvY] = operations(X,Y,train,validation,svX,svY,subset,sig,gam,kernel_type,'c',process_type(k),[]);
                    elseif (strcmp(process_type(k),'L0_norm')||strcmp(process_type(k),'Approx_LSSVM'))
                        [tim,err,newsvX,newsvY] = operations(X,Y,train,validation,svX,svY,subset,sig1,gam1,kernel_type,'c',process_type(k),[]);
                    elseif (strcmp(process_type(k),'FullL0_norm'))
                        [tim,err,newsvX,newsvY] = operations(X,Y,train,validation,svX,svY,subset,sig2,gam2,kernel_type,'c',process_type(k),[]);
                    end;
                    [error,spvc,timreq]=avgoperations(avgerr,k,newsvX,newsvY,err,tim,renyie,folds1);
                    ematrix(k,j) = error;
                    smatrix(k,j) = spvc;
                    if (strcmp(process_type(k),'FullL0_norm'))
                        tmatrix(k,j) = tl2+timreq;
                    elseif (strcmp(process_type(k),'L0_norm')||strcmp(process_type(k),'Approx_LSSVM'))
                        tmatrix(k,j) = tl1+timreq;
                    else
                        tmatrix(k,j) = t1+timreq;
                    end;
                %If we perform the 'WINDOW' based operations
                elseif (~isempty(process_type(strcmp(process_type(:),'WINDOW')))||~isempty(process_type(strcmp(process_type(:),'LSSVMwin')))||~isempty(process_type(strcmp(process_type(:),'LSSVMwinL'))))
                    if (strcmp(process_type(k),'WINDOW'))
                        for n=1:length(windowrange)
                            windowsize=windowrange(n);
                            [tim,err,newsvX,newsvY] = operations(X,Y,train,validation,svX,svY,subset,sig,gam,kernel_type,'c',process_type(k),windowsize);
                            [error,spvc,timreq]=avgwinoperations(avgerr,k,n,newsvX,newsvY,err,tim,renyie,folds1);
                            ematrix(k+n-1,j) = error;
                            smatrix(k+n-1,j) = spvc;
                            tmatrix(k+n-1,j) = t1+timreq;
                        end;
                    elseif (strcmp(process_type(k),'L0_norm'))
                        [tim,err,newsvX,newsvY] = operations(X,Y,train,validation,svX,svY,subset,sig1,gam1,kernel_type,'c',process_type(k),[]);
                        [error,spvc,timreq]=avgoperations(avgerr,k,newsvX,newsvY,err,tim,renyie,folds1); 
                        ematrix(k,j) = error;
                        smatrix(k,j) = spvc;
                        tmatrix(k,j) = tl1+timreq;
                    elseif (strcmp(process_type(k),'FullL0_norm'))
                        [tim,err,newsvX,newsvY] = operations(X,Y,train,validation,svX,svY,subset,sig2,gam2,kernel_type,'c',process_type(k),[]);
                        [error,spvc,timreq]=avgoperations(avgerr,k,newsvX,newsvY,err,tim,renyie,folds1); 
                        ematrix(k,j) = error;
                        smatrix(k,j) = spvc;
                        tmatrix(k,j) = tl2+timreq;   
                    elseif (strcmp(process_type(k),'Approx_LSSVM'))
                        [tim,err,newsvX,newsvY] = operations(X,Y,train,validation,svX,svY,subset,sig1,gam1,kernel_type,'c',process_type(k),[]);
                        [error,spvc,timreq]=avgoperations(avgerr,k,newsvX,newsvY,err,tim,renyie,folds1); 
                        ematrix(k,j) = error;
                        smatrix(k,j) = spvc;
                        tmatrix(k,j) = tl1+timreq; 
                    elseif(strcmp(process_type(k),'LSSVMwin'))
                        for n=1:length(windowrange)
                            windowsize=windowrange(n);
                            [tim,err,newsvX,newsvY] = operations(X,Y,train,validation,svX,svY,subset,sig1,gam1,kernel_type,'c',process_type(k),windowsize);
                            [error,spvc,timreq]=avgwinoperations(avgerr,k,n,newsvX,newsvY,err,tim,renyie,folds1);
                            if (~isempty(process_type(strcmp(process_type(:),'WINDOW'))))
                                ematrix(k+length(windowrange)+n-2,j) = error;
                                smatrix(k+length(windowrange)+n-2,j) = spvc;
                                tmatrix(k+length(windowrange)+n-2,j) = tl1+timreq;
                            elseif(isempty(process_type(strcmp(process_type(:),'WINDOW'))))
                                ematrix(k+n-1,j) = error;
                                smatrix(k+n-1,j) = spvc;
                                tmatrix(k+n-1,j) = tl1+timreq;
                            end
                        end;
                    elseif(strcmp(process_type(k),'LSSVMwinL'))
                        for n=1:length(windowrange)
                            windowsize=windowrange(n);
                            [tim,err,newsvX,newsvY] = operations(X,Y,train,validation,svX,svY,subset,sig1,gam1,kernel_type,'c',process_type(k),windowsize);
                            [error,spvc,timreq]=avgwinoperations(avgerr,k,n,newsvX,newsvY,err,tim,renyie,folds1);
                            if (~isempty(process_type(strcmp(process_type(:),'WINDOW'))) && ~isempty(process_type(strcmp(process_type(:),'LSSVMwin'))))
                                ematrix(k+length(windowrange)+n,j) = error;
                                smatrix(k+length(windowrange)+n,j) = spvc;
                                tmatrix(k+length(windowrange)+n,j) = tl1+timreq;
                            elseif((isempty(process_type(strcmp(process_type(:),'WINDOW')))&& ~isempty(process_type(strcmp(process_type(:),'LSSVMwin'))))...
                                    || (~isempty(process_type(strcmp(process_type(:),'WINDOW')))&& isempty(process_type(strcmp(process_type(:),'LSSVMwin')))))
                                ematrix(k+length(windowrange)+n-2,j) = error;
                                smatrix(k+length(windowrange)+n-2,j) = spvc;
                                tmatrix(k+length(windowrange)+n-2,j) = tl1+timreq;
                            elseif ((isempty(process_type(strcmp(process_type(:),'WINDOW'))))&& (isempty(process_type(strcmp(process_type(:),'LSSVMwin')))))
                                ematrix(k+n-1,j) = error;
                                smatrix(k+n-1,j) = spvc;
                                tmatrix(k+n-1,j) = tl1+timreq;
                            end;
                        end;
                   elseif (~strcmp(process_type(k),'Approx_LSSVM')&&~strcmp(process_type(k),'L0_norm')&&~strcmp(process_type(k),'WINDOW')&&~strcmp(process_type(k),'LSSVMwin')&&~strcmp(process_type(k),'LSSVMwinL')&&~strcmp(process_type(k),'FullL0_norm'))
                        [tim,err,newsvX,newsvY] = operations(X,Y,train,validation,svX,svY,subset,sig,gam,kernel_type,'c',process_type(k),[]);
                        [error,spvc,timreq]=avgoperations(avgerr,k,newsvX,newsvY,err,tim,renyie,folds1); 
                        ematrix(k,j) = error;
                        smatrix(k,j) = spvc;
                        tmatrix(k,j) = t1+timreq;
                    end;
                end;
            end;
            ematrix
        end;
    else
        %Tuning the parameters of the problem using coupled simulating analysis
        if (((isempty(process_type(strcmp(process_type(:),'L0_norm')))) && (isempty(process_type(strcmp(process_type(:),'Approx_LSSVM')))) && (isempty(process_type(strcmp(process_type(:),'LSSVMwin')))) && (isempty(process_type(strcmp(process_type(:),'FullL0_norm'))))...
                && (isempty(process_type(strcmp(process_type(:),'LSSVMwinL')))))||(~isempty(process_type(strcmp(process_type(:),'FS-LSSVM')))))
            t=cputime;
            [gam,sig]=tunefslssvm({X,Y,'c',[],[],kernel_type,global_opt},svX,folds,'misclass','simplex');
            t1=cputime-t;
            modelparam = [gam sig];
            csvwrite('modelparam1.mat',modelparam);
        end;
        if (~isempty(process_type(strcmp(process_type(:),'L0_norm')))||~isempty(process_type(strcmp(process_type(:),'LSSVMwin')))...
                ||~isempty(process_type(strcmp(process_type(:),'FullL0_norm'))) ||~isempty(process_type(strcmp(process_type(:),'LSSVMwinL'))) || ~isempty(process_type(strcmp(process_type(:),'Approx_LSSVM'))))
            tl=cputime;
            if(~isempty(process_type(strcmp(process_type(:),'L0_norm'))) || ~isempty(process_type(strcmp(process_type(:),'Approx_LSSVM'))) ||~isempty(process_type(strcmp(process_type(:),'LSSVMwin')))||~isempty(process_type(strcmp(process_type(:),'LSSVMwinL'))))
                [gam1,sig1] = tunelssvm({svX,svY,'c',[],[],kernel_type,global_opt},'simplex','crossvalidatelssvm',{folds,'misclass'});
                tl1=cputime-tl;
                modelparam = [gam1 sig1];
                csvwrite('modelparam1.mat',modelparam);
            end;
            tll=cputime;
            if (~isempty(process_type(strcmp(process_type(:),'FullL0_norm'))))
                [gam2,sig2]=modtunelssvm({X,Y,'c',[],[],kernel_type,global_opt},svX,folds,'misclass','simplex');
                tl2=cputime-tll;
                modelparam = [gam2 sig2];
                csvwrite('modelparam2.mat',modelparam);
            end
        end;
        %%modsparseoperations(X,Y,train,validation,svX,svY,subset,sigma_optimal,gam_optimal,min_pts,
        %%cutoff,function_type,process_type,window_size,plot_handler)
        for k=1:length(process_type)
            %If no 'WINDOW' based approach
            if (isempty(process_type(strcmp(process_type(:),'WINDOW')))&&isempty(process_type(strcmp(process_type(:),'LSSVMwin')))&&isempty(process_type(strcmp(process_type(:),'LSSVMwinL'))))
                if (~strcmp(process_type(k),'L0_norm') && ~strcmp(process_type(k),'FullL0_norm') && ~strcmp(process_type(k),'Approx_LSSVM'))
                    [tim,err,newsvX,newsvY] = modoperations(X,Y,testX,testY,svX,svY,subset,sig,gam,kernel_type,'c',process_type(k),[]);
                elseif (strcmp(process_type(k),'L0_norm'))
                    [tim,err,newsvX,newsvY] = modoperations(X,Y,testX,testY,svX,svY,subset,sig1,gam1,kernel_type,'c',process_type(k),[]);
                elseif (strcmp(process_type(k),'Approx_LSSVM'))
                    [tim,err,newsvX,newsvY] = modoperations(X,Y,testX,testY,svX,svY,subset,sig1,gam1,kernel_type,'c',process_type(k),[]);
                elseif (strcmp(process_type(k),'FullL0_norm'))
                    [tim,err,newsvX,newsvY] = modoperations(X,Y,testX,testY,svX,svY,subset,sig2,gam2,kernel_type,'c',process_type(k),[]);
                end;
                [error,spvc,timreq]=avgoperations(avgerr,k,newsvX,newsvY,err,tim,renyie,folds1);
                ematrix(k,j) = error;
                smatrix(k,j) = spvc;
                if (strcmp(process_type(k),'FullL0_norm'))
                    tmatrix(k,j) = tl2+timreq;
                elseif (strcmp(process_type(k),'L0_norm'))
                    tmatrix(k,j) = tl1+timreq;
                elseif (strcmp(process_type(k),'Approx_LSSVM'))
                    tmatrix(k,j) = tl1+timreq;
                else
                    tmatrix(k,j) = t1+timreq;
                end;
            %If we perform the 'WINDOW' based operations
            elseif (~isempty(process_type(strcmp(process_type(:),'WINDOW')))||~isempty(process_type(strcmp(process_type(:),'LSSVMwin')))||~isempty(process_type(strcmp(process_type(:),'LSSVMwinL'))))
                if (strcmp(process_type(k),'WINDOW'))
                    for n=1:length(windowrange)
                        windowsize=windowrange(n);
                        [tim,err,newsvX,newsvY] = modoperations(X,Y,testX,testY,svX,svY,subset,sig,gam,kernel_type,'c',process_type(k),windowsize);
                        [error,spvc,timreq] = avgwinoperations(avgerr,k,n,newsvX,newsvY,err,tim,renyie,folds1);
                        ematrix(k+n-1,j) = error;
                        smatrix(k+n-1,j) = spvc;
                        tmatrix(k+n-1,j) = t1+timreq;
                    end;
                elseif (strcmp(process_type(k),'L0_norm'))
                    [tim,err,newsvX,newsvY] = modoperations(X,Y,testX,testY,svX,svY,subset,sig1,gam1,kernel_type,'c',process_type(k),[]);
                    [error,spvc,timreq]=avgoperations(avgerr,k,newsvX,newsvY,err,tim,renyie,folds1); 
                    ematrix(k,j) = error;
                    smatrix(k,j) = spvc;
                    tmatrix(k,j) = tl1+timreq;
                elseif (strcmp(process_type(k),'Approx_LSSVM'))
                    [tim,err,newsvX,newsvY] = modoperations(X,Y,testX,testY,svX,svY,subset,sig1,gam1,kernel_type,'c',process_type(k),[]);
                    [error,spvc,timreq]=avgoperations(avgerr,k,newsvX,newsvY,err,tim,renyie,folds1); 
                    ematrix(k,j) = error;
                    smatrix(k,j) = spvc;
                    tmatrix(k,j) = tl1+timreq;
                elseif (strcmp(process_type(k),'FullL0_norm'))
                    [tim,err,newsvX,newsvY] = modoperations(X,Y,testX,testY,svX,svY,subset,sig2,gam2,kernel_type,'c',process_type(k),[]);
                    [error,spvc,timreq]=avgoperations(avgerr,k,newsvX,newsvY,err,tim,renyie,folds1); 
                    ematrix(k,j) = error;
                    smatrix(k,j) = spvc;
                    tmatrix(k,j) = tl2+timreq;   
                elseif(strcmp(process_type(k),'LSSVMwin'))
                    for n=1:length(windowrange)
                        windowsize=windowrange(n);
                        [tim,err,newsvX,newsvY] = modoperations(X,Y,testX,testY,svX,svY,subset,sig1,gam1,kernel_type,'c',process_type(k),windowsize);
                        [error,spvc,timreq]=avgwinoperations(avgerr,k,n,newsvX,newsvY,err,tim,renyie,folds1);
                        if (~isempty(process_type(strcmp(process_type(:),'WINDOW'))))
                            ematrix(k+length(windowrange)+n-2,j) = error;
                            smatrix(k+length(windowrange)+n-2,j) = spvc;
                            tmatrix(k+length(windowrange)+n-2,j) = tl1+timreq;
                        elseif(isempty(process_type(strcmp(process_type(:),'WINDOW'))))
                            ematrix(k+n-1,j) = error;
                            smatrix(k+n-1,j) = spvc;
                            tmatrix(k+n-1,j) = tl1+timreq;
                        end
                    end;
                elseif(strcmp(process_type(k),'LSSVMwinL'))
                    for n=1:length(windowrange)
                        windowsize=windowrange(n);
                        [tim,err,newsvX,newsvY] = modoperations(X,Y,testX,testY,svX,svY,subset,sig1,gam1,kernel_type,'c',process_type(k),windowsize);
                        [error,spvc,timreq]=avgwinoperations(avgerr,k,n,newsvX,newsvY,err,tim,renyie,folds1);
                        if (~isempty(process_type(strcmp(process_type(:),'WINDOW'))) && ~isempty(process_type(strcmp(process_type(:),'LSSVMwin'))))
                            ematrix(k+length(windowrange)+n,j) = error;
                            smatrix(k+length(windowrange)+n,j) = spvc;
                            tmatrix(k+length(windowrange)+n,j) = tl1+timreq;
                        elseif((isempty(process_type(strcmp(process_type(:),'WINDOW')))&& ~isempty(process_type(strcmp(process_type(:),'LSSVMwin'))))...
                                || (~isempty(process_type(strcmp(process_type(:),'WINDOW')))&& isempty(process_type(strcmp(process_type(:),'LSSVMwin')))))
                            ematrix(k+length(windowrange)+n-2,j) = error;
                            smatrix(k+length(windowrange)+n-2,j) = spvc;
                            tmatrix(k+length(windowrange)+n-2,j) = tl1+timreq;
                        elseif ((isempty(process_type(strcmp(process_type(:),'WINDOW'))))&& (isempty(process_type(strcmp(process_type(:),'LSSVMwin')))))
                            ematrix(k+n-1,j) = error;
                            smatrix(k+n-1,j) = spvc;
                            tmatrix(k+n-1,j) = tl1+timreq;
                        end;
                    end;
               elseif (~strcmp(process_type(k),'L0_norm')&&~strcmp(process_type(k),'Approx_LSSVM')&&~strcmp(process_type(k),'WINDOW')&&~strcmp(process_type(k),'LSSVMwin')&&~strcmp(process_type(k),'LSSVMwinL')&&~strcmp(process_type(k),'FullL0_norm'))
                    [tim,err,newsvX,newsvY] = modoperations(X,Y,testX,testY,svX,svY,subset,sig,gam,kernel_type,'c',process_type(k),[]);
                    [error,spvc,timreq]=avgoperations(avgerr,k,newsvX,newsvY,err,tim,renyie,folds1); 
                    ematrix(k,j) = error;
                    smatrix(k,j) = spvc;
                    tmatrix(k,j) = t1+timreq;
                end;
            end;
        end;
        ematrix
    end;
end


function [ematrix,smatrix,tmatrix] = regression(X,Y,N,renyie,subset,svX,svY,folds,block_size,kernel_type,global_opt,process_type,windowrange,testX,testY)

if ((~isempty(process_type(strcmp(process_type(:),'WINDOW')))&&isempty(process_type(strcmp(process_type(:),'LSSVMwin')))&&isempty(process_type(strcmp(process_type(:),'LSSVMwinL'))))...
        ||(isempty(process_type(strcmp(process_type(:),'WINDOW')))&&isempty(process_type(strcmp(process_type(:),'LSSVMwinL')))&&~isempty(process_type(strcmp(process_type(:),'LSSVMwin'))))...
        ||(isempty(process_type(strcmp(process_type(:),'WINDOW')))&&isempty(process_type(strcmp(process_type(:),'LSSVMwin')))&&~isempty(process_type(strcmp(process_type(:),'LSSVMwinL')))))
    ematrix=zeros(length(process_type)+length(windowrange)-1,folds);
    smatrix=zeros(length(process_type)+length(windowrange)-1,folds);
    tmatrix=zeros(length(process_type)+length(windowrange)-1,folds);
    avgerr=0.0*ones(length(process_type)+length(windowrange)-1,1);
elseif (~isempty(process_type(strcmp(process_type(:),'WINDOW')))&&~isempty(process_type(strcmp(process_type(:),'LSSVMwin')))&&isempty(process_type(strcmp(process_type(:),'LSSVMwinL'))) ...
        ||(~isempty(process_type(strcmp(process_type(:),'WINDOW')))&&isempty(process_type(strcmp(process_type(:),'LSSVMwin')))&&~isempty(process_type(strcmp(process_type(:),'LSSVMwinL'))))...
        ||(isempty(process_type(strcmp(process_type(:),'WINDOW')))&&~isempty(process_type(strcmp(process_type(:),'LSSVMwinL')))&&~isempty(process_type(strcmp(process_type(:),'LSSVMwin')))))
    ematrix=zeros(length(process_type)+2*length(windowrange)-2,folds);
    smatrix=zeros(length(process_type)+2*length(windowrange)-2,folds);
    tmatrix=zeros(length(process_type)+2*length(windowrange)-2,folds);
    avgerr=0.0*ones(length(process_type)+2*length(windowrange)-2,1);
    %glbavgerr=inf*ones(length(process_type)+2*length(windowsize)-2,1);
elseif (~isempty(process_type(strcmp(process_type(:),'WINDOW')))&&~isempty(process_type(strcmp(process_type(:),'LSSVMwin')))&&~isempty(process_type(strcmp(process_type(:),'LSSVMwinL'))))
    ematrix=zeros(length(process_type)+3*length(windowrange)-3,folds);
    smatrix=zeros(length(process_type)+3*length(windowrange)-3,folds);
    tmatrix=zeros(length(process_type)+3*length(windowrange)-3,folds);
    avgerr=0.0*ones(length(process_type)+2*length(windowrange)-3,1);
elseif (isempty(process_type(strcmp(process_type(:),'WINDOW')))&&isempty(process_type(strcmp(process_type(:),'LSSVMwin')))&&isempty(process_type(strcmp(process_type(:),'LSSVMwinL'))))
    ematrix = zeros(length(process_type),folds);
    smatrix = zeros(length(process_type),folds);
    tmatrix = zeros(length(process_type),folds);
    avgerr = 0.0*ones(length(process_type),1);
    %glbavgerr=inf*ones(length(process_type),1);
end;
folds1=1;
%par
for j=1:10
    avgerr=0.0*avgerr;
    %Once obtained the best gamma and sigma perform the 10-fold cross
    %validation to verify the results
    if (isempty(testX) & isempty(testY))
        for l=1:folds1  %10-fold cross-validation
            if (folds1~=1)
                if l==folds1,
                    train = 1:block_size*(l-1); % not used
                    validation = block_size*(l-1)+1:N;
                else
                    train = [1:block_size*(l-1) block_size*l+1:N]; % not used
                    validation = block_size*(l-1)+1:block_size*l;
                end;
            else
                train = 1:N-block_size;
                validation = N-block_size+1:N;
            end;
            if (((isempty(process_type(strcmp(process_type(:),'L0_norm')))) && (isempty(process_type(strcmp(process_type(:),'Approx_LSSVM')))) && (isempty(process_type(strcmp(process_type(:),'LSSVMwin')))) && (isempty(process_type(strcmp(process_type(:),'FullL0_norm'))))...
                    && (isempty(process_type(strcmp(process_type(:),'LSSVMwinL')))))||(~isempty(process_type(strcmp(process_type(:),'FS-LSSVM')))))
                t=cputime;
                [gam,sig]=tunefslssvm({X(train,:),Y(train,:),'f',[],[],kernel_type,global_opt},svX,folds,'mse','simplex','whuber');
                t1=cputime-t;
            end;
            if (~isempty(process_type(strcmp(process_type(:),'L0_norm')))||~isempty(process_type(strcmp(process_type(:),'LSSVMwin')))...
                    ||~isempty(process_type(strcmp(process_type(:),'FullL0_norm'))) ||~isempty(process_type(strcmp(process_type(:),'LSSVMwinL')))||~isempty(process_type(strcmp(process_type(:),'Approx_LSSVM'))))
                tl=cputime;
                if (~isempty(process_type(strcmp(process_type(:),'FullL0_norm'))))
                    [gam2,sig2] = modtunelssvm({X(train,:),Y(train,:),'f',[],[],kernel_type,global_opt},svX,folds,'mse','simplex','whuber');
                    tl2=cputime-tl;
                end;
                tll=cputime;
                if(~isempty(process_type(strcmp(process_type(:),'L0_norm')))||~isempty(process_type(strcmp(process_type(:),'LSSVMwin')))||~isempty(process_type(strcmp(process_type(:),'LSSVMwinL')))||~isempty(process_type(strcmp(process_type(:),'Approx_LSSVM'))))
                    [gam1,sig1] = tunelssvm({svX,svY,'f',[],[],kernel_type,global_opt},'simplex','crossvalidatelssvm',{folds,'mse'},'whuber');
                    tl1=cputime-tll;
                end;

            end;
            %modsparseoperations(X,Y,features,train,validation,svX,svY,subset,sigma_optimal,gam_optimal,min_pts,
            %cutoff,function_type,process_type,window_size,plot_handler)
            for k=1:length(process_type)
                %If no 'WINDOW' based approach
                if (isempty(process_type(strcmp(process_type(:),'WINDOW')))&&isempty(process_type(strcmp(process_type(:),'LSSVMwin')))&&isempty(process_type(strcmp(process_type(:),'LSSVMwinL'))))
                    if (~strcmp(process_type(k),'L0_norm') && ~strcmp(process_type(k),'FullL0_norm') & ~strcmp(process_type(k),'Approx_LSSVM'))
                        [tim,err,newsvX,newsvY] = operations(X,Y,train,validation,svX,svY,subset,sig,gam,kernel_type,'f',process_type(k),[]);
                    elseif (strcmp(process_type(k),'L0_norm')||strcmp(process_type(k),'Approx_LSSVM'))
                        [tim,err,newsvX,newsvY] = operations(X,Y,train,validation,svX,svY,subset,sig1,gam1,kernel_type,'f',process_type(k),[]);
                    elseif (strcmp(process_type(k),'FullL0_norm'))
                        [tim,err,newsvX,newsvY] = operations(X,Y,train,validation,svX,svY,subset,sig2,gam2,kernel_type,'f',process_type(k),[]);
                    end;
                    [error,spvc,timreq]=avgoperations(avgerr,k,newsvX,newsvY,err,tim,renyie,folds1);
                    ematrix(k,j) = error;
                    smatrix(k,j) = spvc;
                    if (strcmp(process_type(k),'FullL0_norm'))
                        tmatrix(k,j) = tl2+timreq;
                    elseif (strcmp(process_type(k),'L0_norm') || strcmp(process_type(k),'Approx_LSSVM'))
                        tmatrix(k,j) = tl1+timreq;
                    else
                        tmatrix(k,j) = t1+timreq;
                    end;
                    %If we perform the 'WINDOW' based operations
                elseif (~isempty(process_type(strcmp(process_type(:),'WINDOW')))||~isempty(process_type(strcmp(process_type(:),'LSSVMwin')))||~isempty(process_type(strcmp(process_type(:),'LSSVMwinL'))))
                    if (strcmp(process_type(k),'WINDOW'))
                        for n=1:length(windowrange)
                            windowsize=windowrange(n);
                            [tim,err,newsvX,newsvY] = operations(X,Y,train,validation,svX,svY,subset,sig,gam,kernel_type,'f',process_type(k),windowsize);
                            [error,spvc,timreq]=avgwinoperations(avgerr,k,n,newsvX,newsvY,err,tim,renyie,folds1);
                            ematrix(k+n-1,j) = error;
                            smatrix(k+n-1,j) = spvc;
                            tmatrix(k+n-1,j) = t1+timreq;
                        end;
                    elseif (strcmp(process_type(k),'L0_norm'))
                        [tim,err,newsvX,newsvY] = operations(X,Y,train,validation,svX,svY,subset,sig1,gam1,kernel_type,'f',process_type(k),[]);
                        [error,spvc,timreq]=avgoperations(avgerr,k,newsvX,newsvY,err,tim,renyie,folds1); 
                        ematrix(k,j) = error;
                        smatrix(k,j) = spvc;
                        tmatrix(k,j) = tl1+timreq;
                    elseif (strcmp(process_type(k),'Approx_LSSVM'))
                        [tim,err,newsvX,newsvY] = operations(X,Y,train,validation,svX,svY,subset,sig1,gam1,kernel_type,'f',process_type(k),[]);
                        [error,spvc,timreq]=avgoperations(avgerr,k,newsvX,newsvY,err,tim,renyie,folds1); 
                        ematrix(k,j) = error;
                        smatrix(k,j) = spvc;
                        tmatrix(k,j) = tl1+timreq;
                    elseif (strcmp(process_type(k),'FullL0_norm'))
                        [tim,err,newsvX,newsvY] = operations(X,Y,train,validation,svX,svY,subset,sig2,gam2,kernel_type,'f',process_type(k),[]);
                        [error,spvc,timreq]=avgoperations(avgerr,k,newsvX,newsvY,err,tim,renyie,folds1); 
                        ematrix(k,j) = error;
                        smatrix(k,j) = spvc;
                        tmatrix(k,j) = tl2+timreq;
                    elseif(strcmp(process_type(k),'LSSVMwin'))
                        for n=1:length(windowrange)
                            windowsize=windowrange(n);
                            [tim,err,newsvX,newsvY] = operations(X,Y,train,validation,svX,svY,subset,sig1,gam1,kernel_type,'f',process_type(k),windowsize);
                            [error,spvc,timreq]=avgwinoperations(avgerr,k,n,newsvX,newsvY,err,tim,renyie,folds1);
                            if (~isempty(process_type(strcmp(process_type(:),'WINDOW'))))
                                ematrix(k+length(windowrange)+n-2,j) = error;
                                smatrix(k+length(windowrange)+n-2,j) = spvc;
                                tmatrix(k+length(windowrange)+n-2,j) = tl1+timreq;
                            elseif(isempty(process_type(strcmp(process_type(:),'WINDOW'))))
                                ematrix(k+n-1,j) = error;
                                smatrix(k+n-1,j) = spvc;
                                tmatrix(k+n-1,j) = tl1+timreq;
                            end
                        end;
                    elseif(strcmp(process_type(k),'LSSVMwinL'))
                        for n=1:length(windowrange)
                            windowsize=windowrange(n);
                            [tim,err,newsvX,newsvY] = operations(X,Y,train,validation,svX,svY,subset,sig1,gam1,kernel_type,'f',process_type(k),windowsize);
                            [error,spvc,timreq]=avgwinoperations(avgerr,k,n,newsvX,newsvY,err,tim,renyie,folds1);
                            if (~isempty(process_type(strcmp(process_type(:),'WINDOW'))) && ~isempty(process_type(strcmp(process_type(:),'LSSVMwin'))))
                                ematrix(k+length(windowrange)+n,j) = error;
                                smatrix(k+length(windowrange)+n,j) = spvc;
                                tmatrix(k+length(windowrange)+n,j) = tl1+timreq;
                            elseif((isempty(process_type(strcmp(process_type(:),'WINDOW')))&& ~isempty(process_type(strcmp(process_type(:),'LSSVMwin'))))...
                                    || (~isempty(process_type(strcmp(process_type(:),'WINDOW')))&& isempty(process_type(strcmp(process_type(:),'LSSVMwin')))))
                                ematrix(k+length(windowrange)+n-2,j) = error;
                                smatrix(k+length(windowrange)+n-2,j) = spvc;
                                tmatrix(k+length(windowrange)+n-2,j) = tl1+timreq;
                            elseif ((isempty(process_type(strcmp(process_type(:),'WINDOW'))))&& (isempty(process_type(strcmp(process_type(:),'LSSVMwin')))))
                                ematrix(k+n-1,j) = error;
                                smatrix(k+n-1,j) = spvc;
                                tmatrix(k+n-1,j) = tl1+timreq;
                            end;
                        end;
                    elseif (~strcmp(process_type(k),'L0_norm')&&~strcmp(process_type(k),'Approx_LSSVM')&&~strcmp(process_type(k),'WINDOW')&&~strcmp(process_type(k),'LSSVMwin')&&~strcmp(process_type(k),'LSSVMwinL')&&~strcmp(process_type(k),'FullL0_norm'))
                        [tim,err,newsvX,newsvY] = operations(X,Y,train,validation,svX,svY,subset,sig,gam,kernel_type,'f',process_type(k),[]);
                        [error,spvc,timreq]=avgoperations(avgerr,k,newsvX,newsvY,err,tim,renyie,folds1); 
                        ematrix(k,j) = error;
                        smatrix(k,j) = spvc;
                        tmatrix(k,j) = t1+timreq;
                    end;
                end;
            end;
            ematrix
        end;
    else
        if (((isempty(process_type(strcmp(process_type(:),'L0_norm')))) && (isempty(process_type(strcmp(process_type(:),'Approx_LSSVM')))) && (isempty(process_type(strcmp(process_type(:),'LSSVMwin')))) && (isempty(process_type(strcmp(process_type(:),'FullL0_norm'))))...
                && (isempty(process_type(strcmp(process_type(:),'LSSVMwinL')))))||(~isempty(process_type(strcmp(process_type(:),'FS-LSSVM')))))
            t=cputime;
            [gam,sig]=tunefslssvm({X,Y,'f',[],[],kernel_type,global_opt},svX,folds,'mse','simplex','whuber');
            t1=cputime-t;
            modelparam = [gam sig];
            csvwrite('modelparam.mat',modelparam);
        end;
        if (~isempty(process_type(strcmp(process_type(:),'L0_norm')))||~isempty(process_type(strcmp(process_type(:),'LSSVMwin')))...
                ||~isempty(process_type(strcmp(process_type(:),'FullL0_norm'))) ||~isempty(process_type(strcmp(process_type(:),'LSSVMwinL'))) || ~isempty(process_type(strcmp(process_type(:),'Approx_LSSVM'))))
            tl=cputime;
            if (~isempty(process_type(strcmp(process_type(:),'FullL0_norm'))))
                [gam2,sig2] = modtunelssvm({X,Y,'f',[],[],kernel_type,global_opt},svX,folds,'mse','simplex','whuber');
                tl2=cputime-tl;
                modelparam = [gam2 sig2];
                csvwrite('modelparam2.mat',modelparam);
            end;
            tll=cputime;
            if(~isempty(process_type(strcmp(process_type(:),'L0_norm')))||~isempty(process_type(strcmp(process_type(:),'LSSVMwin')))||~isempty(process_type(strcmp(process_type(:),'LSSVMwinL')))||~isempty(process_type(strcmp(process_type(:),'Approx_LSSVM'))))
                [gam1,sig1] = tunelssvm({svX,svY,'f',[],[],kernel_type,global_opt},'simplex','crossvalidatelssvm',{folds,'mse'},'whuber');
                tl1=cputime-tll;
                modelparam = [gam1 sig1];
                csvwrite('modelparam1.mat',modelparam);
            end;

        end;
        %modsparseoperations(X,Y,features,train,validation,svX,svY,subset,sigma_optimal,gam_optimal,min_pts,
        %cutoff,function_type,process_type,window_size,plot_handler)
        for k=1:length(process_type)
            %If no 'WINDOW' based approach
            if (isempty(process_type(strcmp(process_type(:),'WINDOW')))&&isempty(process_type(strcmp(process_type(:),'LSSVMwin')))&&isempty(process_type(strcmp(process_type(:),'LSSVMwinL'))))
                if (~strcmp(process_type(k),'L0_norm') && ~strcmp(process_type(k),'FullL0_norm')&&~strcmp(process_type(k),'Approx_LSSVM'))
                    [tim,err,newsvX,newsvY] = modoperations(X,Y,testX,testY,svX,svY,subset,sig,gam,kernel_type,'f',process_type(k),[]);
                elseif (strcmp(process_type(k),'L0_norm')|| strcmp(process_type(k),'Approx_LSSVM'))
                    [tim,err,newsvX,newsvY] = modoperations(X,Y,testX,testY,svX,svY,subset,sig1,gam1,kernel_type,'f',process_type(k),[]);
                elseif (strcmp(process_type(k),'FullL0_norm'))
                    [tim,err,newsvX,newsvY] = modoperations(X,Y,testX,testY,svX,svY,subset,sig2,gam2,kernel_type,'f',process_type(k),[]);
                end;
                [error,spvc,timreq]=avgoperations(avgerr,k,newsvX,newsvY,err,tim,renyie,folds1);
                ematrix(k,j) = error;
                smatrix(k,j) = spvc;
                if (strcmp(process_type(k),'FullL0_norm'))
                    tmatrix(k,j) = tl2+timreq;
                elseif (strcmp(process_type(k),'L0_norm')||strcmp(process_type(k),'Approx_LSSVM')) 
                    tmatrix(k,j) = tl1+timreq;
                else
                    tmatrix(k,j) = t1+timreq;
                end;
                %If we perform the 'WINDOW' based operations
            elseif (~isempty(process_type(strcmp(process_type(:),'WINDOW')))||~isempty(process_type(strcmp(process_type(:),'LSSVMwin')))||~isempty(process_type(strcmp(process_type(:),'LSSVMwinL'))))
                if (strcmp(process_type(k),'WINDOW'))
                    for n=1:length(windowrange)
                        windowsize=windowrange(n);
                        [tim,err,newsvX,newsvY] = modoperations(X,Y,testX,testY,svX,svY,subset,sig,gam,kernel_type,'f',process_type(k),windowsize);
                        [error,spvc,timreq]=avgwinoperations(avgerr,k,n,newsvX,newsvY,err,tim,renyie,folds1);
                        ematrix(k+n-1,j) = error;
                        smatrix(k+n-1,j) = spvc;
                        tmatrix(k+n-1,j) = t1+timreq;
                    end;
                elseif (strcmp(process_type(k),'L0_norm') || strcmp(process_type(k),'Approx_LSSVM'))
                    [tim,err,newsvX,newsvY] = modoperations(X,Y,testX,testY,svX,svY,subset,sig1,gam1,kernel_type,'f',process_type(k),[]);
                    [error,spvc,timreq]=avgoperations(avgerr,k,newsvX,newsvY,err,tim,renyie,folds1); 
                    ematrix(k,j) = error;
                    smatrix(k,j) = spvc;
                    tmatrix(k,j) = tl1+timreq;
                elseif (strcmp(process_type(k),'FullL0_norm'))
                    [tim,err,newsvX,newsvY] = modoperations(X,Y,testX,testY,svX,svY,subset,sig2,gam2,kernel_type,'f',process_type(k),[]);
                    [error,spvc,timreq]=avgoperations(avgerr,k,newsvX,newsvY,err,tim,renyie,folds1); 
                    ematrix(k,j) = error;
                    smatrix(k,j) = spvc;
                    tmatrix(k,j) = tl2+timreq;
                elseif(strcmp(process_type(k),'LSSVMwin'))
                    for n=1:length(windowrange)
                        windowsize=windowrange(n);
                        [tim,err,newsvX,newsvY] = modoperations(X,Y,testX,testY,svX,svY,subset,sig1,gam1,kernel_type,'f',process_type(k),windowsize);
                        [error,spvc,timreq]=avgwinoperations(avgerr,k,n,newsvX,newsvY,err,tim,renyie,folds1);
                        if (~isempty(process_type(strcmp(process_type(:),'WINDOW'))))
                            ematrix(k+length(windowrange)+n-2,j) = error;
                            smatrix(k+length(windowrange)+n-2,j) = spvc;
                            tmatrix(k+length(windowrange)+n-2,j) = tl1+timreq;
                        elseif(isempty(process_type(strcmp(process_type(:),'WINDOW'))))
                            ematrix(k+n-1,j) = error;
                            smatrix(k+n-1,j) = spvc;
                            tmatrix(k+n-1,j) = tl1+timreq;
                        end
                    end;
                elseif(strcmp(process_type(k),'LSSVMwinL'))
                    for n=1:length(windowrange)
                        windowsize=windowrange(n);
                        [tim,err,newsvX,newsvY] = modoperations(X,Y,testX,testY,svX,svY,subset,sig1,gam1,kernel_type,'f',process_type(k),windowsize);
                        [error,spvc,timreq]=avgwinoperations(avgerr,k,n,newsvX,newsvY,err,tim,renyie,folds1);
                        if (~isempty(process_type(strcmp(process_type(:),'WINDOW'))) && ~isempty(process_type(strcmp(process_type(:),'LSSVMwin'))))
                            ematrix(k+length(windowrange)+n,j) = error;
                            smatrix(k+length(windowrange)+n,j) = spvc;
                            tmatrix(k+length(windowrange)+n,j) = tl1+timreq;
                        elseif((isempty(process_type(strcmp(process_type(:),'WINDOW')))&& ~isempty(process_type(strcmp(process_type(:),'LSSVMwin'))))...
                                || (~isempty(process_type(strcmp(process_type(:),'WINDOW')))&& isempty(process_type(strcmp(process_type(:),'LSSVMwin')))))
                            ematrix(k+length(windowrange)+n-2,j) = error;
                            smatrix(k+length(windowrange)+n-2,j) = spvc;
                            tmatrix(k+length(windowrange)+n-2,j) = tl1+timreq;
                        elseif ((isempty(process_type(strcmp(process_type(:),'WINDOW'))))&& (isempty(process_type(strcmp(process_type(:),'LSSVMwin')))))
                            ematrix(k+n-1,j) = error;
                            smatrix(k+n-1,j) = spvc;
                            tmatrix(k+n-1,j) = tl1+timreq;
                        end;
                    end;
                elseif (~strcmp(process_type(k),'L0_norm')&&~strcmp(process_type(k),'Approx_LSSVM')&&~strcmp(process_type(k),'WINDOW')&&~strcmp(process_type(k),'LSSVMwin')&&~strcmp(process_type(k),'LSSVMwinL')&&~strcmp(process_type(k),'FullL0_norm'))
                    [tim,err,newsvX,newsvY] = modoperations(X,Y,testX,testY,svX,svY,subset,sig,gam,kernel_type,'f',process_type(k),[]);
                    [error,spvc,timreq]=avgoperations(avgerr,k,newsvX,newsvY,err,tim,renyie,folds1); 
                    ematrix(k,j) = error;
                    smatrix(k,j) = spvc;
                    tmatrix(k,j) = t1+timreq;
                end;
            end;
        end;
        ematrix
    end
end

function [tim,err,newsvX,newsvY] = operations(X,Y,train,validation,svX,svY,subset,sig,gam,kernel_type,function_type,process_type,windowsize)
t2=cputime;
[err,newsvX,newsvY] = modsparseoperations(X,Y,train,validation,svX,svY,subset,sig,gam,kernel_type,function_type,process_type,windowsize);
tim=cputime-t2;

function [tim,err,newsvX,newsvY] = modoperations(X,Y,testX,testY,svX,svY,subset,sig,gam,kernel_type,function_type,process_type,windowsize)
t2=cputime;
[err,newsvX,newsvY] = testmodsparseoperations(X,Y,testX,testY,svX,svY,subset,sig,gam,kernel_type,function_type,process_type,windowsize);
tim=cputime-t2;

function [error,spvc,timreq] = avgoperations(avgerr,k,newsvX,newsvY,err,tim,renyie,folds1)
avgerr(k) = avgerr(k) + err;
e1=tim;
avgerr(k)=avgerr(k)/folds1;
error = avgerr(k);
spvc = (size(newsvX,1)+length(newsvY))/2;
timreq = (e1+renyie);

function [error,spvc,timreq] = avgwinoperations(avgerr,k,n,newsvX,newsvY,err,tim,renyie,folds1)
avgerr(k+n-1) = avgerr(k+n-1) + err;
e1=tim;
avgerr(k+n-1)=avgerr(k+n-1)/folds1;
error = avgerr(k+n-1);
spvc = (size(newsvX,1)+length(newsvY))/2;
timreq = (e1+renyie);
