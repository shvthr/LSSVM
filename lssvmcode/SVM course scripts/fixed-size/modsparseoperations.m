function [err,newsvX,newsvY] = modsparseoperations(X,Y,train,test,svX,svY,subset,sig,gam,kernel_type,function_type,process_type,window_size,plot_handler)

addpath('../LSSVMlab');
warning off all;

if ((~strcmp(process_type,'FullL0_norm')) && (~strcmp(process_type,'L0_norm')) && (~strcmp(process_type,'LSSVMwin')) && (~strcmp(process_type,'LSSVMwinL')))
    features = AFEm(svX,kernel_type,sig,X);
end;
if (strcmp(process_type,'FullL0_norm'))
    Xfeatures = kernel_matrix(X,kernel_type,sig,svX);
    omega = kernel_matrix(svX,kernel_type,sig,svX);
end;
newsvX=[];
newsvY=[];
%Perform the FS-LSSVM
if (function_type=='c' && (~strcmp(process_type,'FullL0_norm')) && (~strcmp(process_type,'L0_norm')) && (~strcmp(process_type,'LSSVMwin')) && (~strcmp(process_type,'LSSVMwinL')))
    %Obtain the w and b by solving the equation A_v*[w b]' = c_v
    %We are using the simple method to provide results (we can also extend
    %the same to optimized version)
    trainY = Y(train,:);
    testY = Y(test,:);
    [w,b,testYh] = ridgeregress(features(train,:),trainY,gam,svX,features(test,:)); 
    %number of support vectors required
    nb_sv=size(features(train,:),2);
    w=w(1:nb_sv,:);
    trysvX=svX(1:nb_sv,:);
    trysvY=svY(1:nb_sv,:);
    testYh = sign(testYh);
    if (strcmp(process_type,'FS-LSSVM'))
        newsvX = trysvX;
        newsvY = trysvY;
    end;
    err = sum(testYh~=testY)/length(testYh);
    %Note all the support vectors might not be part of the training set 
    subset = subset(1:nb_sv,:);
    svfeatures = features(subset,:);
end;

%Perform the L0_norm based classification
if (function_type=='c' && ((strcmp(process_type,'FullL0_norm')) || (strcmp(process_type,'L0_norm')) || (strcmp(process_type,'LSSVMwin')) || (strcmp(process_type,'LSSVMwinL')) || (strcmp(process_type,'Approx_LSSVM'))))
    Xt = X(test,:);
    testY = Y(test,:);
    if (strcmp(process_type,'FullL0_norm'))
        trainfeatures=Xfeatures(train,:);
        trainY = Y(train,:);
        [~,~,testYh]=modridgeregress(trainfeatures,trainY,gam,kernel_type,sig,svX,omega,Xt);
        testYh=sign(testYh);
        err=sum(testY~=testYh)/length(testY);
        newsvX=svX;
        newsvY=svY;
        return;
    else
        [alpha,b] = trainlssvm({svX,svY,'c',gam,sig,kernel_type});
        if (strcmp(process_type,'Approx_LSSVM'))
            testYh = simlssvm({svX,svY,'c',gam,sig,kernel_type},Xt,{alpha,b});
            %testYh = sign(alpha'*kernel_matrix(svX,kernel_type,sig,Xt)+b)';
            newsvX = svX;
            newsvY = svY;
            err = sum(testY~=testYh)/length(testYh);
            return;
        end;
    end;
end;

%Perform the FS-LSSVM based regression
if (function_type=='f' && (~strcmp(process_type,'FullL0_norm')) && (~strcmp(process_type,'L0_norm')) && (~strcmp(process_type,'LSSVMwin')) && (~strcmp(process_type,'LSSVMwinL')))
    trainY = Y(train,:);
    testY = Y(test,:);
    [w,b,testYh] = ridgeregress(features(train,:),trainY,gam,svX,features(test,:));
    %number of support vectors required
    nb_sv=size(features(train,:),2);
    w=w(1:nb_sv,:);
    trysvX=svX(1:nb_sv,:);
    trysvY=svY(1:nb_sv,:);
    if (strcmp(process_type,'FS-LSSVM'))
        newsvX = trysvX;
        newsvY = trysvY;
    end;
    err = mse(testYh-testY);
    %Note all the support vectors might not be part of the training set 
    subset = subset(1:nb_sv,:);
    svfeatures = features(subset,:);
end;

%Perform the L0_norm based regression
if (function_type=='f' && (strcmp(process_type,'FullL0_norm') || strcmp(process_type,'L0_norm') || (strcmp(process_type,'LSSVMwin')) || (strcmp(process_type,'LSSVMwinL'))|| (strcmp(process_type,'Approx_LSSVM'))))
    Xt = X(test,:);
    testY =Y(test,:);
    if (strcmp(process_type,'FullL0_norm'))
        trainfeatures = Xfeatures(train,:);
        trainY = Y(train,:);
        [~,~,testYh]=modridgeregress(trainfeatures,trainY,gam,kernel_type,sig,svX,omega,Xt);
        err=mse(testYh-testY);
        newsvX=svX;
        newsvY=svY;
        return;
    else
        [alpha,b] = trainlssvm({svX,svY,'f',gam,sig,kernel_type});
        if (strcmp(process_type,'Approx_LSSVM'))
            testYh = simlssvm({svX,svY,'f',gam,sig,kernel_type},Xt,{alpha,b});
            %testYh = (alpha'*kernel_matrix(svX,kernel_type,sig,Xt)+b)';
            err = mse(testY-testYh);
            newsvX = svX;
            newsvY = svY;
        end;
    end;
end;

%Perform second level of sparsity 
%Now to perform L0-norm on this set of selected support vectors 
if(strcmp(process_type,'SV_L0_norm')),
    %We now to get the initial alpha and b values
    alpha=w;
    %alpha=ones(length(w),1);
    subsetsize=length(alpha);
    lambda=zeros(subsetsize,subsetsize); %Initialize the lambda values as alpha values
    for i=1:size(alpha,1),
        lambda(i,i)=alpha(i);
    end
    max_iterations=50;
    alpha_prev=alpha;
    pred_new=alpha;
    K=svfeatures*svfeatures';
    for i=1:max_iterations,         
        %Use approximate formulations using the Nystrom based vectors
        P = lambda\K;
        H = K*P+(1.0/gam)*eye(subsetsize);
        v = H\trysvY;                        %Making inv(H)*y;
        nu = H\ones(subsetsize,1);        %Making inv(H)*ones(subsetsize,1);
        b = ones(subsetsize,1)'*v/(ones(subsetsize,1)'*nu);
        beta = v - b*nu;
        pred_new = P*beta;
        if ((norm((alpha_prev-pred_new),2)/subsetsize)<(10^-4)),
            break;
        end
        for j=1:size(pred_new,1),
            lambda(j,j) = 1.0/(pred_new(j)^2);
        end
        alpha_prev=pred_new;
    end
    indexes=find(abs(pred_new)>10^-6);
    if (isempty(indexes))
        indexes = 1:size(trysvX,1);
        fprintf('Entered Here\n');
    end;
    %Selecting the basis set using the indexes of these alpha
    newsvY=trysvY(indexes,:);
    newsvX=trysvX(indexes,:);
    
%Perform L0_norm taking the contribution of all points 
elseif (strcmp(process_type,'ALL_L0_norm'))
    alpha = w;
    subsetsize = length(alpha);
    lambda = zeros(subsetsize,subsetsize);
    for i=1:size(alpha,1)
        lambda(i,i) = alpha(i);
    end;
    max_iterations = 50;
    alpha_prev = alpha;
    datapoints=length(features(train,:));
    K=zeros(subsetsize,subsetsize);
    OneK = zeros(1,subsetsize);
    OneY = 0;
    One = 0;
    KY = zeros(subsetsize,1);
    if (datapoints>=50000)
        blocks=ceil(datapoints/50000);
        for j=1:blocks
            if (j==blocks)
                blockindex = (j-1)*50000+1;
                Xtrain = features(blockindex:datapoints,:);
                Ytrain = Y(blockindex:datapoints,:);
            else
                blockindex1 = (j-1)*50000+1;
                blockindex2 = (j)*50000;
                Xtrain = features(blockindex1:blockindex2,:);
                Ytrain = Y(blockindex1:blockindex2,:);
            end;
            onevector = ones(1,size(Xtrain,1));
            partK = Xtrain*svfeatures;
            K = K + partK'*partK;
            OneK = OneK + onevector*partK;
            OneY = OneY + onevector*Ytrain;
            One = One + onevector*onevector';
            KY = KY + partK'*Ytrain;
            clear Xtrain;
            clear Ytrain;
        end;
        for i=1:max_iterations,
            H = ((1/gam)*lambda)+K;
            Xe = [H OneK'; OneK One];
            Ye = [KY;OneY];
            sol = Xe\Ye;
            pred_new = sol(1:end-1,:);
            b = sol(end,:);
            if ((norm((alpha_prev-pred_new),2)/subsetsize)<(10^-4)),
                break;
            end;
            for k=1:size(pred_new,1),
                lambda(k,k) = 1.0/(pred_new(k)^2);
            end;
            alpha_prev=pred_new;
        end;
    else
        onevector=ones(1,datapoints);
        Q = features(train,:)*svfeatures';
        K = Q'*Q;
        OneK = onevector*Q;
        OneY = onevector*Y(train,:);
        One = onevector*onevector';
        KY = Q'*Y(train,:);
        for i=1:max_iterations
            %Use approximate formulations using the Nystrom based vectors
            H=((1/gam)*lambda)+K;
            Xe = [H OneK';OneK One];
            Ye = [KY;OneY];
            sol = Xe\Ye;
            pred_new = sol(1:end-1,:);
            b = sol(end,:);
            if ((norm((alpha_prev-pred_new),2)/subsetsize)<(10^-4)),
                break;
            end;
            for j=1:size(pred_new,1),
                lambda(j,j) = 1.0/(pred_new(j)^2);
            end;
            alpha_prev=pred_new;
        end;
    end;
    indexes=find(abs(pred_new)>10^-6);
    if (isempty(indexes))
        fprintf('Entered Here\n');
        indexes = 1:size(trysvY,1);
    end;
    %Selecting the basis set using the indexes of these alpha    
    newsvX=trysvX(indexes,:);
    newsvY=trysvY(indexes,:); 

%Perform L0_norm without the Nystrom Approximation and with original set of support vectors and original data space  
elseif (strcmp(process_type,'L0_norm') || strcmp(process_type,'FullL0_norm'))
    %We now to get the initial alpha and b values
    subsetsize=length(alpha);
    lambda=zeros(subsetsize,subsetsize); %Initialize the lambda values as alpha values
    for i=1:size(alpha,1),
        lambda(i,i)=alpha(i);
    end
    max_iterations=50;
    alpha_prev=alpha;
    b_prev = b;
    datapoints=length(X(train,:));
    K=zeros(subsetsize,subsetsize);
    OneK = zeros(1,subsetsize);
    OneY = 0;
    One = 0;
    KY = zeros(subsetsize,1);
    if (datapoints>=50000)
        blocks=ceil(datapoints/50000);
        for j=1:blocks
            if (j==blocks)
                blockindex = (j-1)*50000+1;
                Xtrain = X(blockindex:datapoints,:);
                Ytrain = Y(blockindex:datapoints,:);
            else
                blockindex1 = (j-1)*50000+1;
                blockindex2 = (j)*50000;
                Xtrain = X(blockindex1:blockindex2,:);
                Ytrain = Y(blockindex1:blockindex2,:);
            end;
            onevector = ones(1,size(Xtrain,1));
            partK = kernel_matrix(Xtrain,kernel_type,sig,svX);
            K = K + partK'*partK;
            OneK = OneK + onevector*partK;
            OneY = OneY + onevector*Ytrain;
            One = One + onevector*onevector';
            KY = KY + partK'*Ytrain;
            clear Xtrain;
            clear Ytrain;
        end;
        for i=1:max_iterations,
            H = ((1/gam)*lambda)+K;
            Xe = [H OneK'; OneK One];
            Ye = [KY;OneY];
            sol = Xe\Ye;
            pred_new = sol(1:end-1,:);
            b = sol(end,:);
            if ((norm((alpha_prev-pred_new),2)/subsetsize)<(10^-4)),
                break;
            end;
            for k=1:size(pred_new,1),
                lambda(k,k) = 1.0/(pred_new(k)^2);
            end;
            alpha_prev=pred_new;
        end;
    else
        onevector = ones(1,datapoints);
        Q = kernel_matrix(X(train,:),kernel_type,sig,svX);
        K = Q'*Q;
        OneK = onevector*Q;
        OneY = onevector*Y(train,:);
        One = onevector*onevector';
        KY = Q'*Y(train,:);
        for i=1:max_iterations
            %Use approximate formulations using the Nystrom based vectors
            H=((1/gam)*lambda)+K;
            Xe = [H OneK';OneK One];
            Ye = [KY;OneY];
            sol = Xe\Ye;
            pred_new = sol(1:end-1,:);
            b = sol(end,:);
            if ((norm((alpha_prev-pred_new),2)/subsetsize)<(10^-4)),
                break;
            end;
            for j=1:size(pred_new,1),
                lambda(j,j) = 1.0/(pred_new(j)^2);
            end;
            alpha_prev=pred_new;
        end;
    end
    indexes=find(abs(pred_new)>10^-6);
    if (isempty(indexes))
        fprintf('Entered Here\n');
        newsvX=svX;
        newsvY=svY;
        req_alpha = alpha;
        req_b = b_prev;
    else
        newsvX=svX(indexes,:);
        newsvY=svY(indexes,:);
        req_alpha = pred_new(indexes,:);
        req_b = b;
    end;
end;

%Now performing the misclassification estimate
if (function_type=='c' && (strcmp(process_type,'SV_L0_norm')||(strcmp(process_type,'ALL_L0_norm'))))
    %Used the precomputed train and test features rather than computing
    %again
    clear features;
    newfeatures = AFEm(newsvX,kernel_type,sig,X);
    [w,b,testYh]=ridgeregress(newfeatures(train,:),trainY,gam,newsvX,newfeatures(test,:));
    clear newfeatures;
    testYh=sign(testYh);
    err=sum(testYh~=testY)/length(testYh);
    
end

%Now performing the misclassification estimate for L0_norm
if (function_type=='c' && ((strcmp(process_type,'L0_norm'))||(strcmp(process_type,'FullL0_norm'))))
    testYh = sign(req_alpha'*kernel_matrix(newsvX,kernel_type,sig,Xt)+req_b)';
    w =  req_alpha;
    b = req_b;
    err = sum(testY~=testYh)/length(testYh);
end;

%Now performing the mse estimate 
if (function_type=='f' && (strcmp(process_type,'SV_L0_norm')||(strcmp(process_type,'ALL_L0_norm'))))
    clear features;
    newfeatures = AFEm(newsvX,kernel_type,sig,X);
    [w,b,testYh]=ridgeregress(newfeatures(train,:),trainY,gam,newsvX,newfeatures(test,:));
    clear newfeatures;
    err = mse(testY-testYh);
end

%Now performing the mse estimate for L0_norm
if (function_type=='f' && ((strcmp(process_type,'L0_norm'))|| (strcmp(process_type,'FullL0_norm'))))
    testYh = (req_alpha'*kernel_matrix(newsvX,kernel_type,sig,Xt)+req_b)';
    w = req_alpha;
    b = req_b;
    err = mse(testY-testYh);
end;

%Now select a window from the support vectors which are correctly
%classified and train on just these 
if (function_type=='c' && strcmp(process_type,'WINDOW'))
    svtesth=svfeatures*w+b;
    %Select the support vectors which are correctly classified
    svtestclass=sign(svtesth);
    svtesth=[svtesth(svtestclass==trysvY) find(svtestclass==trysvY)];         %Have the correctly classified ones with indexes
    svtesth=sortrows(svtesth);
    partsize = ceil((1.0*window_size*length(svtesth))/100);
    %tradeoff = ceil((1.0*window_size*length(svtesth))/400);
    tradeoff = 0;
    %Positive part of greater than margin 1
    possvtesth = svtesth(svtesth(:,1)>0,:);
    %Negative part of lesser than margin -1
    negsvtesth = svtesth(svtesth(:,1)<0,:);
    req_possvtesth=[];
    req_negsvtesth=[];
    %If not enough support vectors away from margin
    if (length(negsvtesth)>=partsize && length(possvtesth)>=partsize)
        pos_left_partsize = ceil(partsize/2);
        pos_right_partsize = floor(partsize/2);
        req_possvtesth = [possvtesth(tradeoff+1:tradeoff+pos_left_partsize,:);possvtesth(end-pos_right_partsize+1:end,:)];
        neg_left_partsize = floor(partsize/2);
        neg_right_partsize = ceil(partsize/2);
        req_negsvtesth = [negsvtesth(1:neg_left_partsize,:);negsvtesth(end-tradeoff-neg_right_partsize+1:end-tradeoff,:)];
    elseif (length(negsvtesth)<partsize)
        req_negsvtesth=negsvtesth;
        if(length(possvtesth)>=partsize)
            pos_left_partsize = ceil(partsize/2);
            pos_right_partsize = floor(partsize/2);
            req_possvtesth = [possvtesth(1:pos_left_partsize,:);possvtesth(end-pos_right_partsize+1:end,:)];
        end;
    elseif(length(possvtesth)<partsize) 
        req_possvtesth=possvtesth;
        if (length(negsvtesth)>=partsize)
            neg_left_partsize = floor(partsize/2);
            neg_right_partsize = ceil(partsize/2);
            req_negsvtesth = [negsvtesth(1:neg_left_partsize,:);negsvtesth(end-neg_right_partsize+1:end,:)];  
        end;
    end;
    %Support vectors within the window limits
    modsvtesth = [req_possvtesth;req_negsvtesth];
    indexes = modsvtesth(:,2);
    newsvX = trysvX(indexes,:);
    newsvY = trysvY(indexes,:);
    clear features;
    newfeatures = AFEm(newsvX,kernel_type,sig,X);
    trainfeatures = newfeatures(train,:);
    testfeatures = newfeatures(test,:);
    [~,~,testYh]=ridgeregress(trainfeatures,trainY,gam,newsvX,testfeatures);
    testYh = sign(testYh);
    err = sum(testYh~=testY)/length(testYh);
end

%Now select the window for regression and perform the Nystrom approximation
%on the better sets of points
if (function_type=='f' && strcmp(process_type,'WINDOW'))
    svtesth=svfeatures*w+b;
    indexes=1:size(trysvY,1);
    svtesth = [(svtesth-trysvY).^2 indexes'];
    partsize = ceil((window_size*size(trysvY,1))/100);
    svtesth = sortrows(svtesth);
    indexes = svtesth(1:partsize,2);
    newsvX=trysvX(indexes,:);
    newsvY=trysvY(indexes,:);
    clear features;
    newfeatures = AFEm(newsvX,kernel_type,sig,X);
    trainfeatures = newfeatures(train,:);
    testfeatures = newfeatures(test,:);
    [~,~,testYh]=ridgeregress(trainfeatures,trainY,gam,newsvX,testfeatures);
    err = mse(testYh-testY);  
end;

%Obtain the support vectors by selecting a pool of support vectors from
%LSSVM using window size and then perform FS-LSSVM (allows to use all the training points) on top of it.
if (function_type=='c' && strcmp(process_type,'LSSVMwin'))
    svtesth=simlssvm({svX,svY,'c',gam,sig,kernel_type},{alpha,b},svX);
    %Select the support vectors which are correctly classified
    svtestclass=sign(svtesth);
    svtesth=[svtesth(svtestclass==svY) find(svtestclass==svY)];         %Have the correctly classified ones with indexes
    svtesth=sortrows(svtesth);
    partsize = ceil((1.0*window_size*length(svtesth))/100);
    tradeoff = 0;
    %Positive part of greater than margin 1
    possvtesth = svtesth(svtesth(:,1)>0,:);
    %Negative part of lesser than margin -1
    negsvtesth = svtesth(svtesth(:,1)<0,:);
    req_possvtesth=[];
    req_negsvtesth=[];
    %If not enough support vectors away from margin
    if (length(negsvtesth)>=partsize && length(possvtesth)>=partsize)
        pos_left_partsize = ceil(partsize/2);
        pos_right_partsize = floor(partsize/2);
        req_possvtesth = [possvtesth(tradeoff+1:tradeoff+pos_left_partsize,:);possvtesth(end-pos_right_partsize+1:end,:)];
        neg_left_partsize = floor(partsize/2);
        neg_right_partsize = ceil(partsize/2);
        req_negsvtesth = [negsvtesth(1:neg_left_partsize,:);negsvtesth(end-tradeoff-neg_right_partsize+1:end-tradeoff,:)];
    elseif (length(negsvtesth)<partsize)
        req_negsvtesth=negsvtesth;
        if(length(possvtesth)>=partsize)
            pos_left_partsize = ceil(partsize/2);
            pos_right_partsize = floor(partsize/2);
            req_possvtesth = [possvtesth(1:pos_left_partsize,:);possvtesth(end-pos_right_partsize+1:end,:)];
        end;
    elseif(length(possvtesth)<partsize) 
        req_possvtesth=possvtesth;
        if (length(negsvtesth)>=partsize)
            neg_left_partsize = floor(partsize/2);
            neg_right_partsize = ceil(partsize/2);
            req_negsvtesth = [negsvtesth(1:neg_left_partsize,:);negsvtesth(end-neg_right_partsize+1:end,:)];  
        end;
    end;
    %Support vectors within the window limits
    modsvtesth = [req_possvtesth;req_negsvtesth];
    indexes = modsvtesth(:,2);
    newsvX = svX(indexes,:);
    newsvY = svY(indexes,:);
    [gam,sig]=tunefslssvm({X(train,:),Y(train,:),'c',[],[],kernel_type},newsvX,10,'misclass','simplex');
    newfeatures = AFEm(newsvX,kernel_type,sig,X);
    [~,~,testYh]=ridgeregress(newfeatures(train,:),Y(train,:),gam,newsvX,newfeatures(test,:));
    testYh = sign(testYh);
    err = sum(testYh~=testY)/length(testYh);
end

%Now select the window for regression based on LSSVM and perform the Nystrom approximation
%on the better sets of points
if (function_type=='f' && strcmp(process_type,'LSSVMwin'))
    svtesth=simlssvm({svX,svY,'f',gam,sig,kernel_type},{alpha,b},svX);
    indexes=1:size(svY,1);
    svtesth = [(svtesth-svY).^2 indexes'];
    partsize = ceil((window_size*size(svY,1))/100);
    svtesth = sortrows(svtesth);
    indexes = svtesth(1:partsize,2);
    newsvX=svX(indexes,:);
    newsvY=svY(indexes,:);
    [gam,sig]=tunefslssvm({X(train,:),Y(train,:),'f',[],[],kernel_type},newsvX,10,'mse','simplex','whuber');
    newfeatures = AFEm(newsvX,kernel_type,sig,X);
    [~,~,testYh]=ridgeregress(newfeatures(train,:),Y(train,:),gam,newfeatures(test,:));
    err = mse(testYh-testY);  
end;

%Obtain the support vectors by selecting a pool of support vectors from
%LSSVM using window size and then perform LSSVM (allows to use all the training points) on top of it in primal.
if (function_type=='c' && strcmp(process_type,'LSSVMwinL'))
    svtesth=simlssvm({svX,svY,'c',gam,sig,kernel_type},{alpha,b},svX);
    %Select the support vectors which are correctly classified
    svtestclass=sign(svtesth);
    svtesth=[svtesth(svtestclass==svY) find(svtestclass==svY)];         %Have the correctly classified ones with indexes
    svtesth=sortrows(svtesth);
    partsize = ceil((1.0*window_size*length(svtesth))/100);
    tradeoff = 0;
    %Positive part of greater than margin 1
    possvtesth = svtesth(svtesth(:,1)>0,:);
    %Negative part of lesser than margin -1
    negsvtesth = svtesth(svtesth(:,1)<0,:);
    req_possvtesth=[];
    req_negsvtesth=[];
    %If not enough support vectors away from margin
    if (length(negsvtesth)>=partsize && length(possvtesth)>=partsize)
        pos_left_partsize = ceil(partsize/2);
        pos_right_partsize = floor(partsize/2);
        req_possvtesth = [possvtesth(tradeoff+1:tradeoff+pos_left_partsize,:);possvtesth(end-pos_right_partsize+1:end,:)];
        neg_left_partsize = floor(partsize/2);
        neg_right_partsize = ceil(partsize/2);
        req_negsvtesth = [negsvtesth(1:neg_left_partsize,:);negsvtesth(end-tradeoff-neg_right_partsize+1:end-tradeoff,:)];
    elseif (length(negsvtesth)<partsize)
        req_negsvtesth=negsvtesth;
        if(length(possvtesth)>=partsize)
            pos_left_partsize = ceil(partsize/2);
            pos_right_partsize = floor(partsize/2);
            req_possvtesth = [possvtesth(1:pos_left_partsize,:);possvtesth(end-pos_right_partsize+1:end,:)];
        end;
    elseif(length(possvtesth)<partsize) 
        req_possvtesth=possvtesth;
        if (length(negsvtesth)>=partsize)
            neg_left_partsize = floor(partsize/2);
            neg_right_partsize = ceil(partsize/2);
            req_negsvtesth = [negsvtesth(1:neg_left_partsize,:);negsvtesth(end-neg_right_partsize+1:end,:)];  
        end;
    end;
    %Support vectors within the window limits
    modsvtesth = [req_possvtesth;req_negsvtesth];
    indexes = modsvtesth(:,2);
    newsvX = svX(indexes,:);
    newsvY = svY(indexes,:);
    OmegasvX = kernel_matrix(newsvX,kernel_type,sig,newsvX);
    Kred = kernel_matrix(X(train,:),kernel_type,sig,newsvX);
    Omegared = Kred'*Kred;
    trainpoints = length(train);
    onevector = ones(trainpoints,1);
    H = (1/gam)*OmegasvX + Omegared;
    P = Kred'*(Y(train,:)-b*onevector);
    beta = H\P;
    b = (Y(train,:)'*onevector - beta'*Kred'*onevector)/trainpoints;
    testYh = (beta'*kernel_matrix(newsvX,kernel_type,sig,X(test,:))+b)';
    testYh = sign(testYh);
    err = sum(testYh~=testY)/length(testYh);
end

%Now select the window for regression based on LSSVM and perform the LSSVM
%on all the training points on this reduced set in the primal
if (function_type=='f' && strcmp(process_type,'LSSVMwinL'))
    svtesth=simlssvm({svX,svY,'f',gam,sig,kernel_type},{alpha,b},svX);
    indexes=1:size(svY,1);
    svtesth = [(svtesth-svY).^2 indexes'];
    partsize = ceil((window_size*size(svY,1))/100);
    svtesth = sortrows(svtesth);
    indexes = svtesth(1:partsize,2);
    newsvX=svX(indexes,:);
    newsvY=svY(indexes,:);
    [gam,sig] = tunelssvm({newsvX,newsvY,'f',[],[],kernel_type},'simplex','crossvalidatelssvm',{10,'mse'},'whuber');
    OmegasvX = kernel_matrix(newsvX,kernel_type,sig,newsvX);
    Kred = kernel_matrix(X(train,:),kernel_type,sig,newsvX);
    Omegared = Kred'*Kred;
    trainpoints = length(train);
    onevector = ones(trainpoints,1);
    H = (1/gam)*OmegasvX + Omegared;
    P = Kred'*(Y(train,:)-b*onevector);
    beta = H\P;
    b = (Y(train,:)'*onevector - beta'*Kred'*onevector)/trainpoints;
    testYh = (beta'*kernel_matrix(newsvX,kernel_type,sig,X(test,:))+b)';
    err = mse(testYh-testY);  
end;
