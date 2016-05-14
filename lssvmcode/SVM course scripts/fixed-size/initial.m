function [X,Y,testX,testY] = initial(X,Y,function_type,testX,testY)

if nargin <4
    testX = [];
    testY = [];
end;

if (function_type=='c')
    BigX = [X ; testX];
    N = size(X,1);
    C = BigX-repmat(mean(BigX),length(BigX),1);  
    D = std(BigX);
    for i=1:size(C,2)
        if (D(i)~=0)
            C(:,i) = C(:,i)/D(i);
        end;
    end;
    BigX = C;
    X = BigX(1:N,:);
    perm = randperm(size(X,1));
    X = X(perm,:);
    Y = Y(perm,:);
    if (~isempty(testX))
        testX = BigX(N+1:end,:);
    end;
else
    data = [X ; testX];
    N = size(X,1);
    C = data-repmat(mean(data),length(data),1);  
    D = std(data);
    for i=1:size(C,2)
        if (D(i)~=0)
            C(:,i) = C(:,i)/D(i);
        end;
    end;
    data = C;
    X = data(1:N,:);
    perm = randperm(size(X,1));
    X = X(perm,:);
    if (~isempty(testX))
        testX = data(N+1:end,:);
    end;
    
    data = [Y ; testY];
    C = data-repmat(mean(data),length(data),1);  
    D = std(data);
    for i=1:size(C,2)
        if (D(i)~=0)
            C(:,i) = C(:,i)/D(i);
        end;
    end;
    data = C;
    Y = data(1:N,:);
    Y = Y(perm,:);
    if (~isempty(testY))
        testY = data(N+1:end,:);
    end;
end;
