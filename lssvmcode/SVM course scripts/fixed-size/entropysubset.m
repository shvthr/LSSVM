function [subset,max_c,logcontribution,maxcontribution] = entropysubset(X,represent_points,kernel_type,sig2,selectionset)
% Performs Renyi Entropy based representative points selection
% INput arguements:
% X = Training set
% represent_points = No of representative points to be selected
% kernel_type = Input kernel type here 'RBF_kernel'
% sig2 = bandwidth for density estimate function
% selectionset = Optional parameter having a random permutation of the
% training set indexes
% Ouput arguements:
% subset = Set of indexes of the representative points
capacity = ceil(represent_points);
if(isempty(selectionset)),
    selectionset = randperm(size(X,1));
    sv = selectionset(1:capacity);
else
    extrainputs = represent_points - length(selectionset);
    leftindices = setdiff((1:size(X,1)),selectionset);
    info = randperm(length(leftindices));
    info = info(1:extrainputs);
    sv = [selectionset leftindices(info)];
end;
%Calculating the Renyi for pre-determined M points
svX = X(sv,:);
totalinfo = zeros(capacity,2);
XXh2 = sum(svX.^2,2)*ones(1,1);
for i=1:capacity
    XXh1 = sum(svX(i,:).^2)*ones(1,capacity);
    temp = XXh1 + XXh2' - 2*svX(i,:)*svX';
    temp = exp(-temp./(2*(sig2(1))));
    totalinfo(i,2) = sum(temp);
    totalinfo(i,1) = i;
end;
capsquare = capacity^2;
totalcrit = sum(totalinfo(:,2),1);
logtotalcrit = -log(totalcrit/capsquare);
%Appending indexes to sv set
%Maximixing the quadratic Renyi Entropy
max_c=logtotalcrit;
e1 = cputime;
for i=1:size(X,1),
    [~,replace]=max(totalinfo(:,2));
    id = totalinfo(totalinfo(:,1)==replace,1);
    %Subtract from totalcrit once for row and once for column and add 1 for
    %diagonal term which is subtracted twice
    temptotalcrit = totalcrit - 2*totalinfo(id,2) + 1;
    %Try to evaluate kernel function 
    tempXXh2 = XXh2;
    tempsvX = svX;
    inputX = X(i,:);
    tempsvX(replace,:) = inputX;
    tempXXh2(id) = sum(inputX.^2,2)*ones(1,1);
    XXh1 = sum(inputX.^2)*ones(1,capacity);
    temp = XXh1' + tempXXh2 - 2*tempsvX*inputX';
    temp1 = exp(-temp./(2*(sig2(1))));
    distance_eval = sum(temp1);
    %Add to totalcrit once for row and once for column and subtract 1 for
    %diagonal term which is added twice;
    temptotalcrit = temptotalcrit + 2*distance_eval - 1;
    logtemptotalcrit = -log(temptotalcrit/capsquare);
    if (max_c < logtemptotalcrit)
        max_c = logtemptotalcrit;
        totalinfo(id,2) = distance_eval;
        totalcrit = sum(totalinfo(:,2));
        XXh2 = tempXXh2;
        svX = tempsvX;
        sv(id) = i;
    end;
end;
t1=cputime-e1;
subset=sv;
svX=X(subset,:);
totalcontribution = sum(sum(kernel_matrix(svX,kernel_type,sig2),2));
max_c;
logcontribution = -log(totalcontribution/capsquare);
maxcontribution = -log(1/capacity);
