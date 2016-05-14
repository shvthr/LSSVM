function [process_matrix_err,process_matrix_sv,process_matrix_time] = fslssvm(X,Y,k,function_type,kernel_type,global_opt,user_process,window,testX,testY)

if (nargin < 8)
    testX =[];
    testY = [];
end;

if (~isempty(user_process(strcmp(user_process(:),'WINDOW')))||~isempty(user_process(strcmp(user_process(:),'LSSVMwin'))) || ~isempty(user_process(strcmp(user_process(:),'LSSVMwinL'))))
    windowrange=window;
else
    windowrange=[10,15,20];
end;
process_matrix_err =[];
process_matrix_sv = [];
process_matrix_time = [];

%Initialization here
if (isempty(testX) && isempty(testY))
    if (function_type=='c')
        [X,Y] = initial(X,Y,'c');
    else
        [X,Y] = initial(X,Y,'f');
    end;
else
    if (function_type=='c')
        [X,Y,testX,testY] = initial(X,Y,'c',testX,testY);
    else
        [X,Y,testX,testY] = initial(X,Y,'f',testX,testY);
    end;
end;
N=length(X);
dim=size(X,2);
represent_points = ceil(k*sqrt(N));     %A heuristic approximation for selecting representative points
sig2 = (std(X)*(N^(-1/(dim+4)))).^2;    %Heuristic used in density estimate for Renyi Entropy

%Renyi Entropy calculation
renyit=cputime;
subset = entropysubset(X,represent_points,kernel_type,sig2,[]);
renyie=cputime-renyit;
subset = subset';
svX = X(subset,:);
svY = Y(subset,:);

%Enter the subset, svX and svY as model variables in a file 
modelvariables = [subset svX svY];
csvwrite('modelvariables.mat',modelvariables);

[e,s,t]=fsoperations(X,Y,function_type,kernel_type,global_opt,user_process,windowrange,renyie,subset,svX,svY,testX,testY);
process_matrix_err = [process_matrix_err e'];
process_matrix_sv = [process_matrix_sv s'];
process_matrix_time = [process_matrix_time t'];

if (~isempty(user_process(strcmp(user_process(:),'WINDOW')))||~isempty(user_process(strcmp(user_process(:),'LSSVMwin')))||~isempty(user_process(strcmp(user_process(:),'LSSVMwinL'))))
    if (~isempty(user_process(strcmp(user_process(:),'WINDOW')))&&isempty(user_process(strcmp(user_process(:),'LSSVMwin')))&&isempty(user_process(strcmp(user_process(:),'LSSVMwinL'))))
        label_process = user_process(1:length(user_process)-1);
        for i=1:length(window)
            label_process(length(label_process)+1) = {strcat('FF',num2str(window(i)))};
        end;
    elseif (isempty(user_process(strcmp(user_process(:),'WINDOW')))&&isempty(user_process(strcmp(user_process(:),'LSSVMwinL')))&&~isempty(user_process(strcmp(user_process(:),'LSSVMwin'))))
        label_process = user_process(1:length(user_process)-1);
        for i=1:length(window)
            label_process(length(label_process)+1) = {strcat('LF',num2str(window(i)))};
        end;
     elseif (isempty(user_process(strcmp(user_process(:),'WINDOW')))&&isempty(user_process(strcmp(user_process(:),'LSSVMwin')))&&~isempty(user_process(strcmp(user_process(:),'LSSVMwinL'))))
        label_process = user_process(1:length(user_process)-1);
        for i=1:length(window)
            label_process(length(label_process)+1) = {strcat('LL',num2str(window(i)))};
        end;   
    elseif (~isempty(user_process(strcmp(user_process(:),'WINDOW')))&&~isempty(user_process(strcmp(user_process(:),'LSSVMwin')))&&isempty(user_process(strcmp(user_process(:),'LSSVMwinL'))))
        label_process = user_process(1:length(user_process)-2);
        for i=1:length(window)
            label_process(length(label_process)+1) = {strcat('FF',num2str(window(i)))};
        end;
        for i=1:length(window)
            label_process(length(label_process)+1) = {strcat('LF',num2str(window(i)))};
        end;
    elseif (~isempty(user_process(strcmp(user_process(:),'WINDOW')))&&isempty(user_process(strcmp(user_process(:),'LSSVMwin')))&&~isempty(user_process(strcmp(user_process(:),'LSSVMwinL'))))
        label_process = user_process(1:length(user_process)-2);
        for i=1:length(window)
            label_process(length(label_process)+1) = {strcat('FF',num2str(window(i)))};
        end;
        for i=1:length(window)
            label_process(length(label_process)+1) = {strcat('LL',num2str(window(i)))};
        end;
    elseif (isempty(user_process(strcmp(user_process(:),'WINDOW')))&&~isempty(user_process(strcmp(user_process(:),'LSSVMwin')))&&~isempty(user_process(strcmp(user_process(:),'LSSVMwinL'))))
        label_process = user_process(1:length(user_process)-2);
        for i=1:length(window)
            label_process(length(label_process)+1) = {strcat('LF',num2str(window(i)))};
        end;
        for i=1:length(window)
            label_process(length(label_process)+1) = {strcat('LL',num2str(window(i)))};
        end;
    elseif (~isempty(user_process(strcmp(user_process(:),'WINDOW')))&&~isempty(user_process(strcmp(user_process(:),'LSSVMwin')))&&~isempty(user_process(strcmp(user_process(:),'LSSVMwinL'))))
        label_process = user_process(1:length(user_process)-3);
        for i=1:length(window)
            label_process(length(label_process)+1) = {strcat('FF',num2str(window(i)))};
        end;
        for i=1:length(window)
            label_process(length(label_process)+1) = {strcat('LF',num2str(window(i)))};
        end;
        for i=1:length(window)
            label_process(length(label_process)+1) = {strcat('LL',num2str(window(i)))};
        end;
    end;
    figure;
    boxplot(process_matrix_err,'Label',label_process);
    ylabel('Error estimate');
    title('Error Comparison for different approaches (user processes)');
    figure;
    boxplot(process_matrix_sv,'Label',label_process);
    ylabel('SV estimate');
    title('Number of SV for different approaches (user processes)');
    figure;
    boxplot(process_matrix_time,'Label',label_process);
    ylabel('Time estimate');
    title('Comparison for time taken by different approaches (user processes)');
else
    figure;
    boxplot(process_matrix_err,'Label',user_process);
    ylabel('Error estimate');
    title('Error Comparison for different approaches (user processes)');
    figure;
    boxplot(process_matrix_sv,'Label',user_process);
    ylabel('SV estimate');
    title('Number of SV for different approaches (user processes)');
    figure;
    boxplot(process_matrix_time,'Label',user_process);
    ylabel('Time estimate');
    title('Comparison for time taken by different approaches (user processes)');
end;
