close all
gam = 10;
sig2 = 0.2;
type = 'classification';

for i=1:10,
    randn('state',100)
    X1 = (2-i/6)+randn(50,2);
    randn('state',200)     
    X2 = -(2-i/6)+randn(51,2);
         
    X = [X1; X2];  
    Y1 = ones(size(X1,1),1); 
    Y2 = -1*ones(size(X2,1),1);
    Y = [Y1; Y2];

    mdl_in = {X,Y,type,gam,sig2,'RBF_kernel','preprocess'};
    
    % clasification
    [alpha,b] = trainlssvm(mdl_in);
    Yc = simlssvm(mdl_in, {alpha,b},X);
    Yc = lda(X,X,Y);
    err = sum(Yc~=Y);
    disp(['# training error = ', num2str(err)])    

    
%     % test set
    x = -5.0:0.25:5.0;
    y = -5.0:0.25:5.0;
    [xt, yt] = meshgrid(x,y);       
    grid = [xt(:) yt(:)];
    class = simlssvm({X,Y,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b}, grid);
    class = lda(grid, X, Y);
    grid = reshape(class,length(x),length(y));

    
    % make figure    
    figure
    subplot(3,4,i)
    plotlssvm(mdl_in, {alpha,b});
    title(['error=' num2str(err)]);
    hold off
    % disp('Note that the boundaries of decision regions are straight lines.')
    
   
   
    disp(' ')
    disp('Press any key to continue.')
    pause

end
