close all
fh1=figure; 

for i=1:10,
    randn('state',100)
    X1 = (2-i/6)+randn(50,2);
    randn('state',200)     
    X2 = -(2-i/6)+randn(51,2);
         
    X = [X1; X2];  
    Y1 = ones(size(X1,1),1); 
    Y2 = -1*ones(size(X2,1),1);
    Y = [Y1; Y2];

    
    %
    % LDA: linear discriminant
    %
    Yc = lda(X,X,Y);
    err = sum(Yc~=Y);
    disp(['# training error = ', num2str(err)])    

    %
    % test set
    %
    x = -5.0:0.25:5.0;
    y = -5.0:0.25:5.0;
    [xt, yt] = meshgrid(x,y);
       
    grid = [xt(:) yt(:)];
    class = lda(grid, X, Y);

    
    %
    % make figure
    % 
    
    subplot(3,4,i)
    grid = reshape(class,length(x),length(y));
    h1 = contourf(x,y,grid,2);hold on;
    plot(X1(:,1),X1(:,2),'k+'); 
    plot(X2(:,1),X2(:,2),'k*'); 
    plot(mean(X1(:,1)),mean(X1(:,2)),'g+');
    plot(mean(X2(:,1)),mean(X2(:,2)),'g*');
    %plot(xt(class == 1), yt(class == 1), 'r.', 'MarkerSize', 6);    
    %plot(xt(class == -1), yt(class == -1), 'b.', 'MarkerSize', 6);
    title(['error=' num2str(err)]);     
    %legend([pos neg m1 m2],'positive data','negative data',...
    %       'mean positive data', 'mean negative data');
    hold off
    % disp('Note that the boundaries of decision regions are straight lines.')
    
   
   
    %disp(' ')
    %disp('Press any key to continue.')
    %pause

end
