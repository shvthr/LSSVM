function r = digitsdn2(sig2)
%
% Experiments on the handwriting data set on kPCA for reconstruction and denoising
%
%%amr
%load('E:\Education\KULEUVEN\2-Semester2\SVM\SVM-rep\lssvm\hw3\Digits'); 
clear size
[N, dim]=size(X);
maxx=max(max(X));



%
% Add noise to the digit maps
%

noisefactor =0.3;
noise = noisefactor*maxx; % sd for Gaussian noise

Xn = X; 
for i=1:N;
  randn('state', i);
  Xn(i,:) = X(i,:) + noise*randn(1, dim);
end

Xnt = Xtest2; 

% select training set
Xtr = X(1:2:end,:);
Xvl = X(2:2:end,:);

sig2 =dim*mean(var(Xtr)); % rule of thumb

sigmafactor = 1;
sig2=sig2*sigmafactor;

% kernel based Principal Component Analysis using the original training data
%


disp('Kernel PCA: extract the principal eigenvectors in feature space');
disp(['sig2 = ', num2str(sig2)]);


% kernel PCA
[lam,U] = kpca(Xtr,'RBF_kernel',sig2,[],'eig',240); 
[lam, ids]=sort(-lam); lam = -lam; U=U(:,ids);


% choose the digits for test
digs=[0:9]; ndig=length(digs);
m=2; % Choose the mth data for each digit 
Xdt=zeros(ndig,dim);


% figure of all digits
% figure(iii); 
% colormap('gray'); 
% title('Denosing using linear PCA'); tic
% which number of eigenvalues of kpca
npcs = [2.^(0:6) 95];
lpcs = length(npcs);
for k=1:lpcs;
 nb_pcs=npcs(k); 
 disp(['nb_pcs = ', num2str(nb_pcs)]); 
 Ud=U(:,(1:nb_pcs)); lamd=lam(1:nb_pcs);
    
 for i=1:ndig
   dig=digs(i);
   fprintf('digit %d : ', dig)
   xt=Xvl(i,:);
%     if k==1 
%          % plot the original clean digits
%          subplot(2+lpcs, ndig, i);
%          pcolor(1:15,16:-1:1,reshape(Xtest1(i,:), 15, 16)'); shading interp; 
%          set(gca,'xticklabel',[]);set(gca,'yticklabel',[]);
%          if i==1, ylabel('original'), end 
%          % plot the noisy digits 
%          subplot(2+lpcs, ndig, i+ndig); 
%          pcolor(1:15,16:-1:1,reshape(xt, 15, 16)'); shading interp; 
%          set(gca,'xticklabel',[]);set(gca,'yticklabel',[]);        
%          if i==1, ylabel('noisy'), end
%          drawnow
%     end  
    tic
    Xdt(i,:) = preimage_rbf(Xtr,sig2,Ud,xt,'denoise');
    r(i) = norm(Xdt(i,:) - xt);
    t(i)=toc;
    %subplot(2+lpcs, ndig, i+(2+k-1)*ndig);
    %pcolor(1:15,16:-1:1,reshape(Xdt(i,:), 15, 16)'); shading interp; 
    %set(gca,'xticklabel',[]);set(gca,'yticklabel',[]);           
    %if i==1, ylabel(['n=',num2str(nb_pcs)]); end
    %drawnow    
 end % for i
end % for k




% % denosing using Linear PCA for comparison
% % which number of eigenvalues of pca
% npcs = [2.^(0:7) 190];
% lpcs = length(npcs);
% 
% 
% figure(iii+1); colormap('gray');title('Denosing using linear PCA');
% 
% for k=1:lpcs;
%  nb_pcs=npcs(k); 
%  Ud=U_lin(:,(1:nb_pcs)); lamd=lam(1:nb_pcs);
%     
%  for i=1:ndig
%     dig=digs(i);
%     xt=Xnt(i,:);
%     proj_lin=xt*Ud; % projections of linear PCA
%     if k==1 
%         % plot the original clean digits
%         %
%         subplot(2+lpcs, ndig, i);
%         pcolor(1:15,16:-1:1,reshape(Xtest1(i,:), 15, 16)'); shading interp; 
%         set(gca,'xticklabel',[]);set(gca,'yticklabel',[]);                
%         if i==1, ylabel('original'), end  
%         
%         % plot the noisy digits 
%         %
%         subplot(2+lpcs, ndig, i+ndig); 
%         pcolor(1:15,16:-1:1,reshape(xt, 15, 16)'); shading interp; 
%         set(gca,'xticklabel',[]);set(gca,'yticklabel',[]);        
%         if i==1, ylabel('noisy'), end
%     end
%     Xdt_lin(i,:) = proj_lin*Ud';
%     r_lin(i) = norm(Xdt_lin(i,:) - xt);
%     subplot(2+lpcs, ndig, i+(2+k-1)*ndig);
%     pcolor(1:15,16:-1:1,reshape(Xdt_lin(i,:), 15, 16)'); shading interp; 
%     set(gca,'xticklabel',[]);set(gca,'yticklabel',[]);        
%     
%     if i==1, ylabel(['n=',num2str(nb_pcs)]), end
%  end % for i
% end % for k

aa=1; %for breakpoint
