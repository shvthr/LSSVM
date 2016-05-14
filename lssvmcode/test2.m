for i=1:10
%make two clusters
X1=(1-i/6)+randn(50,2);
X2=-(1-i/6)+randn(51,2);
Y1=ones(50,1);
Y2=-ones(51,1);
X=[X1;X2];
Y=[Y1;Y2];
%a test data point
Xtest=[0 0];
Ytest=lda(Xtest,X,Y);
%visualize
plot(X1(:,1),X1(:,2),'ro');hold on;
plot(X2(:,1),X2(:,2),'bo');
if Ytest>0,
plot(Xtest(:,1),Xtest(:,2),'r*','markersize',20);
else
plot(Xtest(:,1),Xtest(:,2),'b*','markersize',20);
end
hold off
pause
end