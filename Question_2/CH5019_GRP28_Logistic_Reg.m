 % Getting data from dataset 
 T = readtable('CH5019_GRP28_Dataset_Question2.xlsx');
 
 % Assigning 1 for 'Pass' and 0 for 'Fail' 
 T.test1 = strcmp(T.Test,'Pass');
 
 [m,n] = size(T) ;
 
 %Partitioning dataset into training and test set
 P = 0.70 ;
 idx = randperm(m);
 Traindata = T(idx(1:P*m),:); 
 Testdata = T(idx(P*m+1:end),:);
 
 
 X1 = Traindata(:,1:5);
 X = table2array(X1);
 y = table2array(Traindata(:,7));
 
 % Mean Normalaisation
 mu = mean(X);
 X2 = X-mu;
 Xnorm = [X2(:,1)./300 X2(:,2)./49 X2(:,3)./150 X2(:,4)./2600 X2(:,5)./0.4];
 
 
 itr1=1;
 itr2=1;
 itr3=1;
 itr4=1;
 
% Gradient Descent

 Xnorm = [ones(700,1) Xnorm];
 
 % Initialization 1
 theta1 = zeros(6,1);
 alpha = 0.01;
 [cost1, grad1] = CH5019_GRP28_Costfunction(theta1, Xnorm, y);  
 
 while(sqrt(sum(grad1.^2)) > 10^-6) 
  theta1 = theta1-alpha*grad1; 
  [cost1, grad1] = CH5019_GRP28_Costfunction(theta1, Xnorm, y);
  J1(itr1)=cost1;
  itr1 = itr1+1;
 end
 
 %plot(J1);
 mincost = cost1;
 besttheta = theta1;
 
 
 % Initialization 2
 theta2 = [-45;60;70;80;-60;80];
 alpha = 0.01;
 [cost2, grad2] = CH5019_GRP28_Costfunction(theta2, Xnorm, y); 
 
 while(sqrt(sum(grad2.^2)) > 10^-6) 
  theta2 = theta2-alpha*grad2; 
  [cost2, grad2] = CH5019_GRP28_Costfunction(theta2, Xnorm, y);
  J2(itr2)=cost2;
  itr2 = itr2+1;
 end
 
 %plot(J2);
 if(cost2 < mincost)
     mincost = cost2;
     besttheta = theta2;
 end
 
 
 % Initialization 3
 theta3 = [105;-150;-200;-250;150;-200];
 alpha = 0.01;
 [cost3, grad3] = CH5019_GRP28_Costfunction(theta3, Xnorm, y);  
 
 while(sqrt(sum(grad3.^2)) > 10^-6) 
  theta3 = theta3-alpha*grad3; 
  [cost3, grad3] = CH5019_GRP28_Costfunction(theta3, Xnorm, y);
  J3(itr3)=cost3;
  itr3 = itr3+1;
 end
 
 %plot(J3);
  if(cost3 < mincost)
     mincost = cost3;
     besttheta = theta3;
  end
 
 
 % Initialization 4
 theta4 = [-100;100;150;200;-100;80];
 alpha = 0.01;
 [cost4, grad4] = CH5019_GRP28_Costfunction(theta4, Xnorm, y);  
 
 while(sqrt(sum(grad4.^2)) > 10^-6) 
  theta4 = theta4-alpha*grad4; 
  [cost4, grad4] = CH5019_GRP28_Costfunction(theta4, Xnorm, y);
  J4(itr4)=cost4;
  itr4 = itr4+1;
 end
 
 %plot(J4);
 if(cost4 < mincost)
     mincost = cost4;
     besttheta = theta4;
 end

 % Verifying the obtained theta in the test set
 Xtest1 = Testdata(:,1:5);
 Xtest = table2array(Xtest1);
 ytest = table2array(Testdata(:,7));
 Xtest2 = Xtest-mu;
 
 theta = besttheta;

 Xtest3 = [Xtest2(:,1)./300 Xtest2(:,2)./49 Xtest2(:,3)./150 Xtest2(:,4)./2600 Xtest2(:,5)./0.4];
 Xtest3 = [ones(300,1) Xtest3];
 YU = Xtest3*theta;
 yob = (CH5019_GRP28_sigmoid(Xtest3*theta) >= 0.5);

 % Confusion Matrix
 confusion_matrix = [sum(yob==1 & yob == ytest) sum(yob==0 & yob ~= ytest); sum(yob==1 & yob ~= ytest) sum(yob==0 & yob == ytest);];
 disp('Confusion matrix:')
 disp(confusion_matrix)
 
 % Computing F1 Score
 TP = sum(yob==1 & yob == ytest);
 FP = sum(yob==1 & yob ~= ytest);
 FN = sum(yob==0 & yob ~= ytest);
 TN = sum(yob==0 & yob == ytest);
 p = TP/(TP+FP);
 r = TP/(TP+FN);
 F1 = 2*p*r/(p+r);
 disp('F1 score:')
 disp(F1)
