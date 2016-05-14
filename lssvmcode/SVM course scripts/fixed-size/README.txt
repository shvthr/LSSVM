NysFS-LSSVM is a simple and easy-to-use software for large scale FS-LSSVM regression and two-class classification. This document explains the use of FS-LSSVMv1.0.

We have a script named fslssvm_script.m which is used to run the variants of FS-LSSVM techniques.

Input format:
In the script the end-user has to determine X component or the feature set and Y component or the predictor value/estimate.
The NysFS-LSSVM requires the LSSVMlab toolbox 
Example: You will see "addpath('../LSSVMlab')" in some .m files of NysFS-LSSVM toolbox.

Input set variables to be set are:
1) k -> Used for determining the initial fixed size or number of prototype vectors (PV) according to the formula (PV = sqrt(N)*k where N is the total number of points).
   
   %Example k = 6 (for 'boston housing' or 'breast cancer' datasets) then
   %Prototype vector size = 6*32 (i.e. 6*sqrt(1024)) = 192

2) function_type -> 'c' for classification or 'f' for regression/function estimation.

3) kernel_type -> can be 'RBF_kernel', 'lin_kernel' or 'poly_kernel'

4) global_opt -> can be 'csa' (Coupled Simulated Annealing) or 'ds' (Directional Search)

5) user_process -> A cell of various variations to FS-LSSVM can be provided 
       Example user_process-> {FS-LSSVM,'SV_L0_norm','ALL_L0_norm','L0_norm','WINDOW','LSSVMwin','LSSVMwinL'}
			      FS-LSSVM -> Normal FS-LSSVM approach
			      SV_L0_norm -> FS-LSSVM performed first then L0_norm performed training only on the set of PV
		              ALL_L0_norm -> FS-LSSVM performed first then L0_norm performed using entire training set in primal
			      L0_norm -> LSSVM performed (using only PV for cross-validation to determine sig, gam resulting in much much faster approach) and then L0_norm using entire training set in the primal
			      WINDOW -> FS-LSSVM performed and the appropriate prototype vector selected (SV) using WINDOW based method and then FS-LSSVM re-performed.
			      LSSVMwin -> LSSVM performed (using only PV for cross-validation) and then appropriate prototype vectore selected (SV) using WINDOW based method and FS-LSSVM re-perfomed.
			      LSSVMwinL -> LSSVM performed (using only PV for cross-validation) and then appropriate prototype vectore selected (SV) using WINDOW based method and LSSVM re-perfomed (but using entire training data).

Note: Whenever comparing the WINDOW, LSSVMwin, LSSVMwinL always maintain the order (i.e. WINDOW before LSSVMwin and LSSVMwinL, LSSVMwin before LSSVMwinL) and always include FS-LSSVM in the user_process.

6) window -> A list representing percentage of initial prototype vector to select as appropriate PV 
      Example -> window = [10,15,20]

Sample instruction to run:
-------------------------
[e,s,t] = modsparseoperations_initial(X,Y,k,function_type,kernel_type,user_process,window);

where e = Error Row (contains error results for 10 randomizations)
      s = Support Vector Row (contains support vector results for 10 randomizations)
      t = Time Row (contains time results for 10 randomizations)

 
Note: There will be some warning that will appear like Matrix close to singular precision (You can ignore them).

6) The functionality to provide your own seperate test input and test output
has been included. You can also provide the input test data and obtain the
predictions for the different algorithms

Example:
testX = []; testY = [];
[e,s,t] = modsparseoperations_initial(X,Y,k,function_type,kernel_type,user_process,window,testX,testY);

An example is given in the script

Copyright (c) 2012, Raghvendra Mall, ESAT/SISTA, K.U.Leuven.     
