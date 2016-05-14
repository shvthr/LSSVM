function [pfinal,efinal] = ds(pn,herrfunc)

%
% Internal function based on DFO-based randomized Directional Search
%
% Copyright (c) 2013,  KULeuven-ESAT-SCD, License & help @% http://www.esat.kuleuven.be/sista/lssvmlab

x(:,1) = pn;
alpha(1) = 1;
pdim = length(pn);
ft(1) = feval(herrfunc,pn);
D = [eye(pdim) -eye(pdim)];

for k=1:1000
   % restart polling directions
   dk = crossvalind('Kfold', 2*pdim, 2*pdim);
    
   for i=1:2*pdim
      curr_i = dk(i);
      f_temp = feval(herrfunc, x(:,k) + alpha(k).*D(:,curr_i));
      if (f_temp < ft(k) - 10^-5*(alpha(k)^2))
        x(:,k+1) = x(:,k) + alpha(k).*D(:,curr_i);
        ft(k+1) = f_temp;
        break;
      else
        x(:,k+1) = x(:,k);
        ft(k+1) = ft(k);
      end
   end
   
   % exit on epsilon-based criterion 
   if (norm(ft(k) - ft(k+1)) < 10^-5)
       break;
   end
   
   % decrease step size if needed
   if (ft(k+1) >= ft(k) - 10^-5*(alpha(k)^2))
      alpha(k+1) = 0.5 * alpha(k); 
   else
      alpha(k+1) = alpha(k); 
   end
end

pfinal = x(:,end);
efinal = ft(end);

end