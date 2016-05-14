function [pfinal,efinal] = dsp(pn,herrfunc)

%
% Internal function based on DFO-based randomized Directional Search (runnable in several processes)
%
% Copyright (c) 2013,  KULeuven-ESAT-SCD, License & help @% http://www.esat.kuleuven.be/sista/lssvmlab

% matlabpool open 4

parfor k=1:size(pn,2)
    [par(:,k), fval(k)] = ds(pn(:,k),herrfunc);
end

% matlabpool close

efinal = min(fval);
pfinal = par(:,fval == efinal);

end