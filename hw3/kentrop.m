function [subset, r2] = kentrop(X, sig2)
    ssize = 3;
    subset = zeros(ssize,2);
    r2=[];
    for t = 1:100,

      %
      % new candidate subset
      %
      r = ceil(rand*ssize);
      candidate = [subset([1:r-1 r+1:end],:); X(t,:)];

      %
      % is this candidate better than the previous?
      %
      if kentropy(candidate, 'RBF_kernel',sig2)>...
            kentropy(subset, 'RBF_kernel',sig2),
        subset = candidate;
        r2 = cat(1,r2,r);
      end

      %
      % make a figure
      %
      
    end