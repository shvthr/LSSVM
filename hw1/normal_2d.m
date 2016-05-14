function [ px ] = normal_2d(x, mu )
%sigma is assumed to be [1 0; 0 1];
k=2; %2d

px = (2*pi)^(-k/2) * exp(-0.5*(x-mu)'*(x-mu));


end

