function a = finAvg(X,dim)
% function a = finAvg(X,[dim=1])
if nargin < 2;  dim = 1;  end

Xfin = isfinite(X);
X(~Xfin) = 0;
a = sum(X,dim)./sum(Xfin, dim);