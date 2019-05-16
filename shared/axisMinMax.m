function a = axisMinMax(x, margin)
% function a = axisMinMax(x, [margin=0.05]);

if nargin < 2;  margin = 0.05;  end;
if length(margin)==1; margin = margin * ones(1,2); end;

ind = isfinite(x(:));
x = x(:);
a = [min(x(ind))-margin(1)*range(x(ind)),  max(x(ind))+margin(2)*range(x(ind))];
if a(1)==a(2); a(2) = a(1) + 0.00001; end;