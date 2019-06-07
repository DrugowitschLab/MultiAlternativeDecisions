function C = myconvn(A, B, shape)
% function C = myconvn(A, B, shape)
if nargin < 3 || isempty(shape);  shape = 'same';  end

C = convn(A, B, shape) ./ convn(ones(size(A)), B, shape);
