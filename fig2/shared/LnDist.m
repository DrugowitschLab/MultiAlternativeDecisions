function D = LnDist(x,y,n)
% function D = LnDist(x,y,[n=2])
% IUNPUT:
%  x(iDim,iSample)
%  y(iDim,iSample)
% OUTPUT:
%  D(iSample,iSample)

if nargin<3;  n = 2;  end
switch n
    case 1
        D = shiftdim(sum(abs( repmat(x,[1 1 size(y,2)]) - permute(repmat(y,[1 1 size(x,2)]),[1 3 2]) ),1),1);
    case 2
        D = (bsxfun(@plus, dot(x,x,1)', dot(y,y,1))-2*(x'*y));
end