function [Ds,Ts] = minN(D,nNbr)
% function [Ds,Ts] = minN(D,nNbr)
% INPUT:
%  D(iData,iDataLibrary): distance matrix
%  nNbr: number of nearest neighbors
% OUTPUT:
%  Ds(iData,iNbr): distance
%  Ts(iData,iNbr): sample ID

nT0 = size(D,1); Ds = zeros(nT0,nNbr); Ts = zeros(nT0,nNbr);
for iNbr = 1:nNbr
    [Ds(:,iNbr), Ts(:,iNbr)] = min(D,[],2);
    D((1:nT0)+(Ts(:,iNbr)'-1)*nT0) = Inf;
end