function col = sat(r, colEd, colSt)
% function col = sat(r, colEd, colSt)

if ~exist('colEd','var'); colEd = [0 0 0]; end
if ~exist('colSt','var'); colSt = [.9 .9 .9]; end

for iR = 1:length(r)
    col(iR,:) = r(iR) * colEd + (1-r(iR)) * colSt;
end