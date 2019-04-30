function cmap = gradation(cols, n)
% function gradation(cols, [n=100])
% cols = {[r1 g1 b1],[r3 g2 b2],...}
if nargin<1;  cols = {[0 1 0],[0 0 0],[1 0 1]};  end
if nargin<2;  n = 100;  end

nCol = length(cols);
cmapH = zeros(n*(nCol-1),3);
for iCol = 1:nCol-1
   cmapH((iCol-1)*n+(1:n),:) = sat(linspace(0,1,n), cols{iCol+1}, cols{iCol}); 
end
for iC = 3:-1:1
    cmap(:,iC) = interp1(1:size(cmapH,1), cmapH(:,iC), linspace(1,size(cmapH,1),n));
end