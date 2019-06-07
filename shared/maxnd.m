function [Xmax, idx] = maxnd(X)
[Xmax, idxfull] = max(X(:));
sz = size(X);
n = length(sz);
str = '[';
for idim = 1:n
    str = [str 'idx(' num2str(idim) ')'];
    if idim == n
        str = [str ']'];
    else
        str = [str ', '];
    end
end     
% str: '[idx(1), idx(2), ..., idx(n)]'
eval([str ' = ind2sub(sz, idxfull);']);
