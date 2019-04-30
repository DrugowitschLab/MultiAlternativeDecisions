function c = cumul(p, normalizationFlag)
% function c = cumul(p, normalizationFlag)
% p(iBin,:)
if nargin < 2||isempty(normalizationFlag);  normalizationFlag = true;  end
if normalizationFlag
    c = tril(ones(size(p,1))) * p ./ repmat(sum(p),[size(p,1) 1]);
else
    c = tril(ones(size(p,1))) * p;
end
