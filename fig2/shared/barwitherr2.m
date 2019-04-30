function [hBar, hErr] = barwitherr2(x,y,e,u,col)
% function [hBar, hErr] = barwitherr2(x,y,e(l),u,col)
if nargin < 4 || isempty(u);  u = e;  end
if nargin < 5;  col = [0 0 0];  end
hBar = bar(x,y,'FaceColor',col,'LineStyle','none');
hErr = errorbar(x,y,e,u,'.','Color',sat(.5,col),'Marker','none');end