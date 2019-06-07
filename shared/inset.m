function [hInset, hMain] = inset(hMain, posInset)
% function [hInset, hMain] = inset([hMain=gca], [posInset=[0.2 0.6 0.2 0.3]])
% 
% EXAMPLE:
%  figure;
%  subplot(1,2,2);
%  plot(1:10, cos(1:10));
%  hInset = inset(gca, [0.2 0.6 0.2 0.3])
%  plot(1:10, sin(1:10));

if nargin < 1 || isempty(hMain);     hMain = gca;  end
if nargin < 2 || isempty(posInset);  posInset = [0.2 0.6 0.2 0.3];  end

pos = hMain.Position;
posInsetNew = [ posInset(1) * pos(3) + pos(1) ...
                posInset(2) * pos(4) + pos(2) ...
                posInset(3) * pos(3) ...
                posInset(4) * pos(4)];
hInset = axes('position',posInsetNew);