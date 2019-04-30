function h = moveObject(h, pos, ratio)
% function h = moveObject(h, [pos=0], [r=.5])

if nargin < 2 || isempty(pos);    pos   = 0;  end
if nargin < 3 || isempty(ratio);  ratio = 1;  end

hpos = get(h,'position');

hpos(1:2) = hpos(1:2) + pos .* hpos(3:4) + (1-ratio)/2 .* hpos(3:4);
hpos(3:4) = ratio .* hpos(3:4);

set(h,'Position',hpos);
