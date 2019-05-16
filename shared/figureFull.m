function fig = figureFull(ys,xs,y,x)
% function figureFull(ys,xs,y,x);
% Make figure window at posision and in size of (y,x) of the screen as ys-by-xs matrix.
% by Satohiro Tajima

if nargin < 1; ys=1; end;
if nargin < 2; xs=1; end;
if nargin < 3; y=1:ys; x=1:xs; end;
if nargin == 3; x = 1+mod(y-1,xs); y = ceil(y/xs); end;

posRx = (x(1)-1) / xs;
posRy = (y(1)-1) / ys;
sizeRx = (x(end)-x(1)+1) / xs;
sizeRy = (y(end)-y(1)+1) / ys;

scrsz = get(0,'ScreenSize') + [0 50 0 -112];
posx = scrsz(3)*posRx;
posy = scrsz(4)*posRy;
sizex = scrsz(3)*sizeRx;
sizey = scrsz(4)*sizeRy;

scrsz = round([scrsz(1)+posx scrsz(2)+scrsz(4)-sizey-posy sizex sizey]);

fig = figure('Position', scrsz);

