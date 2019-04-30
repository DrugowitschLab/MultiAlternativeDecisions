function val = getAxRel(rate,xyz,h)
% function val = getAxRel([rate=-0.5],[xyz='x'],[h=gca])

if nargin < 1 || isempty(rate);  rate = -0.5;  end
if nargin < 2 || isempty(xyz);   xyz  = 'x';   end
if nargin < 3 || isempty(h);     h    = gca;   end

switch lower(xyz)
    case 'x'
        mm = xlim(h);
    case 'y'
        mm = ylim(h);
    case 'z'
        mm = zlim(h);
    otherwise
        error('ERROR in funciton getAxRel: second variable must be ''x'', ''y'', or ''z''.');
end

val = rate * (mm(2) - mm(1)) + mm(1);

