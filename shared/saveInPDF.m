function saveInPDF(f, fname, paperOrient)
% function saveInPDF([f=gcf()], [fname='figure'], [paperOrient='rotated'])
if nargin < 1||isempty(f);      f = gcf();  end
if nargin < 2||isempty(fname);  fname = 'figure';  end
if nargin < 3||isempty(paperOrient);  paperOrient = 'landscape';  end

set(groot,'defaultAxesFontName',    'Arial');
set(groot,'defaultTextFontName',    'Arial');
set(groot,'defaultLegendFontName',  'Arial');
set(groot,'defaultColorbarFontName','Arial');
set(groot,'defaultColorbarFontSize', 8);
set(f,'PaperUnits','centimeters');
set(f,'PaperSize',0.03.*f.Position([4 3]));

orient(paperOrient);

print(f,'-painters',fname,'-dpdf')  % PDF

fprintf(['\nSaved PDF:' fname '\n']);