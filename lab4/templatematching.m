%
% Template matching function based on correlation
% 
% ccimg = templatematching(target, template, step)
%
% INPUT
% target    - the image where the template is to be matched with
% template  - the template moved over the target image
% step      - the step size to be used when moving the template
%
% OUTPUT
% ccimg     - the correlation coefficient
%
% The function uses 'progressbar.m' to show the progress.
%
% /Gustaf Kylberg 2008 
%  gustaf@cb.uu.se
%

function ccimg = templatematching(target, template, step)

progressbar('Template matching...'); % opens progressbar

ccrow = 1;
cccol = 1;
rows = 1:step:(size(target,1)-size(template,1));
cols = 1:step:(size(target,2)-size(template,2));
ccimg = zeros(length(rows),length(cols));

for row = 1:step:(size(target,1)-size(template,1))
    for col = 1:step:(size(target,2)-size(template,2))
        
        %calculating the correlation coefficient
        ccimg(ccrow,cccol) = corr2(template,target(row:row+size(template,1)-1,col:col+size(template,2)-1));       
        cccol = cccol + 1;
    end
    increment=(row/(size(target,1)-size(template,1)));
    progressbar(increment);
   
    ccrow = ccrow + 1;
    cccol = 1;
end
progressbar(1); % close progressbar