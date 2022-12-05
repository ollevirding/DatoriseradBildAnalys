function data = im2testdata(im)
  
  % function data = im2testdata(im)
  %
  % Convert (reshape) an image to test data used for classification
  %
  % Maria Axelsson, maria@cb.uu.se
    
 data = [];   
    
 for n = 1:size(im,3)
   tmp = im(:,:,n);
   data(:,n) = tmp(:);
 end
