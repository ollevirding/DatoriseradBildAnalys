function [data,class] = create_training_data(im,lab)

  % function [data,class] = create_training_data(im,lab) 
  %
  % Create training data from labeled image, lab
  % The image can be multispectral with several bands, the first
  % two dimensions are the image dimensions correspondning to lab.
  % The output is data points and class for each data point.
  % 
  % Maria Axelsson, maria@cb.uu.se
  
  m = max(max(lab));
  
  data = [];
  class = [];
  
  for n = 1:size(im,1)
    for m = 1:size(im,2)
      if lab(n,m)~=0
        data = [data; squeeze(im(n,m,:))'];
        class = [class; lab(n,m)];
      end
    end
  end
  
  
  %1-band
  %for n=1:m
  %  tmp = im(find(lab==n))';
  %  data = [data tmp];
  %  class = [class repmat(n,[1 length(tmp)])];
  %end
  
