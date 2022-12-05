function im = class2im(data,sizeX,sizeY)
  
  % function im = testdata2im(data,sizeX,sizeY)
  %
  % Convert (reshape) the classification result to an image.
  % sizeX,sizeY is the spatial size of the original image.
  %
  % Maria Axelsson, maria@cb.uu.se
    
  im = reshape(data,[sizeX,sizeY]);
