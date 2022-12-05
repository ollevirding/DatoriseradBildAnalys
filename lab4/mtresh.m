function out = mtresh(im,t1,t2)
%
% function out = mtresh(im,t1,t2)
%
% Calculates multiple thresholdning and plots the result in gcf
%
% im , image
% t1, first treshold
% t2, second treshold
%
% Maria Axelsson, maria@cb.uu.se
  
  out = (im<t1).*1 + ((im>=t1)&(im<t2)).*2 + (im>=t2).*3;
  imagesc(out)
