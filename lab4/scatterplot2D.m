function scatterplot2D(data,group)

 % function scatterplot2D(data,group)
 %
 % data is a feature vectors aligned in rows  
 % group is the corresponding class for each vector
 % (group can also be a cell array aligned as a column with
 % strings)
 %  support for 12 classes.
 %
 % Maria Axelsson, maria@cb.uu.se
   
 color = {'.r','vb','gs','m.','vc','ks','y.','vr','sb','.g','vc','sk'};
   
 figure(gcf);
 hold off
 [G,Gname] = grp2idx(group);
 for n=1:max(G)
   tmp = data(find(G==n),:);
   plot(tmp(:,1),tmp(:,2),color{n});
   if n == 1
     hold on
   end
 end

 xlabel('X')
 ylabel('Y')
 
 title('2D scatterplot')
 
 legend(Gname)
 
