function scatterplot3D(data,group)

 % function scatterplot3D(data,group)
 %
 % data is a feature vectors aligned in rows  
 % group is the corresponding class for each vector
 % (group can also be a cell array aligned as a column with strings)
 %
 % Maria Axelsson, maria@cb.uu.se

 color = {'.r','vb','gs','m.','vc','ks','y.','vr','sb','.g','vc','sk'};
 %% color = {'.r','.b','g.','m.','.c','k.','y.','sr','sb','sg','sm','sc','sk'};
 
 figure(gcf);
 hold off
 [G,Gname] = grp2idx(group);
 for n=1:max(G)
   tmp = data(find(G==n),:);
   plot3(tmp(:,1),tmp(:,2),tmp(:,3),color{n});
   if n == 1
     hold on
   end
 end

 xlabel('X')
 ylabel('Y')
 zlabel('Z')
 
 title('3D scatterplot')
 
 legend(Gname)
 
 
 
