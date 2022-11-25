imfile = "asdff.png";

mike = imread("images/"+imfile);
mike = im2gray(mike);
disp(size(mike))

[heightI,widthI] = size(mike);
diff = abs(widthI-heightI);

if heightI > widthI
mike = [mike , 255*ones(heightI,diff)];
mike = imresize(mike, [128 128]);

elseif heightI == widthI
mike = imresize(mike, [128 128]);

else
mike = [mike ; 255*ones(diff,widthI)];
mike = imresize(mike, [128 128]);
end

figure()
imshow(mike)

%%
h = 5;

filter = single(zeros(128));

for i = 1:128
    for j = 1:128
        pwindow = mike(max(1,(i-2)):min(128,(i+2)),max(1,(j-2)):min(128,(j+2)));
        filter(i,j) = mean(mean(single(pwindow)))/255;
    end
end
%%
close all
figure
tiledlayout(1,3)
nexttile
imshow(filter)
nexttile
imshow(mike)
nexttile
imshow(mike - uint8(filter*255))

