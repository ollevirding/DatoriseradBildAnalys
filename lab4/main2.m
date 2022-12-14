%%  ASSIGNMENT 2
x = 1;
load landsat_data.mat
close all
%figure(1);
%tiledlayout(4,5)
%v = [5, 7];
%Comb = nchoosek(v, uint8(2));
Comb = [6, 5, 7];

%%

imshow(landsat_data(:,:,[3, 2, 1])./255)

%%

figure()
imshow(landsat_data(:,:,Comb)./255)

%%

imagemike = landsat_data(:,:,Comb)./255;
%title(num2str(Comb(i,:)))

labelarea = zeros(512);

labelarea(17:70,482:512) = 1; % WATUH
%labelarea(479: 485, 110: 167) = 1;
%labelarea(44:150,143:232) = 2; %City
labelarea(105:130,208:236) = 2;
labelarea(80:92,177:188) = 2;
labelarea(311:315,236:243) = 4; %orange-åker
labelarea(315:321,242:248) = 4; %rod-åker
labelarea(308:311,231:236) = 4; %orange-åker
labelarea(122:130,470:485) = 3; %Forrest Gump
labelarea(178:201,388:407) = 3; %Forrest Gump
%labelarea(138:242,395:511) = 3; %Forrest Gump
%labelarea(325:418,292:315) = 4; %Åker

%%

if x == 1
for i=2:length(Comb)
    %nexttile
    %imagemike = landsat_data(:,:,[Comb(i,1), Comb(i,2), 6])./255;
    %title(num2str(Comb(i,:)))
%%
%figure()
%imshow(landsat_data(:,:,[3,2,1])./255)

%%
%imagemike = landsat_data(:,:,[4, 6, 7])./255;
%imtool(landsat_data(:,:,[3, 4, 7])./255)

%%

figure(23)
[data, class] = create_training_data(imagemike, labelarea);
Itest = im2testdata(imagemike);
C = classify(double(Itest), double(data), double(class), "quadratic");
ImC = class2im(C, size(imagemike, 1), size(imagemike, 2));
imagesc(ImC)
title(num2str(Comb))

end
end
%[data1,class1] = create_training_data(IRGB,label_im);  % Arrange the training data into vectors
%Itest1 = im2testdata(IRGB); % Reshape the image before classification
%C = classify(double(Itest1),double(data1),double(class1)); % Train classifier and classify the data
%ImC = class2im(C,size(IRGB,1),size(IRGB,2)); % Reshape the classification to an image
%imagesc(ImC); % View the classification result
%title('Classification of image using Channels: RGB')