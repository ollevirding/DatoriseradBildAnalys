%%  ASSIGNMENT 2
x = 1;
load landsat_data.mat
close all
%figure(1);
%tiledlayout(4,5)
v = 1:7;
C = nchoosek(v, uint8(3));
if x == 1
for i=2:length(C)
    %nexttile
    figure()
    imshow(landsat_data(:,:,C(i,:))./255);
    title(num2str(C(i,:)))
end
end
%%
imagemike = landsat_data(:,:,[4, 6, 7])./255;
imtool(landsat_data(:,:,[4, 6, 7])./255)

%%
labelarea = zeros(512);

labelarea(17:70,482:512) = 1; % WATUH
labelarea(44:150,143:232) = 2; %City
labelarea(138:242,395:511) = 3; %Forrest Gump
labelarea(325:418,292:315) = 4; %Åker

[data, class] = create_training_data(imagemike, labelarea);
Itest = im2testdata(imagemike);
C = classify(double(Itest), double(data), double(class));
ImC = class2im(C, size(imagemike, 1), size(imagemike, 2));
imagesc(ImC)
%[data1,class1] = create_training_data(IRGB,label_im);  % Arrange the training data into vectors
%Itest1 = im2testdata(IRGB); % Reshape the image before classification
%C = classify(double(Itest1),double(data1),double(class1)); % Train classifier and classify the data
%ImC = class2im(C,size(IRGB,1),size(IRGB,2)); % Reshape the classification to an image
%imagesc(ImC); % View the classification result
%title('Classification of image using Channels: RGB')