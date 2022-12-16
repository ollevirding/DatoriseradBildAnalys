%digitDatasetPath = fullfile(matlabroot,'toolbox','nnet', ...
%    'nndemos','nndatasets','DigitDataset');
%imds = imageDatastore(digitDatasetPath, ...
%    'IncludeSubfolders',true, ...
%    'LabelSource','foldernames');

%%
labels = string(readmatrix("labels.txt")); % read the labels from a file
Y = string(zeros(length(labels),1));
for i=1:length(Y)
    Y(i) = strjoin(labels(i,1),',');
end
Y = categorical(Y);
%%
% Loop through the images and resize them
close all
kmax = 0;
for i =1:1200
    filename = "train_" + sprintf('%04d',i) + ".png";
    im = imread(filename);
    im = imbinarize(im,graythresh(im));
    im = imcomplement(im);

    % Erode the image to separate the numbers (Didn't work)
    se = strel('square', 2);
    im = imerode(im, se);

    im = medfilt2(im, [5, 5]);

    im = imcrop(im, [70 60 150 100]);
    im = imresize(im, [56, 56]); % resize the image
    im = uint8(255*im);
    im2 = imgaussfilt(im, 2,"FilterSize",[3, 3]);
    im = imerode(im, strel('square',2));
    im = im + im2;

    %imshow(im)
    X{i} = im; % append the image to the X array
end
%%
ImageFolder ='ProcessedImages/';

for i=1:length(X)
    file_name = sprintf('Image%d.png', i);% name Image with a sequence of number, ex Image1.png , Image2.png....
    fullFileName = fullfile(ImageFolder, file_name);
    imwrite(X{i},file_name,'png') %save the image as a Portable Graphics Format file(png)into the MatLab
    imgName = [ImageFolder,'\Image_',num2str(i),'.png'] ;
    imwrite(X{i},imgName) ; 
end

%%

imds = imageDatastore('ProcessedImages/', 'Labels', Y);


%%

numTrainingFiles = 300;
[imdsTrain,imdsTest] = splitEachLabel(imds,numTrainingFiles,'randomize');

%%

layers = [ ...
    imageInputLayer([56 56 1])
    convolution2dLayer(5,20)
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    fullyConnectedLayer(3)
    softmaxLayer
    classificationLayer];

%%

options = trainingOptions('sgdm', ...
    'MaxEpochs',60,...
    'InitialLearnRate',1e-4, ...
    'Verbose',false, ...
    'Plots','training-progress');

%%

net = trainNetwork(imdsTrain,layers,options);