%digitDatasetPath = fullfile(matlabroot,'toolbox','nnet', ...
%    'nndemos','nndatasets','DigitDataset');
%imds = imageDatastore(digitDatasetPath, ...
%    'IncludeSubfolders',true, ...
%    'LabelSource','foldernames');

%%
labels = string(readmatrix("labels.txt")); % read the labels from a file
Y = string(zeros(1200*3, 1));
for i=1:length(Y)
    Y(i) = labels(i);
end
Y = categorical(Y);
%%
% Loop through the images and resize them
close all
X = cell(3600,1);
for i =1:1200
    filename = "train_" + sprintf('%04d',i) + ".png";
    [im1, im2, im3] = splitimage(filename);
    X{i + (2*(i-1))} = im1; % append the image to the X array
    X{i + 1 + (2*(i-1))} = im2; % append the image to the X array5
    X{i + 2 + (2*(i-1))} = im3; % append the image to the X array
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

numTrainingFiles = 1000;
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