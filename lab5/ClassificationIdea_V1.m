% SPLITING IMAGES USING COMPUTER VISION TOOLBOX
% Sorting out bad images
%%
clear all
GTdata = {};
GTbbox = {};
GTlabels = [];
labels = readmatrix("labels.txt");
index = 0;

Folder = "SegmentingImages/";

if ~exist(Folder, 'dir')
    mkdir(Folder)
end

%%
for i = 1:1200
name = "train_" + sprintf('%04d',i) + ".png";

img = imread("imagedata/" + name);
imshow(img)
pause(2)
img = imclose(img, strel('sphere', 1) );
imshow(img)
pause(2)
bw_img = imbinarize(img,"global");


im = imcomplement(bw_img);
imshow(im)
pause(2)
im = imerode(im, strel('square',3));
imshow(im)
pause(2)



% Get the bounding boxes of the individual nrs
stats = regionprops(im, 'BoundingBox');

    if length(stats) == 3
        % Crop the first nr
        nr1 = imcrop(im, stats(1).BoundingBox);
        % Crop the second nr
        nr2 = imcrop(im, stats(2).BoundingBox);
        % Crop the third nr
        nr3 = imcrop(im, stats(3).BoundingBox);

        % Pad images and resize to same size
        nr1 = padimage(nr1);
        nr2 = padimage(nr2);
        nr3 = padimage(nr3);    
        
        index = index + 1;
        
        GTdata{end + 1} = nr1;
        GTlabels{end + 1} = labels(i,1);
        GTbbox{end + 1} = regionprops(nr1, 'BoundingBox').BoundingBox;
        writeimage(nr1, index)

        index = index + 1;

        GTdata{end + 1} = nr2;
        GTlabels{end + 1} = labels(i,2);
        GTbbox{end + 1} = regionprops(nr2, 'BoundingBox').BoundingBox;
        writeimage(nr2, index)

        index = index + 1;

        GTdata{end + 1} = nr3;
        GTlabels{end + 1} = labels(i,3);
        GTbbox{end + 1} = regionprops(nr1, 'BoundingBox').BoundingBox;
        writeimage(nr3, index)
        %pause(1)
        %tiledlayout(1,3)
        %nexttile
        %imshow(nr1)
        %title(labels(i,1))
        %nexttile
        %imshow(nr2)
        %title(labels(i,2))
        %nexttile
        %imshow(nr3)
        %title(labels(i,3))
        

    end
% Save the individual nrs as separate images
end

%%
% Create imageDatastore

GTlabels = categorical(cell2mat(GTlabels));

imds = imageDatastore("SegmentingImages/", 'Labels',GTlabels);

% På något sätt här vill jag också få in GTBoundingBox. Det finns något som
% heter "boxLabelDatastore" men vet ej hur dessa används

%%
%Layers:

numTrainingFiles = 100;
[imdsTrain,imdsTest] = splitEachLabel(imds,numTrainingFiles,'randomize');


inputLayer = imageInputLayer([28, 28, 1], 'Name', 'input', 'Normalization','none');
filterSize = [3, 3];

middleLayer = [...
    
    convolution2dLayer(filterSize, 4, 'Padding',1, 'Name', ' Conv_1', ...
    'WeightsInitializer','narrow-normal')
    batchNormalizationLayer('Name', 'BN1')
    reluLayer("Name",'relu_1')
    maxPooling2dLayer(2, 'Stride', 2,'Name','MaxPool_1')

    convolution2dLayer(filterSize, 8, 'Padding',1, 'Name', ' Conv_2', ...
    'WeightsInitializer','narrow-normal')
    batchNormalizationLayer('Name', 'BN2')
    reluLayer("Name",'relu_2')
    maxPooling2dLayer(2, 'Stride', 2,'Name','MaxPool_2')

    convolution2dLayer(filterSize, 16, 'Padding',1, 'Name', ' Conv_3', ...
    'WeightsInitializer','narrow-normal')
    batchNormalizationLayer('Name', 'BN3')
    reluLayer("Name",'relu_3')
    maxPooling2dLayer(2, 'Stride', 2,'Name','MaxPool_3')

    convolution2dLayer(filterSize, 32, 'Padding',1, 'Name', ' Conv_4', ...
    'WeightsInitializer','narrow-normal')
    batchNormalizationLayer('Name', 'BN4')
    reluLayer("Name",'relu_4')
    fullyConnectedLayer(10)
    fullyConnectedLayer(3)
    softmaxLayer
    classificationLayer
    ];

lgraph = layerGraph([inputLayer; middleLayer]);


%%
% Specify the training options
options = trainingOptions('adam', ...
    'ExecutionEnvironment','gpu',...    % Ändra 'gpu' till 'cpu'
    'MaxEpochs',10,...
    'InitialLearnRate',1e-3, ...
    'ValidationData',imdsTest, ...
    'ValidationFrequency',10,...
    'Verbose',false, ...
    'Plots','training-progress'); 

% Man kan träna utan att plotta, på så 
% sätt kan vi göra detta "bakom kulliserna"

NET = trainNetwork(imdsTrain, lgraph, options);

%% Functions

function paddedimage = padimage(image)
        % Get the size of the image
        [height, width, ~] = size(image);
    
        % Calculate the amount of padding needed on each side
        pad_top = max(0, floor((max(height, width) - height) / 2));
        pad_bottom = max(0, ceil((max(height, width) - height) / 2));
        pad_left = max(0, floor((max(height, width) - width) / 2));
        pad_right = max(0, ceil((max(height, width) - width) / 2));
    
        % Pad the image using the calculated padding values
        padded_image = padarray(image, [pad_top pad_left], 0, 'pre');
        padded_image = padarray(padded_image, [pad_bottom pad_right], 0, 'post');
        paddedimage = imresize(padded_image, [28 28]);
end

function writeimage(im1, indexnr)
        ImageFolder = "SegmentingImages/";
        file_name = "Image"+num2str(indexnr,'%04.f'); % name Image with a sequence of number, ex Image1.png , Image2.png....
        fullFileName = fullfile(ImageFolder, file_name);
        imgName = strjoin([ImageFolder,'Image_',num2str(indexnr,'%04.f'),'.png']);
        imwrite(im1,imgName);
end
