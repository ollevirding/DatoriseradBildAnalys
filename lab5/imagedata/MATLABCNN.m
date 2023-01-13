
%%
labels = string(readmatrix("labels.txt")); % read the labels from a file
Y = reshape(labels.',1,[]);
Y = categorical(Y);

%%
% Loop through the images and resize them
close all
X = cell(3600,1);
for i =1:1200
    filename = "train_" + sprintf('%04d',i) + ".png";
    
    img = imread("imagedata/" + filename);
    img = medfilt2(img, [5, 5]);
    bw_img = imbinarize(img,"global");

    im = imcomplement(bw_img);
    im = bwareaopen(im,20);
    
    stats = regionprops(im, 'BoundingBox');

    if length(stats) == 3
    %    im = splitimage2(im);
    %    stats = regionprops(im, 'BoundingBox');
        %if length(stats) ~= 3
         %   imshow(im)
         %   i
         %   pause(2)

        %end
    %end
    % Crop the first nr
    nr1 = imcrop(im, stats(1).BoundingBox);
    % Crop the second nr
    nr2 = imcrop(im, stats(2).BoundingBox);
    % Crop the third nr
    nr3 = imcrop(im, stats(3).BoundingBox);

    % Pad images and resize to same size
    im1 = padimage(nr1);
    im2 = padimage(nr2);
    im3 = padimage(nr3);

    X{i + (2*(i-1))} = im1; % append the image to the X array
    X{i + 1 + (2*(i-1))} = im2; % append the image to the X array5
    X{i + 2 + (2*(i-1))} = im3; % append the image to the X array
    end
end
%%
ImageFolder ='ProcessedImages/';

for i=1:length(X)
    file_name = "Image"+num2str(i,'%04.f'); % name Image with a sequence of number, ex Image1.png , Image2.png....
    fullFileName = fullfile(ImageFolder, file_name);
    imgName = [ImageFolder,'Image_',num2str(i,'%04.f'),'.png'];
    imwrite(X{i},imgName);
end

%%

imds = imageDatastore('ProcessedImages/', 'Labels', Y);


%%

numTrainingFiles = 300;
[imdsTrain,imdsTest] = splitEachLabel(imds,numTrainingFiles,'randomize');



layers = [    imageInputLayer([56 56 1])
    convolution2dLayer(5,32)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
 
    convolution2dLayer(5,64)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer  
    dropoutLayer(0.25)
    fullyConnectedLayer(10)
    fullyConnectedLayer(3)
    softmaxLayer
    classificationLayer
];

options = trainingOptions('adam', ...
    'ExecutionEnvironment','gpu',...
    'MaxEpochs',60,...
    'InitialLearnRate',1e-3, ...
    'ValidationData',imdsTest, ...
    'ValidationFrequency',10,...
    'Verbose',false, ...
    'Plots','training-progress');

[net, trainingRec] = trainNetwork(imdsTrain, layers,options);

%%
YPredicted = classify(net,imdsTest);
% Compute the predictions on the validation set
plotconfusion(imdsTest.Labels,YPredicted)
%%

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

