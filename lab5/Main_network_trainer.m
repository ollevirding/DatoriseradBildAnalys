%%

labels = string(readmatrix("labels.txt")); % read the labels from a file
Y = reshape(labels.',1,[]);
Y = categorical(Y)';

%%

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
    stats;
    if length(stats) == 3

         % Crop the first nr
        nr1 = imcrop(im, stats(1).BoundingBox);
        % Crop the second nr
        nr2 = imcrop(im, stats(2).BoundingBox);
        % Crop the third nr
        nr3 = imcrop(im, stats(3).BoundingBox);
    else
        [nr1, nr2, nr3] = splitimage(im, stats);

    end
        % Pad images and resize to same size
    im1 = padimage(nr1);
    im2 = padimage(nr2);
    im3 = padimage(nr3);

    X{i + (2*(i-1))} = im1; % append the image to the X array
    X{i + 1 + (2*(i-1))} = im2; % append the image to the X array5
    X{i + 2 + (2*(i-1))} = im3; % append the image to the X array
end
%%
Folder = "SegmentedImages/";

if ~exist(Folder, 'dir')
    mkdir(Folder)
end

for i=1:length(X)
    imgName = strjoin([Folder,'Image_',num2str(i,'%04.f'),'.png']);
    imwrite(X{i},imgName);
end


%%

imds = imageDatastore('SegmentedImages/', 'Labels', Y);

%%


numTrainingFiles = 300;
[imdsTrain,imdsTest] = splitEachLabel(imds,numTrainingFiles,'randomize');



layers = [    imageInputLayer([28 28 1])
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
    fullyConnectedLayer(3)
    softmaxLayer
    classificationLayer
];

options = trainingOptions('adam', ...
    'ExecutionEnvironment','gpu',...
    'MaxEpochs',120,...
    'InitialLearnRate',1e-3, ...
    'ValidationData',imdsTest, ...
    'ValidationFrequency',10,...
    'Verbose',false, ...
    'Plots','training-progress');

[net, trainingRec]= trainNetwork(imdsTrain, layers,options);
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


function [img1, img2, img3] = splitimage(img, stats)

    n = length(stats);
    
    if n == 2
        
        nr1 = imcrop(img, stats(1).BoundingBox);
        nr2 = imcrop(img, stats(2).BoundingBox);
        [h, w1] = size(nr1);
        [h, w2] = size(nr2);
        
        if w1 < w2
            number_to_split = nr2;
            img1 = nr1;

            [h, w] = size(number_to_split);
        
            rect1 = [1 1 floor(w/2) h];
            rect2 = [floor(w/2)+1 1 floor(w/2) h];
    
            img2 = imcrop(number_to_split, rect1);
            img3 = imcrop(number_to_split, rect2);

        else

            number_to_split = nr1;
            img3 = nr2;

            [h, w] = size(number_to_split);
        
            rect1 = [1 1 floor(w/2) h];
            rect2 = [floor(w/2)+1 1 floor(w/2) h];
    
            img1 = imcrop(number_to_split, rect1);
            img2 = imcrop(number_to_split, rect2);
        end
   
    
    end
    
    if n == 1
        nr1 = imcrop(img, stats(1).BoundingBox);
        [h, w] = size(nr1);

        rect1 = [1 1 floor(w/3) h];
        rect2 = [floor(w/3)+1 1 floor(w/3) h];
        rect3 = [2*floor(w/3)+1 1 2*floor(w/3) h];

        img1 = imcrop(nr1, rect1);
        img2 = imcrop(nr1, rect2);
        img3 = imcrop(nr1, rect3);
        

    end

    if n > 3
        
        images = [];
        for i = 1:length(stats)
            im = imcrop(img, stats(i).BoundingBox);
            [h, w] = size(im);
            if h*w > 500
                images = [images, i];
            end
        end
        
        img1 = imcrop(img, stats(images(1)).BoundingBox);
        img2 = imcrop(img, stats(images(2)).BoundingBox);
        img3 = imcrop(img, stats(images(3)).BoundingBox);


    end
end
