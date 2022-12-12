%% LOAD DATA

load cdata.mat

%% PLOTTING CDATA

figure(1);plot(cdata(:,1),cdata(:,2),'.')

% 1. No, it is not possible due to overlapping of the two regions across
% both x-axis and y-axis


%% READING GRAYSCALE IMAGE

I = imread('handBW.pnm'); % Read the image
figure(2);imshow(I); % Show the image
figure(3);imhist(I); % Show the histogram
gray = I;

%% PLOTTING CLASSIFIED IMAGE USING GRAYSCALE TRESHOLD

t1 = 80;    %Chosen from histogram
t2 = 135;   % ----||-----

figure(4);mtresh(I,t1,t2);

% 2. No, it does not work due to the ring-object and hand-object being of
% the same grayscale color

%% READING COLORED IMAGE

I2 = imread('hand.pnm'); % Read the image
figure(5);imshow(I2); % Show the image
R = I2(:,:,1); % Separate the three layers, RGB
G = I2(:,:,2);
B = I2(:,:,3);
figure(6);plot3(R(:),G(:),B(:),'.') % 3D scatterplot of the RGB data

%% READING TRAINING DATA

label_im = imread('hand_training.png'); % Read image with labels
figure(7);imagesc(label_im); % View the training areas


%% CLASSIFYING DATA BASED ON COLOR CHANNEL

% Can only run one assestment at a time

IRGB = I2;

IRG(:,:,1) = R;
IRG(:,:,2) = G;


IRB(:,:,1) = R;
IRB(:,:,2) = B;


IBG(:,:,1) = B;
IBG(:,:,2) = G;

figure(8)
tiledlayout(2,4)
for i=1:1 % For-loop to hide the ugliness

nexttile
[data1,class1] = create_training_data(IRGB,label_im);  % Arrange the training data into vectors
Itest1 = im2testdata(IRGB); % Reshape the image before classification
C = classify(double(Itest1),double(data1),double(class1)); % Train classifier and classify the data
ImC = class2im(C,size(IRGB,1),size(IRGB,2)); % Reshape the classification to an image
imagesc(ImC); % View the classification result
title('Classification of image using Channels: RGB')

nexttile
[data,class] = create_training_data(IRG,label_im);  % Arrange the training data into vectors
Itest = im2testdata(IRG); % Reshape the image before classification
C = classify(double(Itest),double(data),double(class)); % Train classifier and classify the data
ImC = class2im(C,size(IRG,1),size(IRG,2)); % Reshape the classification to an image
imagesc(ImC); % View the classification result
title('Classification of image using Channels: RG')

nexttile
[data,class] = create_training_data(IRB,label_im);  % Arrange the training data into vectors
Itest = im2testdata(IRB); % Reshape the image before classification
C = classify(double(Itest),double(data),double(class)); % Train classifier and classify the data
ImC = class2im(C,size(IRB,1),size(IRB,2)); % Reshape the classification to an image
imagesc(ImC); % View the classification result
title('Classification of image using Channels: RB')

nexttile
[data,class] = create_training_data(IBG,label_im);  % Arrange the training data into vectors
Itest = im2testdata(IBG); % Reshape the image before classification
C = classify(double(Itest),double(data),double(class)); % Train classifier and classify the data
ImC = class2im(C,size(IBG,1),size(IBG,2)); % Reshape the classification to an image
imagesc(ImC); % View the classification result
title('Classification of image using Channels: BG')

nexttile
[data,class] = create_training_data(R,label_im);  % Arrange the training data into vectors
Itest = im2testdata(R); % Reshape the image before classification
C = classify(double(Itest),double(data),double(class)); % Train classifier and classify the data
ImC = class2im(C,size(R,1),size(R,2)); % Reshape the classification to an image
imagesc(ImC); % View the classification result
title('Classification of image using Channels: R')

nexttile
[data,class] = create_training_data(G,label_im);  % Arrange the training data into vectors
Itest = im2testdata(G); % Reshape the image before classification
C = classify(double(Itest),double(data),double(class)); % Train classifier and classify the data
ImC = class2im(C,size(G,1),size(G,2)); % Reshape the classification to an image
imagesc(ImC); % View the classification result
title('Classification of image using Channels: G')

nexttile
[data,class] = create_training_data(B,label_im);  % Arrange the training data into vectors
Itest = im2testdata(B); % Reshape the image before classification
C = classify(double(Itest),double(data),double(class)); % Train classifier and classify the data
ImC = class2im(C,size(B,1),size(B,2)); % Reshape the classification to an image
imagesc(ImC); % View the classification result
title('Classification of image using Channels: B')

nexttile
[data,class] = create_training_data(gray,label_im);  % Arrange the training data into vectors
Itest = im2testdata(gray); % Reshape the image before classification
C = classify(double(Itest),double(data),double(class)); % Train classifier and classify the data
ImC = class2im(C,size(gray,1),size(gray,2)); % Reshape the classification to an image
imagesc(ImC); % View the classification result
title('Classification of image using Channels: Gray')

end

%figure(8);scatterplot2D(data,class); % View the training feature vectors
% Use the row above (133) to run two chanels

%figure(8);scatterplot3D(data,class); % View the training feature vectors
% Use the row above (136) to run all three chanels


% 3. LDA Assume that all channels are Gaussian distributed and share theIRGB
% same Variance but can differ in Average value.
%
% For a classifier to be linear for each class is described by a linear
% combination of each attribute

% 4. The results have improved using classification instead of
% using tresholding. This was also the case for using single channels or
% grayscale. When using multiple channels the result does improve but in
% the case when using only Green channel or Green-Blue channel the result
% does not significantly improve. We can also not see a clear improvement
% when going from RG or RB to RGB.


