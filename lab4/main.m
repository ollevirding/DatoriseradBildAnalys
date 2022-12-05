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

%% PLOTTING CLASSIFIED IMAGE USING GRAYSCALE TRESHOLD

t1 = 80;    %Chosen from histogram
t2 = 150;   % ----||-----

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

I3(:,:,1) = G; % Create an image with two bands/features
I3(:,:,2) = B;
[data,class] = create_training_data(I3,label_im); % Arrange the training data into vectors
figure(8);scatterplot2D(data,class); % View the training feature vectors