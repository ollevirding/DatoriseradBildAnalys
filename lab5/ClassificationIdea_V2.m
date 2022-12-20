% SPLITING IMAGES USING COMPUTER VISION TOOLBOX
% Sorting out bad images
%%
clear all
GTbase = {};
GTlabel = {};
labels = readmatrix("labels.txt");
index = 0;

Folder = "ClassifyingImages/";

if ~exist(Folder, 'dir')
    mkdir(Folder)
end

%%
for i = 1:1200
name = "train_" + sprintf('%04d',i) + ".png";

img = imread("imagedata/" + name);
img = medfilt2(img, [5, 5]);
bw_img = imbinarize(img,"global");


im = imcomplement(bw_img);
im = bwareaopen(im,20);



% Get the bounding boxes of the individual nrs
stats = regionprops(im, 'BoundingBox');

    if length(stats) == 3
        index = index + 1;
        im_base = uint8(im);
        for j = 1:3
            im = classImage(im, stats(j).BoundingBox, labels(i,j));
        end
        im_base = im_base*255;
        im_label = categorical(im);

        GTlabel{end + 1} = im_label;
        writeimage(im_base, index)

        %Använd map = [0 0 0; 1 0 0, 0 1 0, 0 0 1]
        %och I = label2rgb(im_label, map)
        %För att display bilder
    end
end
%%
% Här går den sönder
imds = imageDatastore('ClassifyingImages/', 'Labels', GTlabel);


%%

function classIm = classImage(image_bin, BoundingBox, label)
y = floor(BoundingBox(1));
x = floor(BoundingBox(2));
b = ceil(BoundingBox(3));
a = ceil(BoundingBox(4));

if label == 0
    label = 3;
end
classIm = image_bin;
for X = x:x+a
    for Y = y:y+b
        if image_bin(X,Y) == 1
            classIm(X,Y) = label;
        end
    end
end
end

function writeimage(im1, indexnr)
        ImageFolder = "ClassifyingImages/";
        file_name = "Image"+num2str(indexnr,'%04.f'); % name Image with a sequence of number, ex Image1.png , Image2.png....
        fullFileName = fullfile(ImageFolder, file_name);
        imgName = strjoin([ImageFolder,'Image_',num2str(indexnr,'%04.f'),'.png']);
        imwrite(im1,imgName);
end
