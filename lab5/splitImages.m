img = imread("imagedata/train_0002.png");
img = medfilt2(img);
bw_img = imbinarize(img,"global");

figure(3)
im = imcomplement(bw_img);
im = bwareaopen(im,20);
imshow(im)

imm = 255*uint8(im);
%figure(4)
%imshow(imm)

[h,w] = size(uint8(255.*im));
y = sum(im);
x = 1:1:w;
%figure(5)
%plot(x,y) % funkar som tänkt

% begin by removing edges

cutfirst = find(y,1,'first');
cutlast = find(y,1,'last');

im = im(:,cutfirst:cutlast);
%figure(24)
%imshow(im)

% find where gaps appear and split into 3 parts

y = sum(im);
cutfirst = find(y == 0,1,'first') - 1;
cutlast = find(y == 0,1,'last') + 1;
split1 = im(:,1:cutfirst);
ysplit1 = sum(split1);
split2 = im(:,(cutfirst+1):(cutlast-1));
ysplit2 = sum(split2);
split3 = im(:,cutlast:end);
ysplit3 = sum(split3);
%figure(12)
%imshow(split1)
%figure(13)
%imshow(split2)
%figure(14)
%imshow(split3)

if any(ysplit2) % if middle part has non-zero values = no overlaps, all images will be done after this
    vals = find(ysplit2);
    cutfirst = vals(1);
    cutlast = vals(end);
    split2 = split2(:,cutfirst:cutlast);
    %figure(15)
    %imshow(split2)
    im1 = split1;
    im2 = split2;
    im3 = split3;

else % middle part only has zeros which means one of the other images has overlaps, need special treatment

    if length(ysplit1) > length(ysplit3) %
        locmin = find(islocalmin(ysplit1)); % tar fram index för alla locala minima
        %behöver kolla vilken av dom som ger lägsta y-värdet men vill inte
        %tappa indexet
        ylocmin = ysplit1(locmin);
        [a,b] = min(ylocmin);
        split = locmin(b);
        
        im1 = split1(:,1:split);
        im2 = split1(:,split:end);
        im3 = split3;
    else
        locmin = find(islocalmin(ysplit3));
        ylocmin = ysplit3(locmin);
        [a,b] = min(ylocmin);
        split = locmin(b);
        
        im1 = split1;
        im2 = split3(:,1:split);
        im3 = split3(:,split:end);

    end
end

% remove edges åt andra hållet här

y1 = sum(im1,2);
y2 = sum(im2,2);
y3 = sum(im3,2);

cutfirst1 = find(y1,1,'first');
cutlast1 = find(y1,1,'last');
cutfirst2 = find(y2,1,'first');
cutlast2 = find(y2,1,'last');
cutfirst3 = find(y3,1,'first');
cutlast3 = find(y3,1,'last');

im1 = im1(cutfirst1:cutlast1,:);
im2 = im2(cutfirst2:cutlast2,:);
im3 = im3(cutfirst3:cutlast3,:);

% resize images

im1 = imresize(im1,[40 40]);
im2 = imresize(im2,[40 40]);
im3 = imresize(im3,[40 40]);

% superimpose on empty matrix to add padding

%[h1,w1] = size(im1);
%[h2,w2] = size(im2);
%[h3,w3] = size(im3);

nr1 = zeros(56,56);
nr2 = zeros(56,56);
nr3 = zeros(56,56);

nr1(8:47,8:47) = im1;
nr2(8:47,8:47) = im2;
nr3(8:47,8:47) = im3;

nr1 = imgaussfilt(255*nr1,1.5);
nr2 = imgaussfilt(255*nr2,1.5);
nr3 = imgaussfilt(255*nr3,1.5);

figure(15)
subplot(1,3,1), imshow(uint8(nr1))
subplot(1,3,2), imshow(uint8(nr2))
subplot(1,3,3), imshow(uint8(nr3))
