% import image
coins = imread("images/coins.tif");
%imshow(uint8(coins))
%imhist(coins)

% Plan:
%
%
%%
% Convert to binary
% histogram shows clear valley with lowest at around 110
bw_coins = imbinarize(coins,"global"); % Binarizing uing Otsu's method
figure(1)
imshow(bw_coins)

%%
% fill in holes
% first invert image
bw_coins = imcomplement(bw_coins);
bw_coins_fill = imcomplement(imfill(bw_coins,"holes"));
figure(2)
imshow(bw_coins_fill)
%%
% Trying out distance transform

dist = bwdist(bw_coins_fill);
figure(3)
imshow(mat2gray(dist))
%%
% Inverting image

dist = imcomplement(dist);
figure(4)
imshow(mat2gray(dist))
%%
% Try to use watershed to seperate coins touching

%ws = watershed(dist);
%figure(5)
%imshow(ws)
%%
% watershed not working as intended because of oversegmentation
% will try to mask out only the desired local minima and then apply

mask = imextendedmin(dist,2);
figure(5)
imshowpair(bw_coins_fill,mask,'blend')
%%
% now we only have desired minima

dist_2 = imimposemin(dist,mask);
ws = watershed(dist_2);
bw_coins_ws = bw_coins_fill;
bw_coins_ws(ws == 0) = 1;
figure(6)
imshow(bw_coins_ws)
%%
% much better
% time to label
imlabel = bwlabel(imcomplement(bw_coins_ws),8);
figure(7)
imshow(label2rgb(imlabel, "spring"));
%%
% extract info from labeled objects

F = regionprops(imlabel,"Area");
A = [F.Area];
figure(8)
histogram(A,20)