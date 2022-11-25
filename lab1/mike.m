close all

% ==== % READING FILE % ==== %
%%
I = imread("images/napoleon.png");

% ==== % Showing picture % ==== %

imshow(I)
figure
image(I)
colorbar
figure
imagesc(I)
colorbar

close all

% ==== % IMtool % ==== %

imtool(I) % "Pixel info (1,1) 89
disp(I(1,1))
%%
% Q1: Yes, 89

I1 = I;
I2 = imread("images/napoleon_dark.png");
I3 = imread("images/napoleon_light.png");

figure
imhist(I1)
figure
imhist(I2)
figure
imhist(I3)
%%
close all

Is = single(I);
imtool(Is)
imtool(Is/255)

%Q3: Is is 4 Byte data while I is 1 Byte
figure
imagesc((I/64)*64)
figure
imagesc((Is/64)*64)
%%
close all
%Q4, Q5
Ib = I + 75;
Ic = I*0.5;

figure
imshow(I)
figure
imshow(Ib)
figure
imshow(Ic)

%%
close all

g = 2;

imhist(I);
L = double(I).^g;
out = uint8(L .* (255/max(max(L))));
figure
imhist(out);

figure
imshow(I)
figure
imshow(out)

%%
close all
figure
imhist(I)
figure
imshow(I)
Ieq = histeq(I);
figure
imhist(Ieq)
figure
imshow(Ieq)

%%
I = imread("images/cameraman.png");
Jnf = imresize(I, [78 78], 'nearest', 'antialiasing', false);
Jnt = imresize(I, [78 78], 'nearest', 'antialiasing', true);
Jbf = imresize(I, [78 78], 'bilinear', 'antialiasing', false);
Jbt = imresize(I, [78 78], 'bilinear', 'antialiasing', true);
figure
imshow(I)
figure
tiledlayout(2,2)
nexttile
imshow(Jnf)
nexttile
imshow(Jnt)
nexttile
imshow(Jbf)
nexttile
imshow(Jbt)
%%
%BRAINSCAN
close all

B1 = imread("images/brain1.png");
B2 = imread("images/brain2.png");
B3 = imread("images/brain3.png");

figure
tiledlayout(2,3)
nexttile
imshow(B1)
nexttile
imshow(B2)
nexttile
imshow(B3)

nexttile
Bmb = (B1+B2)./2;
imshow(Bmb)
nexttile
Bm = uint8((single(B1)+single(B2))./2);
imshow(Bm)
nexttile
Bc = Bm - B3;
imshow(Bc)

%%
close all

W = imread("images/wrench.png");

J = imrotate(W,20);
K = imrotate(W,20, 'bilinear');

figure
tiledlayout(1,3)
nexttile
imshow()





