imfile = "napoleon.png";

mike = imread("images/"+imfile);
mike = im2gray(mike);
histogram(mike, 256)
histval = histogram(mike, 256).Values;


histvalcum = cumsum(histval);
greys = 0:1:255;
histvalcum = histvalcum/max(max(histvalcum))*255;
plot(greys,histvalcum)

mike2 = zeros(size(mike));
[a,b] = size(mike);
for i = 1:a
    for j = 1:b
        pixelval = mike(i,j) + 1;
        mike2(i,j) = histvalcum(round(pixelval));
    end
end

figure()
imshow(uint8(mike2))
title("After hist-eq")
figure()
imshow(mike)
title("Before")