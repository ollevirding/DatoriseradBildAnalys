I = imread("images/cameraman.png");
im = double(I);
f = fftshift(fft2(im));


[width, height] = size(f);

radius = 25;

filter = zeros(width, height); %Skapar matrix med nollor av size I
for i=1:width
    for j=1:height
        if norm([width/2 - j, height/2 - i]) < radius 
            filter(i,j) = 1; % Om vi befinner oss inom cikelns radie byter vi 0 till 1
        end
    end
end

newf = f.*filter; % Multiplicerar elementvis filtret med FT-bild
invers = ifft2(ifftshift(newf)); % Skapar filtrerad bild


figure(1) % FOURIER BILDER
tiledlayout(1,3)
nexttile
imagesc(abs(log(f)))
colorbar
title('Original Fourier image')
nexttile
imagesc(filter)
colorbar
title('Filter')
nexttile()
imagesc(abs(log(newf)))
colorbar
title("Filterd Fourier image")
abs(log(newf))

figure(2)   % SPATIAL BILDER
tiledlayout(1,2)
nexttile
imshow(I)
title("Original Image")
nexttile
imshow(uint8(abs(invers)))
title("Filtred Image")  