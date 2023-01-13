function S = my_classifier(im, parameters1, parameter2)

img = medfilt2(im, [5, 5]);
bw_img = imbinarize(img,"global");

im = imcomplement(bw_img);
im = bwareaopen(im,20);
    
stats = regionprops(im, 'BoundingBox');
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


S = zeros(1,3);

NETWORK = parameters1.net;

S(1) = classify(NETWORK, im1);
S(2) = classify(NETWORK, im2);
S(3) = classify(NETWORK, im3);
S = double(S) - 1;

end

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


