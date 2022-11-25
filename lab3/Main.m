
%% READING FILE:

I = imread("images/coins.tif");
Ismall = I(1:50,301:350); % Used for testing

%% BINARIZATION FOR EASIER HANDLING:

%I = Ismall; % Uncomment this line to use small image

Ibin = imbinarize(I, 'global');

%% MEAN FILTER FOR NOISE REDUCTION

IbinMed = medfilt2(Ibin);

%% DISTANCE TRANSFORM

Idist = bwdist(IbinMed,'euclidean');


%% INVERSE ON DISTANCE TRANSFORM

one = ones(size(IbinMed));

Iidist = one - Idist;
imshow(mat2gray(Iidist))

%% REDUCING RADIUS: (THIS MIGHT BE COMPLETLY REDUNDANT)

[w, h] = size(Iidist);
for i=1:w
    for j=1:h
        if Iidist(i,j) > - 16   % If the distance from black is not greater than 16, it counts as black. This helps separating
            Iidist(i,j) = 1;
        end
    end
end

imshow(mat2gray(Iidist))

%% WATERSHED ON DISTANCE TRANSFORM

Iw = watershed(Iidist);
imshow(Iw)  % Don't know what to do with this result. I dont think it looks that great