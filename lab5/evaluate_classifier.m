close all, clear all; % Close all figures, and clear all variables

%% Load the labels
true_labels = importdata('labels.txt');

%% Load or set the parameters for your classifier.
% For instance, if you choose to save your parameters as an .mat-file
% you can load them using the load(filename) function.
% You are allowed to have more/less than 2 parameters.
parameters{1} = 123;
parameters{2} = 'this is another parameter for my classifier.';

%% Evaluate the classifier
tic; % Start the timer
my_labels = zeros(size(true_labels));
N = size(true_labels,1);
for k = 1:N
    im = imread(sprintf('imagedata/train_%04d.png', k));
    my_labels(k,:) = my_classifier(im, parameters{:});
end
fprintf('\n\nAverage precision: \n');
fprintf('%f\n\n',mean(sum(abs(true_labels - my_labels),2)==0));
toc; % Stop the timer

