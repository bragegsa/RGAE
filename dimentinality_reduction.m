addpath(genpath('utils'))

clear all;
close all;
clc
warning('off');

% Loading the dataset
dataset_name = 'abu-airport-1';
file_path = 'datasets/';
load(join([file_path, dataset_name]));

Y=myPCA(data);
y=Y(:,:,end);
y=(y-min(y(:)))./(max(y(:))-min(y(:)));

filename = ['dim_red/PCA/',dataset_name, '_PCA.mat'];
save(join(filename),'y')
% imshow(y);