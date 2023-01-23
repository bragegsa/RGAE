addpath(genpath('utils'))

clear all;
close all;
clc
warning('off');

dataset_name = 'abu-airport-1';
file_path = 'datasets/';
load(join([file_path, dataset_name]));

% Normalizing the data
data = (data-min(data(:)))./(max(data(:))-min(data(:)));

% Try gauss, sigm, lapl
kpca_type = 'lapl';
Y=myKPCA(data, kpca_type, dataset_name);