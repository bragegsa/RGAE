%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% APPLICATION:
%   Hyperspectral Anomaly Detection.
% INPUTS:
%   - data:   HSI data set (rows by columns by bands);
%   - lambda: the tradeoff parameter;
%   - S:      the number of superpixels;
%   - n_hid:  the number of hidden layer nodes.
% OUTPUTS:
%   - y:    final detection map (rows by columns);
%   - AUC:  AUC value of 'y'.
%  REFERENCE:
%   G. Fan, Y. Ma, X. Mei, F. Fan, J. Huang and J. Ma, "Hyperspectral Anomaly
%   Detection With Robust Graph Autoencoders," IEEE Transactions on Geoscience 
%   and Remote Sensing, 2021.
%   G. Fan, Y. Ma, J. Huang, X. Mei and J. Ma, "Robust Graph Autoencoder for 
%   Hyperspectral Anomaly Detection," ICASSP 2021 - 2021 IEEE International 
%   Conference on Acoustics, Speech and Signal Processing (ICASSP), 2021, 
%   pp. 1830-1834.
% Written and sorted by Ganghui Fan in 2021. All rights reserved.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% addpath(fullfile('utils'));
% addpath(fullfile('utils/KPCA/SuperPCA'));
% addpath(fullfile('utils/KPCA/LanZhSuperKPCA'));
addpath(genpath('utils'))

clear all;
close all;
clc
warning('off');

model_name = 'def_adam';
% Loading the dataset
% dataset_name = 'abu-airport-4';
% dataset_name = 'abu-beach-4';
dataset_name = 'abu-urban-5';
file_path = 'datasets/';
load(join([file_path, dataset_name]));
	
mask = map;

% Normalizing the data
data = (data-min(data(:)))./(max(data(:))-min(data(:)));

% Parameters to optimize: (default values)
% lambda=1e-1;
lambda = [1e-1, 1e-2, 1e-3, 1e-4];
S=[50, 100, 150, 300, 500];
n_hid=100;
epochs = 5000;
lr = [0.1, 0.01, 0.001];

% Defining the results list:
results = [];


for l = lambda
    results = test_parameters(data,l,150,n_hid, epochs, 0.01, map, results, ...
        dataset_name, model_name);
end

[max_auc, idx] = max(results(:,1));

lambda = results(idx, 2);
disp('Best lambda:');
disp(lambda)

for s = S
    results = test_parameters(data,lambda,s,n_hid, epochs, 0.01, map, results, ...
        dataset_name, model_name);
end

[max_auc, idx] = max(results(:,1));

S = results(idx, 3);
disp('Best S:');
disp(S)

for u = lr
    results = test_parameters(data,lambda,S,n_hid, epochs, u, map, results, ...
        dataset_name, model_name);
end

[max_auc, idx] = max(results(:,1));
disp(results(idx,:));
