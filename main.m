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

model_name = 'default';
% model_name = 'KPCA4';
% model_name = 'Clustering3';
% Loading the dataset
% dataset_name = 'abu-airport-4';
% dataset_name = 'abu-beach-3';
dataset_name = 'abu-urban-5';
file_path = 'datasets/';
load(join([file_path, dataset_name]));

mask = map;

% Normalizing the data
data = (data-min(data(:)))./(max(data(:))-min(data(:)));

% Parameters to optimize: (default values)
lambda = 1e-3;
S=100;
n_hid=100;

lr = 0.01;
epochs = 1500;

% Defining the results list:
results = [];

% Training
results = test_parameters(data,lambda,S,n_hid, epochs, lr, map, results, ...
    dataset_name, model_name);

disp(results);
