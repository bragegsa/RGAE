addpath(genpath('utils'))

clear all;
close all;
clc
warning('off');

load('y1.mat');
load('y2.mat');

dataset_name = 'abu-airport-1';
file_path = 'datasets/';
load(join([file_path, dataset_name]));

mask = map;

alpha_values = 0:0.01:1;
best_alpha = 0;
best_auc = 0;

for alpha=alpha_values
    y = alpha*y1 + (1-alpha)*y2;
    AUC = ROC(y, map, 0);
    if AUC > best_auc
        best_auc = AUC;
        best_alpha = alpha;
    end
end


% alpha = fminbnd(find_optimal_alpha, 0, 1);
% 
% function [AUC] = find_optimal_alpha(alpha)
%     
%     addpath(genpath('utils'))
%     load('y1.mat');
%     load('y2.mat');
%     
%     y = alpha*y1 + (1-alpha)*y2;
%     AUC = ROC(y, map, 0);
%     
% end