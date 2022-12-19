clear all;
close all;
clc
warning('off');

% What to test:
test_parameter = 'n_hid';

% Loading the dataset
dataset_name = 'abu-airport-1';
file_path = 'datasets/anomaly/';
load(join([file_path, dataset_name]));
	
mask = map;

% Normalizing the data
data = (data-min(data(:)))./(max(data(:))-min(data(:)));

% Parameters to optimize: (default values)
lambda=1e-2;
S=200;
n_hid=200;

% Lists to fill in testing values
% lambda_list = [1e-2];
lambda_list = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5];
S_list = [50, 100, 150, 200, 400];
n_hid_list = [50, 100, 150, 200, 400];
epochs = 2500;
lr = 0.01;

% Defining the results list:
results = [];

% Training the RGAE
if strcmp(test_parameter, 'n_hid')
    for n_hid = n_hid_list
        results = test_parameters(data,lambda,S,n_hid, epochs, lr, map, results, dataset_name);
    end
elseif strcmp(test_parameter,'lambda')
    for lambda = lambda_list
        results = test_parameters(data,lambda,S,n_hid, epochs, lr, map, results, dataset_name);
    end
elseif strcmp(test_parameter, 'S')
    for S = S_list
        results = test_parameters(data,lambda,S,n_hid, epochs, lr, map, results, dataset_name);
    end
end

disp(results);
% filename = 'results/param_opt_lambda_abu_beach_1';
% save(filename);