addpath(genpath('utils'))

clear all;
close all;
clc
warning('off');

% Her bør KPCA hentes først, ikke ferdig data
dataset_name = 'abu-airport-1';
file_path = 'datasets/';
load(join([file_path, dataset_name]));

mask = map;

file_path = 'dim_red/KPCA3D/';
kpca_type_1 = 'sigm';
kpca_type_2 = 'lapl';

file_name = [file_path, dataset_name, '_', kpca_type_1];
load(join([file_name]));
data1 = real(Data_KPCA_3D);
data1 = (data1-min(data1(:)))./(max(data1(:))-min(data1(:)));
% figure, imshow(data1(:,:,1))


file_name = [file_path, dataset_name, '_', kpca_type_2];
load(join([file_name]));
data2 = real(Data_KPCA_3D);
data2 = (data2-min(data2(:)))./(max(data2(:))-min(data2(:)));
% figure, imshow(data2(:,:,1))

% Parameters
lambda = 1e-2; % MK not 0 or 1 if lambda = 0;
S=300;
n_hid=100;

lr = 0.01;
epochs = 1500;

tic;
y1 = RGAE_MK(data1,lambda,S,n_hid, map, epochs, lr, dataset_name);
time = toc;

tic;
y2 = RGAE_MK(data2,lambda,S,n_hid, map, epochs, lr, dataset_name);
y2 = 1 - y2; % If laplace
time = toc;

AUC1=ROC(y1,map,0);
AUC2=ROC(y2,map,0);
disp(AUC1);
disp(AUC2);

y1=reshape(y1,100,100);
y2=reshape(y2,100,100);
% figure, imshow(y1);
% figure, imshow(y2);

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

print_statement = ['The best alpha value was ', num2str(best_alpha), ...
    ' with an AUC score of ', num2str(best_auc), '.'];

disp(join(print_statement));
