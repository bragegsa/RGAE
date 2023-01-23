% Test of cbad

addpath(fullfile('utils'));


% Loading the dataset
% dataset_name = 'abu-airport-1';
% dataset_name = 'abu-beach-4';
dataset_name = 'abu-urban-5';
file_path = 'datasets/';
load(join([file_path, dataset_name]));
hsi_img = data;
mask = map;

det_out = {};
clusters = 2;

tic;
det_out{end+1}.parameters.n_cluster = clusters;
[cbad_out,cluster_img] = cbad_anomaly(hsi_img,mask,det_out{end}.parameters.n_cluster);
det_out{end}.result = cbad_out; 
% det_out{end}.result = cluster_img; 
% disp(cbad_out);
det_out{end}.cluster_img = cluster_img;
det_out{end}.method = 'cbad\_anomaly';
% filename = ['Results/', name_of_dataset, '/cbad_out_', int2str(clusters), 'c.mat'];
% save(join(filename),'cbad_out');
toc;

% image(cbad_out);
filename = ['dim_red/C2/', dataset_name, '_clustered'];
save(join(filename), 'cbad_out');

disp(ROC(cbad_out,mask,1));