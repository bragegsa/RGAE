function Y=myClustering(data, mask)
% Clustering
    
    det_out = {};
    clusters = 2;
    
    det_out{end+1}.parameters.n_cluster = clusters;
    [cbad_out,cluster_img] = cbad_anomaly(data,mask,det_out{end}.parameters.n_cluster);
    det_out{end}.result = cbad_out; 
    
    det_out{end}.cluster_img = cluster_img;
    det_out{end}.method = 'cbad\_anomaly';


    Y = cbad_out;
end
