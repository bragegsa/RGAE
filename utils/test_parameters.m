function results=test_parameters(data,lambda,S,n_hid, epochs, lr, map, ...
    results, dataset_name, model_name)
% Calculate the AUC value of result.
% INPUTS:
%   - data:    HSI data set (rows by columns by bands);
%   - lambda:  the tradeoff parameter;
%   - S:       the number of superpixels;
%   - n_hid:   the number of hidden layer nodes.
%   - epochs:  the number of epochs
%   - lr:      the learning rate
%   - map:     the ground truth
%   - results: The list that will be filled with results
% OUTPUT:
%   - results: A 2D list of AUC values and what parameters
%              that were used.

    new_results = [];

%   Training the RGAE:
    tic;
    y = RGAE(data,lambda,S,n_hid, map, epochs, lr, dataset_name);
    time = toc;
    
%   Evaluating the RGAE:
    y=reshape(y,size(map,1),size(map,2));
%     imshow(y);
    AUC=ROC(y,map,0);
%     dataset_name = 'abu-airport-1';
    parameters_str = ['lambda', num2str(lambda), 'S', int2str(S) 'n_hid', int2str(n_hid), ...
        'lr', num2str(lr), 'epochs', int2str(epochs), '_', model_name];
    
    file_path = ['results/', dataset_name];
    save_file_to_path = ['results/', dataset_name, '/', join(parameters_str), '.mat'];
    
    status = mkdir(join(file_path));
    save(join(save_file_to_path),'y', 'AUC', 'time');
    
    save_message = ['Saved file to: ', save_file_to_path];
    disp(join(save_message));
    
%     AUC=ROC(y,map,0);
    
%   Adding the results to the result list
    new_results = [new_results; AUC lambda S n_hid, lr];
    disp(new_results);
    results = [results; new_results]; 
end