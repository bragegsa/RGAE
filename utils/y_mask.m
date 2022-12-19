function [y, mask] = y_mask(dataset_name, code_name)
    dataset_path = join(['datasets/', dataset_name, '.mat']);
    result_path = ['results/', dataset_name, '/', code_name, '.mat'];

    load(dataset_path);
    mask = map;
    load(join(result_path));
end