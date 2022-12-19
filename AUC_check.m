clear all;
close all;
clc
warning('off');



% Choose the dataset you want to load

% dataset_name = 'abu-airport-2';
% dataset_name = 'abu-beach-1';
dataset_name = 'abu-urban-3';

code_name_def = 'lambda0.01S500n_hid100lr0.001epochs1200_default';
code_name_adam = 'lambda0.01S500n_hid100lr0.001epochs5000_def_adam';


[y_def, mask] = y_mask(dataset_name, code_name_def);
[y_adam, mask] = y_mask(dataset_name, code_name_adam);


[auc, p_f_def, p_d_def] = comp_ROC(y_def,mask, 0);
disp(auc);
[auc, p_f_adam, p_d_adam] = comp_ROC(y_adam,mask, 0);

hold on
grid on

title(dataset_name);
xlabel('False positive rate');
ylabel('True positive rate');

plot(p_f_def, p_d_def);
plot(p_f_adam, p_d_adam);

hold off
legend({'Default RGAE','ADAM RGAE'}, 'Location', 'southeast');
