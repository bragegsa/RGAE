list = [0.9,2,3; 
        0.8,3,4; 
        0.7,4,5];

[max_AUC, Idx] = max(list(:,1));

disp(max_AUC);
disp(Idx);

disp(list(Idx,3));