function y=myRGAE_momentum(X,SG,lamda,hid, map, epochs, lr, idex)%
% Training of RGAE for hyperspectral anomaly detection
% INPUTS:
%   - X:     HSI data set (rows*columns by bands);
%   - SG:    the Laplacian matrix;
%   - lambda:the tradeoff parameter;
%   - hid:   the number of hidden layer nodes.
%   - epochs:  the number of epochs
%   - lr:      the learning rate
%   - map:     the ground truth
%   - idex:  the indices of the rows of 'X_new' corresponding to the reshaped 'data'.
% OUTPUT:
%   - y:     final detection map (rows by columns).

    % Parameter settings
    [n,L]=size(X);X=X';
%     lr=0.01;epochs=1200;                   % learning rate and Itermax
%     lr=0.01;epochs=100;                   % learning rate and Itermax
%     lr=0.01;epochs=200;                   % learning rate and Itermax
    W1=0.01*rand(hid,L);b1=rand(hid,1);     % weights and biases for encoder
    W2=0.01*rand(L,hid);b2=rand(L,1);       % weights and biases for decoder
    batch_num=10;batch_size=n/batch_num;    % training with MBGD
    
    % Momentum for b1, W1, b2 and W2
    mt_prev_b1 = 0;
    mt_prev_W1 = 0;
    
    mt_prev_b2 = 0;
    mt_prev_W2 = 0;
    
    % RMSP for b1, W1, b2 and W2
    vt_prev_b1 = 0;
    vt_prev_W1 = 0;
    
    vt_prev_b2 = 0;
    vt_prev_W2 = 0;
    
    n_AUC = 10;
    AUC_values = zeros(1,n_AUC); % Holds the previous n_AUC AUC values for dropout
    All_AUC = [0];
    
    for epoch=1:epochs
        ind=randperm(n);
        for j=1:batch_num
            x=X(:,ind((j-1)*batch_size+1:j*batch_size));            % fetch the batch and the corresponding Laplacian sub-matrix
            s_G=SG(ind((j-1)*batch_size+1:j*batch_size),ind((j-1)*batch_size+1:j*batch_size));
            % Forward
            z=sigmoid(W1,x,b1);
            x_hat=sigmoid(W2,z,b2);
            res=sqrt(sum((x-x_hat).^2));
            % Backward
            [W1,b1,W2,b2,mt_prev_W1,mt_prev_b1,mt_prev_W2,mt_prev_b2,vt_prev_W1,vt_prev_b1,vt_prev_W2,vt_prev_b2] ...
                =trainAE(x,z,x_hat,W1,b1,W2,b2,res,s_G,lr,lamda,mt_prev_W1,mt_prev_b1,mt_prev_W2,mt_prev_b2,vt_prev_W1,vt_prev_b1,vt_prev_W2,vt_prev_b2);
        end

        % For every epoch:
        string = ['Epoch number: ', num2str(epoch), '.']; 
        disp(string);
        
        % Droput:
        Z_test=sigmoid(W1,X,b1);
        X_hat_test=sigmoid(W2,Z_test,b2);

        y_test=sum((X-X_hat_test).^2);
        
        zips=[idex,y_test'];            % Recover the original image with 'idex'
        zips_sort=sortrows(zips,1);
        y_test=zips_sort(:,2);
        
        y_test=reshape(y_test,size(map,1),size(map,2));
        
        AUC = ROC(y_test,map,0);
        prev_AUC = All_AUC(end);
        AUC_change = abs(AUC - prev_AUC);
        
        All_AUC(end + 1) = AUC;
        
        dropout = true;
        if dropout == true
            if AUC < min(AUC_values) && epoch > 400 | AUC_change < 0.0001
                string = ['Last AUC value was ', num2str(AUC), ' and the minimal of the last AUCs was ', num2str(min(AUC_values)), '.'];
                disp(string);
                break
            else
                AUC_values = [AUC_values(2:end)];
                AUC_values(end + 1) = AUC;
            end
        end
        
        
        
    end
    
    plot_AUC = true;
    if plot_AUC == true
        L = length(All_AUC);
        empty = 1:L;
        plot(empty, All_AUC);
    end

    % Output
    Z=sigmoid(W1,X,b1);

    % X_hat is the predicted background of the image?
    % Perhaps return it to analyze
    X_hat=sigmoid(W2,Z,b2);
    y=sum((X-X_hat).^2);

end

function y=sigmoid(W,x,b)
% Sigmoid function
    y=1./(1+exp(-(W*x+repmat(b,[1,size(x,2)]))));
end

function [hat_mt, mt] = momentum(mt_prev, grad, beta1)
    
    mt = beta1*mt_prev + (1-beta1)*grad;
    
    hat_mt = mt/(1 - beta1);
end

function [hat_vt, vt] = RMSP(vt_prev, grad, beta2)
    
    vt = beta2*vt_prev + (1-beta2)*grad.^2;
    
    hat_vt = vt/(1 - beta2);
end

function [WorB, mt, vt] = ADAM(WorB_prev, grad, mt_prev, vt_prev, beta1, beta2, epsilon, lr)
    
    [hat_mt, mt] = momentum(mt_prev, grad, beta1);
    [hat_vt, vt] = RMSP(vt_prev, grad, beta2);
    
    WorB = WorB_prev - hat_mt.*(lr./sqrt(hat_vt + epsilon));

end

function [W1_,b1_,W2_,b2_,mt_W1,mt_b1,mt_W2,mt_b2,vt_W1,vt_b1,vt_W2,vt_b2]= ...
    trainAE(x,z,x_hat,W1,b1,W2,b2,res,L,lr,lamda,mt_prev_W1,mt_prev_b1,mt_prev_W2,mt_prev_b2,vt_prev_W1,vt_prev_b1,vt_prev_W2,vt_prev_b2)
% Training the network by BP
    [n,num]=size(x);
    D=repmat(1./(2*res),n,1);
    grad=(x_hat-x).*D;
    delta1=grad.*x_hat.*(1-x_hat);
    delta2=z*(L+L');
    tmp=(W2'*delta1+lamda*delta2).*(z.*(1-z));
    
    epsilon = 0.00001;
    beta1 = 0.9;
    beta2 = 0.99;
    
    [W2_, mt_W2, vt_W2] = ADAM(W2, delta1*z'/num, mt_prev_W2, vt_prev_W2, beta1, beta2, epsilon, lr);
    [b2_, mt_b2, vt_b2] = ADAM(b2, delta1*ones(1,num)'/num, mt_prev_b2, vt_prev_b2, beta1, beta2, epsilon, lr);
    
%     [mt_W2, W2_] = momentum(W2, delta1*z'/num, mt_prev_W2, lr, beta);
%     [mt_b2, b2_] = momentum(b2, delta1*ones(1,num)'/num, mt_prev_b2, lr, beta);
%     W2_=W2-lr*delta1*z'/num;
%     b2_=b2-lr*delta1*ones(1,num)'/num;

    [W1_, mt_W1, vt_W1] = ADAM(W1, tmp*x'/num, mt_prev_W1, vt_prev_W1, beta1, beta2, epsilon, lr);
    [b1_, mt_b1, vt_b1] = ADAM(b1, tmp*ones(1,num)'/num, mt_prev_b1, vt_prev_b1, beta1, beta2, epsilon, lr);
    
%     [mt_W1, W1_] = momentum(W1, tmp*x'/num, mt_prev_W1, lr, beta);
%     [mt_b1, b1_] = momentum(b1, tmp*ones(1,num)'/num, mt_prev_b1, lr, beta);
%     W1_=W1-lr*tmp*x'/num;
%     b1_=b1-lr*tmp*ones(1,num)'/num;
end