function demo_multiclassSMM
% A demo for our method: multiclass support matrix machine



%% -------------Load Data--------------------
    load multiclassdata.mat
    % input: X: p X q X n 
    %        X_test: p X q X n_test
    %        y: n x 1 : {1,2,3,...}
    %        y_test: n_test x 1: {1,2,3,..}
%% ------------Parameter Setting-------------
    C = 1;
    tau = 0.5;
    rho = 0.1;    

    param.X = X;
    param.y = y;
    param.class = length(unique(param.y));
    param.dim = size(X(:,:,1));
    param.train_num = size(X,3);
    param.lossFn = @lossCB;
    param.featureFn = @featureCB;
    param.DelPsiFn = @DelPsiCB;
    param.constraintFn = @constrainCB;
    param.prediction = @predictionCB;
    param.norm_nuc = @norm_nuc;
     
    args = sprintf('-C %g -tau %g -rho %g',C,tau,rho);
    tic;
    [model,~] = ssmm_learn(args, param);
    time_train = toc;
    
    acc_train = param.prediction(param,model.W,X,y);
    kappa_train = (acc_train - 1/param.class)/(1-1/param.class);
    
    tic;
    [acc_test,y_test_hat] = param.prediction(param,model.W,X_test,y_test);
    time_test = toc;

    kappa_test = (acc_test - 1/param.class)/(1-1/param.class);
    fprintf('training kappa is %.4f \n', kappa_train);
    fprintf('testing kappa is %.4f \n\n',kappa_test);
    
    %calculate the error rate
    err = length(find(y_test ~= y_test_hat))/length(y_test)*100;
    %calculate for each MI task
    numClass = param.class;
    for i = 1:numClass
        tp(i) = length(find(y_test_hat == i & y_test == i));
        fp(i) = length(find(y_test_hat == i)) - tp(i);
        fn(i) = length(find(y_test == i)) - tp(i);
        precision_all(i) = tp(i)/(tp(i)+fp(i));
        recall_all(i) = tp(i)/(tp(i)+fn(i));
        acc_class(i) = length(find(y_test_hat == i & y_test == i))/length(find(y_test==i)); 
        kappa_class(i) = (acc_class(i) - 1/numClass)/(1-1/numClass);
    end
    fprintf('testing accuracy in each class %.3f,%.3f,%.3f %.3f\n',acc_class);
        %calculate the precision (sum tp)/(sum (tp+fp))
    precision = sum(precision_all)/numClass;
    %calculate the recall (sum tp)/(sum(tp+fn))
    recall = sum(recall_all)/numClass;
    %calculate the F1 score
    F1score = 2*precision*recall/(precision+recall);
    
    
    fprintf('%s\n',[' training time is: ' num2str(time_train)]); 
    fprintf('%s\n',[' testing time is: ' num2str(time_test)]);
    fprintf('%s\n',[' testing error rate(%) is: ' num2str(err)]);
    fprintf('%s\n',[' testing kappa value is: ' num2str(kappa_test)]);

    fprintf('%s\n',[' testing acc on each MI are  ' num2str(acc_class)]);
    fprintf('%s\n',[' testing kappa on each MI are' num2str(kappa_class)]);
        
    
    fprintf('%s\n',[' testing precision is: ' num2str(precision)]);
    fprintf('%s\n',[' testing recall is: ' num2str(recall)]);
    fprintf('%s\n\n\n',[' testing F1 score is: ' num2str(F1score)]);
end



    % ----- Calculate the loss cost ------------------
    function delta = lossCB(y, ybar)
      delta = sum(double(y ~= ybar)) ;
    end

    % ----- Calculate the most likely yhat_i ---------
    function yhat_fn = constrainCB(param,W,X)
         num = size(X,3);
         if num == 1
             f_tmp = zeros(param.class,1);
             for j = 1:param.class
                 tmp = times(W(:,:,j),X);
                 f_tmp(j) = sum(tmp(:));
             end
             [~,ind] = max(f_tmp);
             yhat_fn = ind;
         else 
             yhat_fn = zeros(num,1);
             for i = 1:num 
                f_tmp = zeros(param.class,1);
                for j = 1:param.class
                    tmp = times(W(:,:,j),X(:,:,i));
                    f_tmp(j) = sum(tmp(:));
                end
                [~,ind] = max(f_tmp);
                yhat_fn(i) = ind;
             end
         end
    end
    
    % ------ Calculate the feature tensor --------------
    function psi = featureCB(param,X,y)
        num = size(X,3);
        if num == 1
            psi = zeors(param.dim(1),param.dim(2),param.class);
            psi(:,:,y) = X;
        else
            psi = zeros(param.dim(1),param.dim(2),param.class,num);
            for i = 1:num 
                psi(:,:,y(i),i) = X(:,:,i);
            end
        end
    end
    
    % ------ Calculate the delta feature tensor ----------
    function del_psi = DelPsiCB(param,X,yhat,y)
        num = size(X,3);
        if num == 1
            del_psi = zeros(param.dim(1),param.dim(2),param.class);
            if yhat ~= y
                del_psi(:,:,yhat) = X;
                del_psi(:,:,y) = -X;
            end
%             del_psi = zeros(param.dim(1),param.dim(2),param.class);
%             del_psi(:,:,yhat) = X;
%             del_psi(:,:,y) = -X;
        else
            del_psi = zeros(param.dim(1),param.dim(2),param.class,num);
            for i = 1:num
                if yhat(i) ~= y(i) 
                    del_psi(:,:,yhat(i),i) = X(:,:,i);
                    del_psi(:,:,y(i),i) = -X(:,:,i);
                end
            end
            
        end
    end
    
    % ------- Calculate the yhat ---------------------------
    function [acc_rate,y_pred] = predictionCB(param,W,X,y)
        y_pred = param.constraintFn(param,W,X);
        acc_rate = length(find(y_pred == y))/length(y);
    end
    
     function z = norm_nuc(X)
        z = sum(svd(X));
     end
