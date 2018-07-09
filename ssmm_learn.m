function [model,obj] = ssmm_learn(args,param)

%Setting the default parameter for the ssmm model
model.C = 1;
model.tau = 0.1;
model.rho = 1;
model.eps = 1e-5;

%Handle the model parameter if it is fixed advanced
args = strread(args, '%s');
for i = 1:2:size(args,1)
    if strcmp(args{i},'-C')
        model.C = str2num(args{i+1});
    elseif strcmp(args{i},'-tau')
        model.tau = str2num(args{i+1});
    elseif strcmp(args{i},'-rho')
        model.rho = str2num(args{i+1});
    else
        error('Unknown parameters');
    end
end
fprintf('C=%.2f, tau = %.2f, rho = %.2f\n',model.C,model.tau,model.rho);
clear i;

%Handle error of the number of samples and labels 
if(size(param.X,3)~=length(param.y))
    error('Number of samples should equal to the number of labels');
end

X = param.X;
y = param.y;

sz = [param.dim(1),param.dim(2),param.class];
model.W = zeros(sz);
model.S = zeros(sz);
model.xi = 0;
model.Lambda = zeros(sz);


iter = 1;
max_iter = 100;
iterFlag = 1;

% initialze for the most violated constraints
expandstep = max_iter;
fdiffs = zeros(param.dim(1),param.dim(2),param.class,expandstep);
margins = zeros(expandstep,1);
activeCons_num = 0;

while (iter <= max_iter && iterFlag)
    if ~mod(iter,5)
        fprintf('*');
    end
    
    %Update S first 
    model.S = shrinkage(model.rho*model.W - model.Lambda, model.tau)/(1+model.rho);
    %Update W then
    yhat = param.constraintFn(param,model.W,X);
    loss = param.lossFn(yhat,y)/size(X,3);
    fd_all = param.DelPsiFn(param,X,yhat,y); %fd is a 4d arrays
    fd = sum(fd_all,4)/size(X,3);
    clear fd_all;
    
    acc = sum(yhat == y)/length(y);
    w_fd = times(model.W,fd);
    cost = loss + sum(w_fd(:));
    tmp_nuc = 0;
    for cla = 1:param.class
        tmp_nuc = tmp_nuc + param.norm_nuc(model.W(:,:,cla));
    end
    obj(iter) = model.C*cost + 0.5*norm(model.W(:),2)^2+ model.tau*tmp_nuc;
    
    clear w_fd;

    if cost > model.xi + model.eps
        activeCons_num = activeCons_num + 1;
        fdiffs(:,:,:,activeCons_num) = fd;
        margins(activeCons_num) = loss;
        [W_opt, xi_opt] = ssmm_pegasos_w(fdiffs,margins,activeCons_num,model);
        model.W = W_opt;
        model.xi = xi_opt;  
    else
        fprintf('the stop iteration is %d \n',iter);
        iterFlag = false;
    end
    %Update Lambda
     model.Lambda = model.Lambda + model.rho*(model.S - model.W);
    iter = iter +1;
end
