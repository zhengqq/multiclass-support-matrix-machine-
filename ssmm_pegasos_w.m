function [W,xi] = ssmm_pegasos_w(fdiffs,margins,activeCons_num,model,inner_iter)

    if (nargin < 5)
        inner_iter = 3000;
    end
    sz_fd = size(fdiffs);
    W = model.W;
    for t = 1:inner_iter 
        subgradient = zeros(sz_fd(1),sz_fd(2),sz_fd(3));
        for k = 1:activeCons_num
            tmp = times(W,fdiffs(:,:,:,k));
            dis(k) = margins(k) + sum(tmp(:));
        end
        [mvc_val,mvc_idx] = max(dis);
        xi = max([0,mvc_val]);
        if mvc_val > 0
            subgradient = model.rho*(W-model.S) - model.Lambda + model.C*fdiffs(:,:,:,mvc_idx);
        else 
            break;
        end
%         eta_t = model.C/(model.rho*t);
        eta_t = 0.000001;
%         eta_t = 0.0000001;
        W = W - eta_t*subgradient;
    end 
end 