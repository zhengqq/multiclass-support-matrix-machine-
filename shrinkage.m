function [D nuc rk] = shrinkage(X, tau)
    D = zeros(size(X));
    for i = 1:size(X,3)
        [U, S, V] = svd(X(:,:,i));
        s = max(0, S-tau);
        nuc(i) = sum(diag(s));
        D(:,:,i) = U *  s * V';
        rk(i) = sum(diag(s>0));
    end
end