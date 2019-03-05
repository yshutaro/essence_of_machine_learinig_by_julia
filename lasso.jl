module lasso

soft_thresholding(x, y) = sign(x) * max(abs(x) - y, 0)

mutable struct Lasso
    lambda_
    tol
    max_iter
    w_
    function Lasso(lambda_, tol = 0.0001, max_iter = 1000)
        new(lambda_, tol, max_iter, Nothing)
    end
end

function fit(s::Lasso, X::Array, t)
    n = shape(X)[1]
    if ndims(X) == 1
        d = 0
    else
        d = shape(X)[2]
    end
    s.w_ = zeros(d + 1)
    avgl1 = 0.
    for i in collect(0:s.max_iter)
        avgl1_prev = avgl1
    end
end

function _update(s::Lasso, n, d, X, t)
    s.w_[1] = sum(t - X * s.w_[1:]) / n
    w0vec = ones(n) .* w_[1]
    for k in collect(1:d)
        ww = self.w_[1:]
        ww[k] = 0
        q = (t - w0vec - X * ww) * X[:, k]
        r = X[:, k] * X[:, k]
        w_[k + 1] = soft_thresholding(q / r, s.lambda_)
    end
end

function predict(s::Lasso, X)
    if ndims(X) == 1
        X = reshape(X, 1, :)
    end
    Xtil = hcat(ones(size(X)[1]), X)
    return Xtil * s.w
end


end