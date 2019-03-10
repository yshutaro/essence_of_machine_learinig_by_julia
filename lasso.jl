module lasso
using LinearAlgebra

soft_thresholding(x, y) = sign(x) * max(abs(x) - y, 0)

mutable struct Lasso
    λ_
    tol
    max_iter
    w_
    function Lasso(λ_, tol = 0.0001, max_iter = 1000)
        new(λ_, tol, max_iter, Nothing)
    end
end

function fit(s::Lasso, X::Array, t)
    n = size(X)[1]
    if ndims(X) == 1
        d = 0
    else
        d = size(X)[2]
    end
    s.w_ = zeros(d + 1)
    avgl1 = 0.
    for i in collect(0:s.max_iter)
        avgl1_prev = avgl1
        _update(s, n, d, X, t)
        avgl1 = sum(abs.(s.w_)) / size(s.w_)[1]
        if abs(avgl1 - avgl1_prev) <= s.tol
            break
        end
    end
end

function _update(s::Lasso, n, d, X, t)
    s.w_[1] = sum(t .- X * s.w_[2:end]) ./ n
    w0vec = ones(n) .* s.w_[1]
    for k in collect(1:d)
        ww = s.w_[2:end]
        ww[k] = 0
        q = dot((t - w0vec - X * ww) ,X[:, k])
        r = dot(X[:, k], X[:, k])
        s.w_[k + 1] = soft_thresholding(q / r, s.λ_)
    end
end

function predict(s::Lasso, X)
    if ndims(X) == 1
        X = reshape(X, 1, :)
    end
    Xtil = hcat(ones(size(X)[1]), X)
    return Xtil * s.w_
end

end
