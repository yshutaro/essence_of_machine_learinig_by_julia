module logisticreg

using LinearAlgebra
using Random

THRESHMIN = 1e-10

sigmoid(x) = 1 / (1 + ℯ^(-1))

struct LogisticRegression
    tol
    max_iter
    w_
    function LogisticRegression(tol, max_iter=3, random_seed=0)
        new(tol, max_iter, Nothing)
    end
end

function fit(s::LogisticRegression, X, y)
    s.w_ = randn(size(X)[1] + 1)
    Xtil = hcat(ones(size(X)[1]), X)
    diff = Inf
    w_prev = s.w_
    iter = 0
    while diff > s.tol && iter < s.max_iter
        yhat = sigmoid(dot(Xtil, s.w_))
        r = clamp.(yhat .* (1 .- yhat), THRESHMIN, Inf)
        XR = Xtil' * r
        XRX = dot(Xtil' * r, Xtil)
        w_prev = s.w_
        b = dot(XR, dot(Xtil, s.w_) - 1 / r * (yhat - y))
        s.w_ = XRX / b
        diff = mean(abs(w_prev - s.w_))
        iter = iter + 1
    end
end

function predict(s::LogisticRegression, X)
    Xtil = hcat(ones(size(X)[1]), X)
    yhat = sigmoid(dot(Xtil, s.w_))
end

end