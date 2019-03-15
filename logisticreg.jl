module logisticreg

using LinearAlgebra
using Random
using Statistics

THRESHMIN = 1e-10

sigmoid(x) = 1 ./ (1 .+ ℯ.^(-x))

mutable struct LogisticRegression
    tol
    max_iter
    random_seed
    w_
    function LogisticRegression(tol, max_iter=3, random_seed=0)
        new(tol, max_iter, random_seed, Nothing)
    end
end

function fit(s::LogisticRegression, X, y)
    Random.seed!(s.random_seed)
    s.w_ = randn(size(X)[2] + 1)
    println("size(s.w_):", size(s.w_))
    Xtil = hcat(ones(size(X)[1]), X)
    println("size(Xtil):", size(Xtil))
    diff = Inf
    w_prev = s.w_
    iter = 0
    while diff > s.tol && iter < s.max_iter
        yhat = sigmoid(Xtil * s.w_)
        println("size(yhat):", size(yhat))
        r = clamp.(yhat .* (1 .- yhat), THRESHMIN, Inf)
        println("size(r):", size(r))
        println("size(Xtil'):", size(Xtil'))
        XR = Xtil' .* r'
        println("size(XR):", size(XR))
        XRX = dot(Xtil' .* r', Xtil)
        println("size(XRX):", size(XRX))
        w_prev = s.w_
        println("size(Xtil * s.w_):", size(Xtil * s.w_))
        println("size(yhat -y) :", size(yhat -y))
        println("size(1 ./ r) :", size(1 ./ r))
        println("size(s.w_) :", size(s.w_))
        println("size(Xtil * s.w_) :", size(Xtil * s.w_))
        println("size dot((1 ./ r) , (yhat - y)):",size( dot((1 ./ r), (yhat - y) )))
        println("size( (Xtil * s.w_) - (1 ./ r) .* (yhat - y) ):",size( (Xtil * s.w_) .- (1 ./ r) .* (yhat - y) ))
        b = XR * (Xtil * s.w_) .- dot((1 ./ r) , (yhat - y)) 
        println("size(b):", size(b))
        s.w_ = XRX \ b
        diff = mean(abs.(w_prev - s.w_))
        iter = iter + 1
    end
end

function predict(s::LogisticRegression, X)
    Xtil = hcat(ones(size(X)[1]), X)
    yhat = sigmoid(Xtil * s.w_)
    return [ifelse(x .> .5, 1, 0) for x in yhat]
end

end