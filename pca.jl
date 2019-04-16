module pca

using Random
using Statistics
using LinearAlgebra

mutable struct PCA
    n_components
    tol
    rng_::MersenneTwister
    VT_
    function PCA(n_components)
        rng = MersenneTwister(0)
        new(n_components, 0.0, rng)
    end
end

function fit(obj::PCA, X)
    v0 = randn(obj.rng_, min(size(X)))
    xbar = mean(X, dims=1)
    Y = X .- xbar
    S = dot(Y', Y)
    F = svd(S)
    obj.VT_ = F.VT_
end

function transform(obj::PCA, X)
    dot(obj.VT_, X')'
end

end
