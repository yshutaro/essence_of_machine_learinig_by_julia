module pca

using Random
using Statistics
using LinearAlgebra
using PyCall

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
    v0 = randn(obj.rng_, minimum(size(X)))
    xbar = mean(X, dims=1)
    Y = X .- xbar
    S = Y' * Y
    linalg = pyimport("scipy.sparse.linalg")
    U, Σ, VT = linalg.svds(S, k=obj.n_components, tol=obj.tol, v0=v0)
    obj.VT_ = VT[end:-1:1, :]
end

function transform(obj::PCA, X)
    (obj.VT_ * X')'
end

end
