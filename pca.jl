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
    #v0 = randn(obj.rng_, min(size(X)))
    println("size X:", size(X))
    xbar = mean(X, dims=1)
    println("xbar:", xbar)
    Y = X .- xbar
    S = Y' * Y
    U, Σ, VT = svd(S)
    println("VT size:",size(VT))
    obj.VT_ = VT
end

function transform(obj::PCA, X)
    println("size Obj.VT_", size(obj.VT_))
    (obj.VT_ * X')'
end

end
