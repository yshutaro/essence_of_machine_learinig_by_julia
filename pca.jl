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
    println("size X:", size(X))
    println("min size X:", minimum(size(X)))
    xbar = mean(X, dims=1)
    println("xbar:", xbar)
    Y = X .- xbar
    S = Y' * Y
    linalg = pyimport("scipy.sparse.linalg")
    U, Σ, VT = linalg.svds(S, k=obj.n_components, tol=obj.tol, v0=v0)
    println("U size:",size(U))
    println("VT size:",size(VT))
    println("Σ size:",size(Σ))
    println("U :",(U))
    println("Σ:",Σ)
    println("VT :",(VT))
    println("reverse VT :", VT[end:-1:1, :])
    println("reverse VT size:", size(VT[end:-1:1, :]))
    obj.VT_ = VT[end:-1:1, :]
end

function transform(obj::PCA, X)
    println("size Obj.VT_", size(obj.VT_))
    (obj.VT_ * X')'
end

end
