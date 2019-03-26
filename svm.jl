module svm
using LinearAlgebra

mutable struct RBFKernel
    σ2
    X
    values_
    function RBFKernel(X, σ)
        # TODO
        new(σ^2, X, Nothing)
    end
end

function value(s::RBFKernel, i, j)
    sum(exp(-(s.X[i, :] .- s.X[j, :]).^2)) / (2*s.σ2)
end

function eval(s::RBFKernel, Z, s)
        # TODO
end

mutable struct SVC
   C
   σ
   max_iter
   function SVC()
       new(C=1., σ=1, max_iter=10000)
   end
end

function fit(s::SVC, X, y, selections=None)
    a = zeros(size(X)[1])
    ay = 0
    kernel = RBFKernel(X, s.σ)
    for i in 1:s.max_iter
        s = a != 0.
        ydf = y * (1 - y * dot(a[s]*y[s], svm.eval(kernel, X, s)))
        i = findfirst(ydf .== minimum(ydf[((a .> 0) .& (y .> 0)) .| ((a .< s.C) .& (y .< 0))]))
        println("i :", i)
        j = findfirst(ydf .== maximum(ydf[((a .> 0) .& (y .< 0)) .| ((a .< s.C) .& (y .> 0))]))
        println("j :", j)
        if ydf[i] >= ydf[j]
            break
        end

        ay2 = ay - y[i]*a[i] - y[j]*a[j]
        println("ay2 :", ay2)
        kii = svm.value(kernel, i, i)
        kij = svm.value(kernel, i, j)
        kjj = svm.value(kernel, j, j)
        s = a ! 0.
        s[i] = false
        s[j] = false
        # TODO
        kxi = svm.eval(kernel, X[i, :], s)
    end
end

end