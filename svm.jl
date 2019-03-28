module svm
using LinearAlgebra

mutable struct RBFKernel
    σ2
    X
    values_
    function RBFKernel(X, σ)
        new(σ^2, X, Nothing, similar(Array{Float64}, (size(X)[1],size(X)[1])))
    end
end

function value(s::RBFKernel, i, j)
    sum(exp(-(s.X[i, :] .- s.X[j, :]).^2)) / (2*s.σ2)
end

function eval(s::RBFKernel, Z, s)
    exp(-(sum((s.X[s, 1, :] .- Z[1, : , :]).^2,dims=2) / 2(s.σ2)))
end

mutable struct SVC
   a_
   w0_
   y_
   kernel_
   C
   σ
   max_iter
   function SVC()
       new(Nothing, Nothing, Nothing, Nothing, C=1., σ=1, max_iter=10000)
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
        s = a != 0.
        s[i] = false
        s[j] = false
        # TODO
        kxi = vec(svm.eval(kernel, X[i, :], s))
        kxj = vec(svm.eval(kernel, X[j, :], s))
        ai = (1 - y[i]*y[j] + y[i]*( (kij - kjj)*ay2 -sum(a[s]*y[s]*(kxi - kxj)) ) ) / (kii + kjj - 2*kij)
        if ai < 0
            ai = 0
        elseif ai > s.C
            ai = s.C
        end

        aj = (-ai*y[i] - ay2)*y[i]
        if aj < 0
            aj = 0
            ai = (-ai*y[i] - ay2)*y[i]
        elseif aj > s.C
            aj = s.C
            ai = (-ai*y[i] - ay2)*y[i]
        end
        ay = ay + y[i] * (ai - a[i]) + y[j] * (aj -a[j])
        if ai == a[i]
            break
        end
        a[i] = ai
        a[j] = aj
    end
    self.a_ = a
    self.y_ = y
    self.kernel_ = kernel
    s = a != 0.
    s.w0_ = sum(y[s] - dot(a[s]*y[s], eval(kernel, X[s], s))) / sum(s)
end

function predict(s::SVC, X)
    s = s.a_ != 0.
    sign(s.w0_ + dot(s.a_[s]*s.y_[s], eval(s.kernel_, X, s)))
end

end