module svm
using LinearAlgebra

mutable struct RBFKernel
    X
    σ2
    values_
    function RBFKernel(X, σ)
        new(X, σ^2, similar(Array{Float64}, (size(X)[1],size(X)[1])))
    end
end

value(obj::RBFKernel, i, j)  = exp((sum(-(obj.X[i, :] .- obj.X[j, :]).^2)) / (2*obj.σ2))

function eval(obj::RBFKernel, Z, s)
    XX = obj.X[s, :]
    X_Z = [sum((XX[i, :] .- Z[j, :]).^2) for i in 1:size(XX)[1], j in 1:size(Z)[1]]
    col, row = size(X_Z)
    if (col, row) == (0, 0)
        return []
    end
    exp.(-X_Z ./ (2*obj.σ2))
end

mutable struct SVC
   a_
   w0_
   y_
   kernel_
   C::Float64
   σ
   max_iter::Int
   function SVC()
       new(Nothing, Nothing, Nothing, Nothing, 1., 1, 10000)
   end
end

function fit(obj::SVC, X, y)
    a = zeros(size(X)[1])
    ay = 0
    kernel = RBFKernel(X, obj.σ)
    for i in 1:obj.max_iter
        s = a .!= 0.
        if isempty(a[s].*y[s]) || isempty(svm.eval(kernel, X, s))
            ydf = y
            i = findfirst(ydf .== minimum(ydf[(((a .> 0) .& (y .> 0)) .| ((a .< obj.C) .& (y .< 0)))]))
            j = findfirst(ydf .== maximum(ydf[(((a .> 0) .& (y .< 0)) .| ((a .< obj.C) .& (y .> 0)))]))
        else
            ydf = y .* (1 .- (y .* ((a[s].*y[s])' * svm.eval(kernel, X, s))'))
            i = findfirst(ydf .== minimum(ydf[(((a .> 0) .& (y .> 0)) .| ((a .< obj.C) .& (y .< 0)))]))
            j = findfirst(ydf .== maximum(ydf[(((a .> 0) .& (y .< 0)) .| ((a .< obj.C) .& (y .> 0)))]))
        end
        if ydf[i] >= ydf[j]
            break
        end

        ay2 = ay - y[i]*a[i] - y[j]*a[j]
        kii = svm.value(kernel, i, i)
        kij = svm.value(kernel, i, j)
        kjj = svm.value(kernel, j, j)
        s = a .!= 0.
        s[i] = false
        s[j] = false
        kxi = (svm.eval(kernel, reshape(X[i, :], 1, :), s))
        kxj = (svm.eval(kernel, reshape(X[j, :], 1, :), s))
        ai = (1 - y[i]*y[j] + y[i]*( (kij - kjj)*ay2 - sum( (a[s].*y[s].*(kxi .- kxj) == []) ? 0 : a[s].*y[s].*(kxi .- kxj) ) ) ) / (kii + kjj - 2*kij)
        if ai < 0
            ai = 0
        elseif ai > obj.C
            ai = obj.C
        end

        aj = (-ai*y[i] - ay2)*y[j]
        if aj < 0
            aj = 0
            ai = (-ai*y[j] - ay2)*y[i]
        elseif aj > obj.C
            aj = obj.C
            ai = (-ai*y[j] - ay2)*y[i]
        end
        ay = ay + y[i] * (ai - a[i]) + y[j] * (aj -a[j])
        if ai == a[i]
            break
        end
        a[i] = ai
        a[j] = aj
    end
    obj.a_ = a
    obj.y_ = y
    obj.kernel_ = kernel
    s = a .!= 0.
    obj.w0_ = sum(y[s] - ((a[s].*y[s])' * eval(kernel, X[s, :], s))') / sum(s)
end

function predict(obj::SVC, X)
    s = obj.a_ .!= 0.
    sign.(obj.w0_ .+ ((obj.a_[s].*obj.y_[s])' * eval(obj.kernel_, X, s))')
end

end
