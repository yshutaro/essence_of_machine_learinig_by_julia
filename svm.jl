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

value(obj::RBFKernel, i, j)  = sum(exp(-(obj.X[i, :] .- obj.X[j, :]).^2)) / (2*obj.σ2)

function eval(obj::RBFKernel, Z, s)
    #println("eval X:" , obj.X)
    println("eval X size:" , size(obj.X))
    println("eval Z size:" , size(Z))
    println("eval s size:" , size(s))
    println("obj.X[s, 1, :] ", obj.X[s, :] )
    exp(-(sum((obj.X[s, 1, :] .- Z[1, : , :]).^2,dims=2) / 2(obj.σ2)))
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
        println("a :",a)
        println("s :",s)
        println("size(y) :",size(y))
        println("size(a[s]) :",size(a[s]))
        println("size(y[s]) :",size(y[s]))
        println("size svm.eval(kernel, X, s)",size(svm.eval(kernel, X, s)))
        ydf = y * (1 - y * dot(a[s].*y[s], svm.eval(kernel, X, s)))
        i = findfirst(ydf .== minimum(ydf[((a .> 0) .& (y .> 0)) .| ((a .< obj.C) .& (y .< 0))]))
        println("i :", i)
        j = findfirst(ydf .== maximum(ydf[((a .> 0) .& (y .< 0)) .| ((a .< obj.C) .& (y .> 0))]))
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
        elseif ai > obj.C
            ai = obj.C
        end

        aj = (-ai*y[i] - ay2)*y[i]
        if aj < 0
            aj = 0
            ai = (-ai*y[i] - ay2)*y[i]
        elseif aj > obj.C
            aj = obj.C
            ai = (-ai*y[i] - ay2)*y[i]
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
    s = a != 0.
    obj.w0_ = sum(y[s] - dot(a[s]*y[s], eval(kernel, X[s], s))) / sum(s)
end

function predict(obj::SVC, X)
    s = obj.a_ != 0.
    sign(obj.w0_ + dot(obj.a_[s]*obj.y_[s], eval(obj.kernel_, X, s)))
end

end