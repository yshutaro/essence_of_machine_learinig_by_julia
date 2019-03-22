module svm_hard
#using LinearAlgebra

mutable struct SVC
    a
    w_
    w0_
    function SVC()
        new(0, Nothing, Nothing)
    end
end

function fit(s::SVC, X, y, selections=Nothing)
    println("size(X):", size(X))
    println("size(y):", size(y))
    a = zeros(size(X)[1])
    ay = 0
    ayx = zeros(size(X)[2])
    yx = y .* X
    indices = collect(1:size(X)[1])
    while true
        println("size(yx):", size(yx))
        println("size(ayx):", size(ayx))
        ydf = y .* (1 .- (yx * ayx))
        println("size(ydf):", size(ydf))
        println("size(indices):", size(indices))
        iydf = hcat(indices, ydf)
        println("size(iydf):", size(iydf))
        #println(size((y .< 0) .| (a .> 0)))
        println(iydf)
        println((y .< 0) .| (a .> 0))
        println(iydf[:, 2])
        println(iydf[:, 2][(y .< 0) .| (a .> 0)])
        println(size(iydf[:, 2][(y .< 0) .| (a .> 0)]))
        println("#####")
        i = minimum(iydf[:, 2][(y .< 0) .| (a .> 0)])
        j = maximum(iydf[:, 2][(y .< 0) .| (a .> 0)])
        if ydf[i] >= ydf[j]
            break
        end
        ay2 = ay - y[i]*a[i] - y[j]*a[j]
        ayx2 = ayx .- y[i]*a[i]*X[i, :] .- y[j]*a[j]*X[j, :]
        ai = (1 - y[i]*y[j] + y[i]* ( (X[i, :] - X[j, :]) * (X[j, :] * ay2 - ayx2) ) ) / sum((X[i] .- X[j]).^2)
        ai = (ai < 0 ? 0 : ai)
        aj = (-ai * y[i] - ay2) * y[j]
        if aj < 0
            aj = 0
            ai = (-aj * y[j] - ay2) * y[i]
        end

        ay = ay + y[i] * (ai - a[i]) + y[j] * (aj -a[j])
        ayx = ayx .+ y[i] * (ai - a[i]) .* X[i, :] + y[j] * (aj -a[j]) .* X[j, :]
        if ai == a[i]
            break
        end
        a[i] = ai
        a[j] = aj
    end
    s.a = a
    s.w_ = sum(reshape(a[ind] * y[ind], 1, :) .* X[ind, :])
    s.w0_ = sum(y[ind] - (X[ind, :] * s.w_)) / sum(ind)
end

function predict(s::SVC, X)
    sign(s.w0_ + X * s.w_)
end

end