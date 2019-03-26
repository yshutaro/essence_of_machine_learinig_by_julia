module svm_hard
using LinearAlgebra

mutable struct SVC
    a_
    w_
    w0_
    function SVC()
        new(Nothing, Nothing, Nothing)
    end
end

function fit(s::SVC, X, y, selections=Nothing)
    println("size(X):", size(X))
    println("size(y):", size(y))
    a = zeros(size(X)[1])
    println("a: ", a)
    ay = 0
    ayx = zeros(size(X)[2])
    println("ayx: ", ayx)
    yx = y .* X
    println("yx: ", yx)
    count = 0
    while true
        println("##### ", count, " ######")
        #println("size(yx):", size(yx))
        #println("size(ayx):", size(ayx))
        #println("1 . -(yx * ayx):",1 .- (yx * ayx))
        ydf = y .* (1 .- (yx * ayx))
        println("ydf:", ydf)
        #println("size(ydf):", size(ydf))
        i = findfirst(ydf .== minimum(ydf[(y .< 0) .| (a .> 0)]))
        println("i  :",i)
        j = findfirst(ydf .== maximum(ydf[(y .> 0) .| (a .> 0)]))
        println("j  :",j)
        println("a:", a)
        println("y:", y)
        println("ydf[i]: ", ydf[i])
        println("ydf[j]: ", ydf[j])
        if ydf[i] >= ydf[j]
            break
        end
        println("y[i]:", y[i])
        println("y[j]:", y[j])
        println("a[i]:", a[i])
        println("a[j]:", a[j])
        ay2 = ay - y[i]*a[i] - y[j]*a[j]
        println("ay2 :", ay2)
        ayx2 = ayx .- y[i]*a[i].*X[i, :] .- y[j]*a[j].*X[j, :]
        println("ayx2 :", ayx2)
        println("X[i, :] :", X[i, :])
        println("X[j, :] :", X[j, :])
        println("X[i] :", X[i])
        println("X[j] :", X[j])
        ai = (1 - y[i]*y[j] .+ y[i] .* dot( (X[i, :] .- X[j, :]) , (X[j, :] .* ay2 .- ayx2) ) ) / sum((X[i, :] .- X[j, :]).^2)
        println("ai X[i] - X[j] :", X[i,:] .- X[j,:])
        println("ai (X[i] - X[j])^2 :", (X[i,:] .- X[j,:]).^2)
        println("ai sum((X[i] - X[j])^2) :", sum((X[i,:] - X[j,:]).^2))
        println("ai dot( (X[i, :] .- X[j, :]) , (X[j, :] .* ay2 .- ayx2)) :", dot( (X[i, :] .- X[j, :]) , (X[j, :] .* ay2 .- ayx2)))
        println("ai :", ai)
        ai = (ai < 0 ? 0 : ai)
        aj = (-ai * y[i] - ay2) * y[j]
        println("aj :", aj)
        if aj < 0
            aj = 0
            ai = (-aj * y[j] - ay2) * y[i]
        end

        ay = ay + y[i] * (ai - a[i]) + y[j] * (aj -a[j])
        ayx = ayx .+ y[i] * (ai - a[i]) .* X[i, :] + y[j] * (aj -a[j]) .* X[j, :]
        println("a[i]: ", a[i])
        if ai == a[i]
            break
        end
        a[i] = ai
        a[j] = aj
        count += 1
    end
    s.a_ = a
    ind = a .!= 0.
    println("ind:", ind)
    println("size(X[ind, :]):", size(X[ind, :]))
    println("X[ind, :]:", X[ind, :])
    println("a:", a)
    println("a[ind]:", a[ind])
    println("y[ind]:", y[ind])
    println("X[ind,:]:", X[ind, :])
    s.w_ = sum((a[ind] .* y[ind]) .* X[ind, :], dims=1)
    println("s.w_:", s.w_)
    s.w0_ = sum(y[ind] .- (X[ind, :] * (s.w_)')) / sum(ind)
    println("s.w0_:", s.w0_)
end

function predict(s::SVC, X)
    sign.(s.w0_ .+ X * (s.w_)')
end

end