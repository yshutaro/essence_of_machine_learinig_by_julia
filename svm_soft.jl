module svm_soft
using LinearAlgebra

mutable struct SVC
    a_
    w_
    w0_
    C
    function SVC(C=1.)
        new(Nothing, Nothing, Nothing, C)
    end
end

function fit(s::SVC, X, y, selections=Nothing)
    a = zeros(size(X)[1])
    ay = 0
    ayx = zeros(size(X)[2])
    yx = y .* X
    count=0
    while true
        println("############### ", count, " ##############")
        #println("yx :", yx)
        #println("ayx :", ayx)
        ydf = y .* (1 .- (yx * ayx))
        println("ydf :", ydf)
        #println("((a .> 0) .& (y .> 0)) .| ((a .< s.C) .& (y .< 0)) :", ((a .> 0) .& (y .> 0)) .| ((a .< s.C) .& (y .< 0)))
        println("find all i", findfirst(ydf .== minimum(ydf[((a .> 0) .& (y .> 0)) .| ((a .< s.C) .& (y .< 0))])))
        i = findfirst(ydf .== minimum(ydf[((a .> 0) .& (y .> 0)) .| ((a .< s.C) .& (y .< 0))]))
        println("i :", i)
        println("ydf[i] :", ydf[i])
        #println("((a .> 0) .& (y .< 0)) .| ((a .< s.C) .& (y .> 0)) :", ((a .> 0) .& (y .< 0)) .| ((a .< s.C) .& (y .> 0)))
        println("ydf true : ", ydf[((a .> 0) .& (y .< 0)) .| ((a .< s.C) .& (y .> 0))])
        println("ydf[18] : ", ydf[19])
        println("ydf[25] : ", ydf[25])
        println("find all j",findfirst(ydf .== maximum(ydf[((a .> 0) .& (y .< 0)) .| ((a .< s.C) .& (y .> 0))])))
        j = findfirst(ydf .== maximum(ydf[((a .> 0) .& (y .< 0)) .| ((a .< s.C) .& (y .> 0))]))
        println("j :", j)
        println("ydf[j] :", ydf[j])
        if ydf[i] >= ydf[j]
            break
        end
        println("ay :", ay)
        println("y[i] :", y[i])
        println("a[i] :", a[i])
        println("y[j] :", y[j])
        println("a[j] :", a[j])
        ay2 = ay - y[i]*a[i] - y[j]*a[j]
        println("ay2 :", ay2)
        ayx2 = ayx .- y[i]*a[i].*X[i, :] .- y[j]*a[j].*X[j, :]
        println("ayx2 :", ayx2)
        ai = (1 - y[i]*y[j] + y[i] * dot( (X[i, :] .- X[j, :]) , (X[j, :] .* ay2 .- ayx2) ) ) / sum((X[i, :] .- X[j, :]).^2)
        #println("X[i, :] :", X[i, :])
        #println("X[j, :] :", X[j, :])
        #println("sum((X[i, :] .- X[j, :]).^2): ", sum((X[i, :] .- X[j, :]).^2))
        println("ai 1:", ai)
        if ai < 0
            println("############ai<0#######")
            ai = 0
        elseif ai > s.C
            println("############ai>s.C#######")
            ai = s.C
        end
        println("ai 2:", ai)

        aj = (-ai * y[i] - ay2) * y[j]
        println("aj 1:", aj)
        if aj < 0
            println("############aj<0#######")
            aj = 0
            ai = (-aj * y[j] - ay2) * y[i]
        elseif aj > s.C
            println("############aj>s.C#######")
            aj = s.C
            ai = (-aj * y[j] - ay2) * y[i]
        end
        println("ai 3 :", ai)
        println("aj 2:", aj)

        println("ay before:", ay)
        println("y[i] :", y[i])
        println("a[i] :", a[i])
        println("a[j] :", a[j])
        ay = ay + y[i] * (ai - a[i]) + y[j] * (aj -a[j])
        println("ay: ", ay)
        println("ai - a[i]: ", ai - a[i])
        ayx = ayx .+ y[i] * (ai - a[i]) .* X[i, :] + y[j] * (aj -a[j]) .* X[j, :]
        println("ayx: ", ayx)
        if ai == a[i]
            break
        end
        a[i] = ai
        a[j] = aj
        count += 1
        if count > 100
            break
        end
    end
    s.a_ = a
    ind = a .!= 0.
    s.w_ = sum((a[ind] .* y[ind]) .* X[ind, :], dims=1)
    s.w0_ = sum(y[ind] .- (X[ind, :] * (s.w_)')) / sum(ind)
end

function predict(s::SVC, X)
    sign.(s.w0_ .+ X * (s.w_)')
end

end