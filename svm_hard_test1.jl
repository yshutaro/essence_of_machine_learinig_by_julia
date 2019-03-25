include("svm_hard.jl")

using .svm_hard
using Plots
using Random

Random.seed!(0)
X0 = randn(20, 2)
println("size X0:",size(X0))
X1 = randn(20, 2) .+ [5 5]
println("size X1:",size(X1))
y = vcat([1 for x in 1:20], [-1 for x in 1:20])
println(y)

println(X0)
println(X1)
X = vcat(X0, X1)

model = svm_hard.SVC()
svm_hard.fit(model, X, y)

scatter(X0[:, 1], X0[:, 2], color="black", markershape=:+)
scatter!(X1[:, 1], X1[:, 2], color="black", markershape=:star6)

f(model, x) = (-model.w0_ - model.w_[1] * x) / model.w_[2]

x1 = -0.2
x2 = 6
#println("size(a_): ", size(model.a_))
println("[model.a_ .!= 0] :", [model.a_ .!= 0])
println("X: ", X)
println("size(X): ", size(X))
plot!([x1, x2], [f(model, x1), f(model, x2)], color="black")
tf = model.a_ .!= 0
Xfalse = [false for x in 1:size(X)[1]]
println(X[hcat(tf, Xfalse)])
scatter!(X[hcat(tf, Xfalse)], X[hcat(Xfalse, tf)], color="red", markersize=10, markershape=:circle, markeralpha=0.1)