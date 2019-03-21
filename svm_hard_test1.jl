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