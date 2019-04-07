include("svm_soft.jl")

using .svm_soft
using Plots
using Random

Random.seed!(0)
X0 = randn(20, 2)
X1 = randn(20, 2) .+ [2.5 3]

y = vcat([1 for x in 1:20], [-1 for x in 1:20])

X = vcat(X0, X1)

model = svm_soft.SVC()
svm_soft.fit(model, X, y)

scatter(X0[:, 1], X0[:, 2], color="black", markershape=:+, label="")
scatter!(X1[:, 1], X1[:, 2], color="black", markershape=:star6, label="")

f(model, x) = (-model.w0_ - model.w_[1] * x) / model.w_[2]

x1 = -2
x2 = 4
plot!([x1, x2], [f(model, x1), f(model, x2)], color="black", label="")
tf = model.a_ .!= 0
Xfalse = [false for x in 1:size(X)[1]]
print("正しく分類できた数:", sum(svm_soft.predict(model, X) .== y))
scatter!(X[hcat(tf, Xfalse)], X[hcat(Xfalse, tf)], color="red", markersize=10, markershape=:circle, markeralpha=0.1, label="")
