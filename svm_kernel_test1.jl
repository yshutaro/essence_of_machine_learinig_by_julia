include("svm.jl")

using .svm
using Plots
using Random

Random.seed!(0)
X0 = randn(100, 2)
X1 = randn(100, 2) .+ [2.5 3]

y = vcat([1 for x in 1:100], [-1 for x in 1:100])

X = vcat(X0, X1)

model = svm.SVC()
svm.fit(model, X, y)

xmin = minimum(X[:,1])
xmax = maximum(X[:,1])
ymin = minimum(X[:,2])
ymax = maximum(X[:,2])

x_range = LinRange(xmin, xmax, 200)
y_range = LinRange(ymin, ymax, 200)
xmesh = repeat(x_range', outer=(length(y_range),1))
ymesh = repeat(y_range,  outer=(1,length(x_range)))

Z = reshape(svm.predict(model, hcat(vec(xmesh), vec(ymesh))), size(xmesh))

print("正しく分類できた数:", sum(svm.predict(model, X) .== y))

scatter(X0[:, 1], X0[:, 2], color="black", markershape=:star6, label="")
scatter!(X1[:, 1], X1[:, 2], color="black", markershape=:+, label="")
contour!(x_range, y_range, Z, levels=[0], color=:black)
