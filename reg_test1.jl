using .linearreg
using Random
using Plots

n = 100
scale = 10
Random.seed!(0)
X = rand(n, 2) .* scale
w0 = 1
w1 = 2
w2 = 3
y = w0 .+ w1 * X[:, 1] .+ w2 * X[:, 2] .+ randn(n)

model = linearreg.LinearRegression()
linearreg.fit(model, X, y)
println("係数:", model.w_)
println("(1, 1)に対する予測値:", linearreg.predict(model, [1,1]))

x = LinRange(0, 20, scale)
y = LinRange(0, 20, scale)
xmesh = repeat(x', outer=(length(y),1))
ymesh = repeat(y,  outer=(1,length(x)))
zmesh = reshape(model.w_[1] .+ model.w_[2] .+ vec(xmesh) .+ model.w_[3] .* vec(ymesh), size(xmesh))

plot(xmesh, ymesh, zmesh, color="black", st=:wireframe)
