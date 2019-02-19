using .ridge
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

model = ridge.RidgeRegression()
ridge.fit(model, X, y)

x_range = LinRange(0, scale, 20)
y_range = LinRange(0, scale, 20)
xmesh = repeat(x_range', outer=(length(y_range),1))
ymesh = repeat(y_range,  outer=(1,length(x_range)))
zmesh = reshape(model.w_[1] .+ model.w_[2] .* vec(xmesh) .+ model.w_[3] .* vec(ymesh), size(xmesh))

plot(x_range, y_range, zmesh, color="red", st=:wireframe)
scatter!(X[:,1], X[:,2], y, color="black")
