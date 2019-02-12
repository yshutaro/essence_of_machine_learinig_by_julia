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
y = w0 .+ w1 * X[:, 1] .+ w2 * X[:, 2] .+ rand(n)

model = linearreg.LinearRegression()
linearreg.fit(model, X, y)
