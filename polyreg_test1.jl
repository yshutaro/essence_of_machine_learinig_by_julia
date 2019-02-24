using .polyreg
using Random
using Plots

Random.seed!(0)

f(x) = 1 .+ 2 .* x

x = rand(10) * 10
y = f(x) .+ randn(10)

# 多項式回帰
model = polyreg.PolynomialRegression(10)
polyreg.fit(model, x, y)

plt_p_1 = scatter(x, y, color="black", ylim=(minimum(y) - 1, maximum(y) + 1), label="")
xx = LinRange(minimum(x), maximum(x), 300)
yy = [polyreg.predict(model, u) for u in xx]

plt_p_2 = plot!(plt_p_1, xx, yy, color="black", label="")
