include("polyreg.jl")
using .linearreg
using .polyreg
using Random
using Plots

f(x) = 1 ./ (1 .+ x)

function sample(n)
    x = rand(n) * 5
    y = f(x)
    return x, y
end

xx = collect(0:0.01:4.99)
Random.seed!(0)
y_poly_sum = zeros(length(xx))
y_poly_sum_sq = zeros(length(xx))
y_lin_sum = zeros(length(xx))
y_lin_sum_sq = zeros(length(xx))
y_true = f(xx)
n = 100000
for i in 0:n
    x, y = sample(5)
    poly = polyreg.PolynomialRegression(4)
    polyreg.fit(poly, x, y)
    lin = linearreg.LinearRegression()
    linearreg.fit(lin, x, y)
    y_poly = polyreg.predict(poly, xx)
    global y_poly_sum = y_poly_sum .+ y_poly
    global y_poly_sum_sq = y_poly_sum_sq + (y_poly .- y_true).^2
    y_lin = linearreg.predict(lin, reshape(xx, :, 1))
    global y_lin_sum = y_lin_sum + y_lin
    global y_lin_sum_sq = y_lin_sum_sq + (y_lin .- y_true).^2
end

ax1_1 = plot(xx, xx, fillrange=(0, (y_lin_sum / n .- y_true).^2), fillcolor=:black, label="bias", title="Linear reg.")
ax1 = plot!(xx, xx, fillrange=((y_lin_sum / n .- y_true).^2, y_lin_sum_sq / n),fillcolor=:blue, label="variance")

ax1_2 = plot(xx, xx, fillrange=(0, (y_poly_sum / n .- y_true).^2), fillcolor=:black, label="bias", title="Polynomial reg.")
ax2 = plot!(xx, xx, fillrange=((y_poly_sum / n .- y_true).^2, y_poly_sum_sq / n), fillcolor=:blue, label="variance")

plot(ax1, ax2, layout=(1,2))
