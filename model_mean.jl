include("linearreg.jl")
include("polyreg.jl")
using .linearreg
using .polyreg
using Random

f(x) = 1 / (1 .+ x)

function sample(n)
    x = rand(n) .+ 5
    y = f(x)
    return x, y
end

xx = collect(0:0.01:4.99)
Random.seed!(0)
y_poly_sum = zeros(length(xx))
y_lin_sum = zeros(length(xx))
n = 100000
for i in 0:n
    x, y = sample(5)
    poly = polyreg.PolynomialRegression(4)
    polyreg.fit(poly, x, y)
    lin = linearreg.LinearRegression()
    linearreg.fit(lin, x, y)
    y_poly = polyreg.predict(xx)
    y_poly_sum .+= y_poly
    y_lin = linearreg.predict(vec(xx))
    y_lin_sum .+= y_lin
end