using LinearAlgebra
using Plots

reg1dim1(x, y) = dot(x, y) / sum(x.^2)

x = [1, 2, 4, 6, 7]
y = [1, 3, 3, 5, 4]
a = reg1dim1(x, y)

xmax = maximum(x)

scatter(x, y, color="black")
plot!([0, xmax], [0, a*xmax], color="black")
