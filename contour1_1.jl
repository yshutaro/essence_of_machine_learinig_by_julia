using Plots

f(x, y) = x^2 + y^2 / 4

x = LinRange(-5, 5, 300)
y = LinRange(-5, 5, 300)

z = f.(x', y)

contour(x, y, z, levels=[1,2,3,4,5])
