using Plots

f(x, y) = x^2 + y^2 / 4

x = LinRange(-5, 5, 300)
y = LinRange(-5, 5, 300)

z = f.(x', y)

colors = [0.1, 0.3, 0.5, 0.7]
levels = [1, 2, 3, 4, 5]
contourf(z, fill=(true,cgrad(:grays,colors)), levels=levels)
