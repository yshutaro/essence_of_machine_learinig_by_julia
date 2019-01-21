using Plots

x = collect(LinRange(-5, 5, 300))
y = x.^2

plot(x, y, color="red")
