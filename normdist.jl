using Distributions
using Plots

m = Normal(0.0, 1.0)
x = LinRange(-5,5,300)
y = pdf.(m, x)

plot(x, y, color="red")
