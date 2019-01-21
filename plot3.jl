using Plots

f(x) = x^2
g(x) = (x-2)^2

plot(f, color="red", label="f")
plot!(g, color="black", linestyle=:dash, label="g")
