using Plots

f(x) = sin(x)
g(x) = cos(x)

p1 = plot(f, color="red", label="sin(x)")
p2 = plot(g, color="black", label="cos(x)")
plot(p1, p2, layout=(2,1))
