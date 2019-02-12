using LinearAlgebra
using Plots

function reg1dim2(x, y)
  n = length(x)
  a = (dot(x, y) - sum(y) * sum(x) / n ) / (sum(x.^2) - (sum(x)^2) / n)
  b = (sum(y) -  a * sum(x)) / n
  return a, b
end

x = [1, 2, 4, 6, 7]
y = [1, 3, 3, 5, 4]
a, b = reg1dim2(x, y)

xmax = maximum(x)

scatter(x, y, color="black")
plot!([0, xmax], [b, a*xmax + b], color="black")
