using Plots
using Random

Random.seed!(0)
l = []
for i in 1:1000
  append!(l, sum(rand(1:6, 10)))
end

histogram(l, bins=20, color="gray")
