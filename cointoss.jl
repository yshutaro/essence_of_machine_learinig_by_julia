using Random
using Plots

function cointoss(n, m)
  l = []
  for i in 1:m
    r = rand(0:1, n)
    append!(l, sum(r))
  end
  return l
end

Random.seed!(0)

l = cointoss(100, 1000000)
plt1 = histogram(30:70, l, bins=50, color="black")

l = cointoss(10000, 1000000)
plt2 =histogram(4800:5200, l, bins=50, color="black")

plot(plt1, plt2, layout=(1,2))
