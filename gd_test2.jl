using Plots
using .gd

function f(xx::Array)
  x = xx[1]
  y = xx[2]
  5 * x^2 - 6 * x*y + 3 * y^2 + 6*x - 6*y
end

function df(xx::Array)
  x = xx[1]
  y = xx[2]
  [10 * x - 6 * y + 6, -6 * x + 6 * y - 6]
end

xmin = -3
xmax = 3
ymin = -3
ymax = 3

algos = []
initial = [1, 1]
alphas = [0.1, 0.2]

for α in alphas
  algo = gd.GradientDescent()
  gd.solveGradientDescent(algo, f, df, initial, α)
  push!(algos , algo)
  println(algo.x_)
  println(algo.opt_)
end


xs = LinRange(-2, 2, 300)
ys = LinRange(-2, 2, 300)

h(x,y) = f([x, y]) # 配列の形に直す
zs = h.(xs', ys)
levels = [-3, -2.9, -2.8, -2.6, -2.4, -2.2, -2, -1, -1, 0, 1, 2, 3, 4]

plt1 = scatter(algos[1].path_x_, algos[1].path_y_, color="black", markersize=1)
plt2 = contour(zs, levels=levels, color="black", linestyle=:dash)
plot(plt1, plt2)
#plt3 = scatter(algos[2].path_x_, algos[2].path_y_, color="black", markersize=1)
#plt4 = contour(zs, levels=levels, color="black", linestyle=:dash)
#  plot(plt1, plt2, plt3, plt4)
#plot(plt3)
