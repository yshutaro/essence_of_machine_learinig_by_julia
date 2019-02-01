using Plots
using .gd #自作Moduleは.付きで呼び出す

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

algo = gd.GradientDescent()
initial = [1, 1]
gd.solveGradientDescent(algo, f, df, initial)

println(algo.x_)
println(algo.opt_)

plt0 = scatter([initial[1]], [initial[2]], color="black", markersize=5)
plt1 = scatter!(plt0, algo.path_x_, algo.path_y_, color="black", markersize=1)

xs = LinRange(-2, 2, 300)
ys = LinRange(-2, 2, 300)

h(x,y) = f([x, y]) # 配列の形に直す
zs = h.(xs', ys)
levels = [-3, -2.9, -2.8, -2.6, -2.4, -2.2, -2, -1, -1, 0, 1, 2, 3, 4]

plot(plt1)
contour!(xs, ys, zs, levels=levels, color="black", linestyle=:dash)
