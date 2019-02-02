using .newton
using Plots

f1(x, y) = x^3 - 2*y
f2(x, y) = x^2 + y^2 -1

function f(xx)
  x = xx[1]
  y = xx[2]
  return [f1(x, y), f2(x, y)]
end

function df(xx)
  x = xx[1]
  y = xx[2]
  return [3*x^2 -2; 2*x 2*y]
end

xmin = -3
xmax = 3
ymin = -3
ymax = 3

x = LinRange(xmin, xmax, 200)
y = LinRange(ymin, ymax, 200)

z1 = f1.(x',y)
z2 = f2.(x',y)

solver = newton.Newton()
initials = [[1,1],[-1,-1],[1,-1]]
solvers = []
for x0 in initials
  sol = newton.solveNewton(solver, f, df, x0)
  push!(solvers, solver)
  println(sol)
end

contour(x, y, z1, color="red", levels=[0], xlim=(xmin, xmax), ylim=(ymin, ymax))
contour!(x, y, z2, color="black", levels=[0])

scatter!(solvers[1].path_x_, solvers[1].path_y_, color="black", markershape=:+)
scatter!(solvers[2].path_x_, solvers[2].path_y_, color="black", markershape=:star8)
scatter!(solvers[3].path_x_, solvers[3].path_y_, color="black", markershape=:x)

