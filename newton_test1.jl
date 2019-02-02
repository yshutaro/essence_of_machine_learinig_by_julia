using .newton

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

solver = newton.Newton()
initials = [[1,1],[-1,-1],[1,-1]]
for x0 in initials
  sol = newton.solveNewton(solver, f, df, x0)
  println(sol)
end
