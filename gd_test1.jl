using Plots
using .gd #自作Moduleは.付きで呼び出す

function f(xx::Array)
  x = xx[1]
  y = xx[2]
  5*x^2 - 6*x*y + 3*y^2 + 6*x - 6*y
end

function df(xx::Array)
  x = xx[1]
  y = xx[2]
  [10*x - 6*x + 6, -6*x + 6*y - 6]
end

gdStruct = gd.GradientDescent()
initial = [1, 1]
gd.solveGradientDescent(gdStruct, f, df, initial)

#print(gdStruct.x_)
#print(gdStruct.opt_)
