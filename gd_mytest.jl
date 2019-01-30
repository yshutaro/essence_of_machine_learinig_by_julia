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

x = [1, 1]

grad = df(x)
α = 0.01
ε = 1e-6

println("x init=", x)
println("grad init=", grad)

j = 1
while(sum(grad.^2)) > ε^2

  global x
  global grad

  println("##### ", j ,"#####")

  println("x before", x)
  println("α * grad=", α * grad)
  x = x - α * grad
  grad = df(x)
  global j = j + 1
  println("x after", x)
  println("grad=", grad)
  println("f(x)=", f(x))
end

println("----------")
println(x)
println(f(x))
